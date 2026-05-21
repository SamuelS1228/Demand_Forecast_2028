import io
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="UIO Demand Forecast Model", layout="wide")

st.title("UIO Demand Forecast Model")
st.caption(
    "Forecasts demand using UIO, probabilistic VMT profiles, mileage-band incidence rates, "
    "mileage-band units per repair, location outputs, partial-year factors, and data quality logs."
)

REQUIRED_UIO_COLS = ["CBSA", "Forecast Year", "Vehicle Age", "Calibrated Retained UIO Rounded"]
REQUIRED_VMT_COLS = ["Annual VMT", "VMT Probability"]
PRODUCT_COL = "Product Category"


# -----------------------------
# Logging helpers
# -----------------------------
def add_log(logs: list, severity: str, step: str, message: str, count: int = 0, detail: str = ""):
    logs.append({
        "Severity": severity,
        "Step": step,
        "Message": message,
        "Count": int(count) if pd.notna(count) else 0,
        "Detail": detail
    })


# -----------------------------
# General helpers
# -----------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def normalize_probability(series: pd.Series) -> pd.Series:
    s = clean_numeric(series)
    if len(s.dropna()) and s.dropna().max() > 1:
        s = s / 100
    return s


def parse_mileage_band(label: str) -> Optional[Tuple[int, int]]:
    nums = re.findall(r"\d[\d,]*", str(label))
    if len(nums) < 2:
        return None
    return int(nums[0].replace(",", "")), int(nums[1].replace(",", ""))


def build_mileage_bands_from_headers(headers, logs: list) -> pd.DataFrame:
    rows = []
    skipped = []
    for h in headers:
        parsed = parse_mileage_band(h)
        if parsed:
            rows.append({"Min Miles": parsed[0], "Max Miles": parsed[1], "Mileage Band": str(h).strip()})
        else:
            skipped.append(str(h))

    if skipped:
        add_log(
            logs, "Warning", "Mileage Band Parsing",
            "Some non-mileage columns were ignored when parsing mileage bands.",
            len(skipped), ", ".join(skipped[:20])
        )

    bands = pd.DataFrame(rows).drop_duplicates().sort_values("Min Miles").reset_index(drop=True)
    if bands.empty:
        raise ValueError("No mileage band headers could be parsed. Expected headers like '0 To 9,999'.")
    return bands


def assign_mileage_band(miles: pd.Series, bands: pd.DataFrame, cap_to_max_band: bool, logs: list) -> pd.Series:
    bands = bands.sort_values("Min Miles").reset_index(drop=True)

    min_miles = bands["Min Miles"].iloc[0]
    max_miles = bands["Max Miles"].iloc[-1]
    min_band = bands["Mileage Band"].iloc[0]
    max_band = bands["Mileage Band"].iloc[-1]

    bins = list(bands["Min Miles"]) + [max_miles + 1]
    labels = list(bands["Mileage Band"])

    assigned = pd.cut(
        miles,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    ).astype("object")

    above_count = int((miles > max_miles).sum())
    below_count = int((miles < min_miles).sum())

    if cap_to_max_band:
        if above_count:
            add_log(
                logs, "Warning", "Mileage Band Assignment",
                f"Cumulative mileage exceeded the highest available band and was capped to '{max_band}'.",
                above_count,
                f"Highest band max miles: {max_miles:,}"
            )
        if below_count:
            add_log(
                logs, "Warning", "Mileage Band Assignment",
                f"Cumulative mileage was below the lowest available band and was capped to '{min_band}'.",
                below_count,
                f"Lowest band min miles: {min_miles:,}"
            )

        assigned = assigned.where(miles <= max_miles, max_band)
        assigned = assigned.where(miles >= min_miles, min_band)
        assigned = assigned.fillna(max_band)
    else:
        assigned = assigned.fillna("Above Max Mileage")
        if above_count:
            add_log(
                logs, "Error", "Mileage Band Assignment",
                "Some cumulative mileage values exceeded the highest available band.",
                above_count,
                "Turn on high-mileage capping or add higher mileage bands to incidence and units files."
            )

    return assigned


def clean_wide_rate_table(df: pd.DataFrame, table_name: str, logs: list) -> pd.DataFrame:
    df = df.copy()
    original_rows = len(df)
    df.columns = [str(c).strip() for c in df.columns]

    if PRODUCT_COL not in df.columns:
        raise ValueError(f"{table_name} is missing required column: {PRODUCT_COL}")

    product_clean = df[PRODUCT_COL].astype(str).str.strip()

    mask_repeated_header = product_clean.str.lower().eq(PRODUCT_COL.lower())
    mask_blank_product = df[PRODUCT_COL].isna() | product_clean.eq("") | product_clean.str.lower().isin(["nan", "none"])
    mask_drop = mask_repeated_header | mask_blank_product

    if mask_repeated_header.sum():
        add_log(
            logs, "Warning", f"{table_name} Cleanup",
            "Removed rows where Product Category was the repeated header text.",
            int(mask_repeated_header.sum())
        )

    if mask_blank_product.sum():
        add_log(
            logs, "Warning", f"{table_name} Cleanup",
            "Removed rows with blank/null Product Category.",
            int(mask_blank_product.sum())
        )

    df = df.loc[~mask_drop].copy()

    # Normalize product text
    df[PRODUCT_COL] = df[PRODUCT_COL].astype(str).str.strip()

    # Remove exact duplicate product rows after cleaning, keep first
    dup_count = int(df.duplicated(subset=[PRODUCT_COL], keep="first").sum())
    if dup_count:
        add_log(
            logs, "Warning", f"{table_name} Cleanup",
            "Removed duplicate Product Category rows, keeping the first occurrence.",
            dup_count
        )
        df = df.drop_duplicates(subset=[PRODUCT_COL], keep="first").copy()

    add_log(
        logs, "Info", f"{table_name} Cleanup",
        f"{table_name} cleaned.",
        original_rows - len(df),
        f"Original rows: {original_rows:,}; final rows: {len(df):,}"
    )

    return df


def normalize_rate_table(df: pd.DataFrame, value_name: str, logs: list) -> pd.DataFrame:
    df = df.copy()
    mileage_cols = [c for c in df.columns if c != PRODUCT_COL]

    long_df = df.melt(
        id_vars=[PRODUCT_COL],
        value_vars=mileage_cols,
        var_name="Mileage Band",
        value_name=value_name,
    )

    long_df[PRODUCT_COL] = long_df[PRODUCT_COL].astype(str).str.strip()
    long_df["Mileage Band"] = long_df["Mileage Band"].astype(str).str.strip()
    long_df[value_name] = clean_numeric(long_df[value_name])

    null_count = int(long_df[value_name].isna().sum())
    if null_count:
        add_log(
            logs, "Warning", f"{value_name} Cleanup",
            f"Rows with blank/non-numeric {value_name} were filled with 0.",
            null_count
        )
        long_df[value_name] = long_df[value_name].fillna(0)

    if value_name == "Incidence Rate" and len(long_df[value_name].dropna()) and long_df[value_name].dropna().max() > 1:
        add_log(
            logs, "Warning", "Incidence Rate Scaling",
            "Incidence values greater than 1 were detected, so incidence rates were divided by 100.",
            int((long_df[value_name] > 1).sum()),
            "This treats values like 2.5 as 2.5%, or 0.025."
        )
        long_df[value_name] = long_df[value_name] / 100

    return long_df


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def excel_bytes(sheets: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            sheet_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                ws.set_column(idx, idx, min(max(len(str(col)) + 2, 12), 48))
    return output.getvalue()


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def make_default_partial_years(uio_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if uio_df is None or uio_df.empty or "Forecast Year" not in uio_df.columns:
        return pd.DataFrame({"Forecast Year": [], "Demand Year Factor": []})

    years = (
        pd.to_numeric(uio_df["Forecast Year"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
    )

    return pd.DataFrame({
        "Forecast Year": years,
        "Demand Year Factor": 1.0
    })


def clean_year_factors(year_factor_df: pd.DataFrame, available_years: pd.Series, logs: list) -> pd.DataFrame:
    yf = year_factor_df.copy()
    yf.columns = [str(c).strip() for c in yf.columns]

    if "Forecast Year" not in yf.columns:
        raise ValueError("Partial-year table must include 'Forecast Year'.")
    if "Demand Year Factor" not in yf.columns:
        raise ValueError("Partial-year table must include 'Demand Year Factor'.")

    yf["Forecast Year"] = clean_numeric(yf["Forecast Year"]).astype("Int64")
    yf["Demand Year Factor"] = clean_numeric(yf["Demand Year Factor"])

    if len(yf["Demand Year Factor"].dropna()) and yf["Demand Year Factor"].dropna().max() > 1:
        add_log(
            logs, "Warning", "Partial-Year Factors",
            "Demand Year Factor values greater than 1 were detected and divided by 100.",
            int((yf["Demand Year Factor"] > 1).sum()),
            "This treats 50 as 50%, or 0.50."
        )
        yf["Demand Year Factor"] = yf["Demand Year Factor"] / 100

    null_years = int(yf["Forecast Year"].isna().sum())
    if null_years:
        add_log(logs, "Warning", "Partial-Year Factors", "Removed rows with blank Forecast Year.", null_years)

    yf = yf.dropna(subset=["Forecast Year"])
    yf["Forecast Year"] = yf["Forecast Year"].astype(int)
    yf["Demand Year Factor"] = yf["Demand Year Factor"].fillna(1.0).clip(lower=0.0, upper=1.0)

    base = pd.DataFrame({"Forecast Year": sorted(pd.Series(available_years).dropna().astype(int).unique())})
    out = base.merge(yf[["Forecast Year", "Demand Year Factor"]], on="Forecast Year", how="left")

    missing_count = int(out["Demand Year Factor"].isna().sum())
    if missing_count:
        add_log(
            logs, "Info", "Partial-Year Factors",
            "Some forecast years had no demand factor and defaulted to 1.00.",
            missing_count
        )

    out["Demand Year Factor"] = out["Demand Year Factor"].fillna(1.0)
    return out


@st.cache_data(show_spinner=False)
def run_forecast(
    uio_df: pd.DataFrame,
    incidence_wide_raw: pd.DataFrame,
    units_wide_raw: pd.DataFrame,
    vmt_df: pd.DataFrame,
    year_factor_df: pd.DataFrame,
    age0_factor: float,
    make_location_product_output: bool,
    make_location_age_output: bool,
    cap_to_max_band: bool,
    strict_missing_rates: bool,
):
    logs = []

    # -----------------------------
    # Clean UIO
    # -----------------------------
    uio = uio_df.copy()
    uio.columns = [str(c).strip() for c in uio.columns]

    missing = [c for c in REQUIRED_UIO_COLS if c not in uio.columns]
    if missing:
        raise ValueError(f"UIO file is missing required columns: {missing}")

    original_uio_rows = len(uio)

    uio["CBSA"] = uio["CBSA"].astype(str).str.strip()
    uio["Forecast Year"] = clean_numeric(uio["Forecast Year"]).astype("Int64")
    uio["Vehicle Age"] = clean_numeric(uio["Vehicle Age"]).astype("Int64")
    uio["UIO"] = clean_numeric(uio["Calibrated Retained UIO Rounded"]).fillna(0)

    bad_uio_rows = int(uio["Forecast Year"].isna().sum() + uio["Vehicle Age"].isna().sum())
    uio = uio.dropna(subset=["Forecast Year", "Vehicle Age"])
    uio["Forecast Year"] = uio["Forecast Year"].astype(int)
    uio["Vehicle Age"] = uio["Vehicle Age"].astype(int)

    if bad_uio_rows:
        add_log(logs, "Warning", "UIO Cleanup", "Removed UIO rows with missing Forecast Year or Vehicle Age.", bad_uio_rows)

    add_log(logs, "Info", "UIO Cleanup", "UIO data loaded and cleaned.", original_uio_rows - len(uio), f"Final UIO rows: {len(uio):,}")

    # -----------------------------
    # Partial-year factors
    # -----------------------------
    year_factors = clean_year_factors(year_factor_df, uio["Forecast Year"], logs)
    uio = uio.merge(year_factors, on="Forecast Year", how="left")
    uio["Demand Year Factor"] = uio["Demand Year Factor"].fillna(1.0)

    # -----------------------------
    # Clean VMT
    # -----------------------------
    vmt = vmt_df.copy()
    vmt.columns = [str(c).strip() for c in vmt.columns]
    missing = [c for c in REQUIRED_VMT_COLS if c not in vmt.columns]
    if missing:
        raise ValueError(f"VMT file is missing required columns: {missing}")

    original_vmt_rows = len(vmt)
    vmt["Annual VMT"] = clean_numeric(vmt["Annual VMT"])
    vmt["VMT Probability"] = normalize_probability(vmt["VMT Probability"])
    vmt = vmt.dropna(subset=["Annual VMT", "VMT Probability"])

    if original_vmt_rows - len(vmt):
        add_log(logs, "Warning", "VMT Cleanup", "Removed rows with blank/non-numeric Annual VMT or VMT Probability.", original_vmt_rows - len(vmt))

    if not np.isclose(vmt["VMT Probability"].sum(), 1.0, atol=0.0001):
        raise ValueError(f"VMT probabilities must sum to 100%. Current sum: {vmt['VMT Probability'].sum():.2%}")

    # -----------------------------
    # Clean rate tables
    # -----------------------------
    incidence_wide = clean_wide_rate_table(incidence_wide_raw, "Incidence", logs)
    units_wide = clean_wide_rate_table(units_wide_raw, "Units Per Repair", logs)

    incidence_headers = [c for c in incidence_wide.columns if c != PRODUCT_COL]
    unit_headers = [c for c in units_wide.columns if c != PRODUCT_COL]

    missing_unit_headers = sorted(set(incidence_headers) - set(unit_headers))
    extra_unit_headers = sorted(set(unit_headers) - set(incidence_headers))

    if missing_unit_headers:
        add_log(
            logs, "Error" if strict_missing_rates else "Warning",
            "Header Alignment",
            "Units table is missing mileage-band headers found in incidence table.",
            len(missing_unit_headers),
            ", ".join(missing_unit_headers[:25])
        )

    if extra_unit_headers:
        add_log(
            logs, "Warning",
            "Header Alignment",
            "Units table has mileage-band headers not found in incidence table. These will not drive mileage-band assignment.",
            len(extra_unit_headers),
            ", ".join(extra_unit_headers[:25])
        )

    bands = build_mileage_bands_from_headers(incidence_headers, logs)

    inc_long = normalize_rate_table(incidence_wide, "Incidence Rate", logs)
    units_long = normalize_rate_table(units_wide, "Units Per Repair", logs)

    # Product category alignment
    incidence_products = set(incidence_wide[PRODUCT_COL].astype(str).str.strip())
    unit_products = set(units_wide[PRODUCT_COL].astype(str).str.strip())

    missing_in_units = sorted(incidence_products - unit_products)
    extra_in_units = sorted(unit_products - incidence_products)

    if missing_in_units:
        add_log(
            logs, "Error" if strict_missing_rates else "Warning",
            "Product Category Alignment",
            "Product categories exist in incidence file but are missing from units file.",
            len(missing_in_units),
            ", ".join(missing_in_units[:50])
        )

    if extra_in_units:
        add_log(
            logs, "Warning",
            "Product Category Alignment",
            "Product categories exist in units file but are not in incidence file. These will be excluded.",
            len(extra_in_units),
            ", ".join(extra_in_units[:50])
        )

    # Products are driven by incidence table only, excluding categories without units if not strict.
    valid_products = sorted(incidence_products)
    if not strict_missing_rates and missing_in_units:
        valid_products = sorted(incidence_products & unit_products)
        add_log(
            logs, "Warning",
            "Product Category Exclusion",
            "Excluded product categories that were missing from the units file.",
            len(missing_in_units),
            ", ".join(missing_in_units[:50])
        )

    if strict_missing_rates and (missing_in_units or missing_unit_headers):
        log_df = pd.DataFrame(logs)
        raise ValueError(
            "Critical input alignment issue found. Turn off strict missing-rate handling to exclude invalid categories, "
            "or fix the input files. See validation log."
        )

    # -----------------------------
    # Age/VMT bridge
    # -----------------------------
    ages = pd.DataFrame({"Vehicle Age": sorted(uio["Vehicle Age"].unique())})
    ages["_key"] = 1
    vmt["_key"] = 1
    age_vmt = ages.merge(vmt, on="_key").drop(columns="_key")

    age_vmt["Cumulative Miles"] = np.where(
        age_vmt["Vehicle Age"] == 0,
        age_vmt["Annual VMT"] * age0_factor,
        (age_vmt["Vehicle Age"] + age0_factor) * age_vmt["Annual VMT"],
    )

    max_band_miles = bands["Max Miles"].max()
    max_band_label = bands.loc[bands["Max Miles"].idxmax(), "Mileage Band"]
    age_vmt["Above Max Band Flag"] = age_vmt["Cumulative Miles"] > max_band_miles

    age_vmt["Mileage Band"] = assign_mileage_band(
        age_vmt["Cumulative Miles"],
        bands,
        cap_to_max_band=cap_to_max_band,
        logs=logs
    )

    products = pd.DataFrame({PRODUCT_COL: valid_products})
    age_vmt["_key"] = 1
    products["_key"] = 1
    bridge = age_vmt.merge(products, on="_key").drop(columns="_key")

    bridge = bridge.merge(inc_long, on=[PRODUCT_COL, "Mileage Band"], how="left")
    bridge = bridge.merge(units_long, on=[PRODUCT_COL, "Mileage Band"], how="left")

    missing_mask = bridge["Incidence Rate"].isna() | bridge["Units Per Repair"].isna()
    missing_detail = (
        bridge.loc[missing_mask, [PRODUCT_COL, "Mileage Band"]]
        .drop_duplicates()
        .sort_values([PRODUCT_COL, "Mileage Band"])
    )

    if len(missing_detail):
        add_log(
            logs,
            "Error" if strict_missing_rates else "Warning",
            "Rate Join",
            "Some product/mileage-band combinations were missing incidence or units values.",
            len(missing_detail),
            missing_detail.head(50).to_string(index=False)
        )

        if strict_missing_rates:
            raise ValueError(
                "Missing rate data found. See validation log for unique missing Product Category / Mileage Band combinations."
            )

        # Exclude missing combos from weighted rate calc
        before = len(bridge)
        bridge = bridge.loc[~missing_mask].copy()
        add_log(
            logs, "Warning", "Rate Join",
            "Excluded rows with missing incidence or units from weighted rate calculation.",
            before - len(bridge)
        )

    bridge["Weighted Incident Rate Component"] = bridge["VMT Probability"] * bridge["Incidence Rate"]
    bridge["Weighted Piece Rate Component"] = bridge["VMT Probability"] * bridge["Incidence Rate"] * bridge["Units Per Repair"]

    age_product_rates = (
        bridge.groupby(["Vehicle Age", PRODUCT_COL], as_index=False)
        .agg(
            Weighted_Incident_Rate=("Weighted Incident Rate Component", "sum"),
            Weighted_Piece_Rate=("Weighted Piece Rate Component", "sum"),
        )
    )

    # Any age/product combos completely missing after exclusions?
    expected_age_products = len(ages) * len(valid_products)
    missing_age_product_count = expected_age_products - len(age_product_rates)
    if missing_age_product_count:
        add_log(
            logs, "Warning", "Age/Product Rates",
            "Some age/product rate combinations could not be calculated after exclusions.",
            missing_age_product_count
        )

    # -----------------------------
    # Demand by year/product
    # -----------------------------
    uio_year_age = (
        uio.groupby(["Forecast Year", "Vehicle Age", "Demand Year Factor"], as_index=False)
        .agg(UIO=("UIO", "sum"))
    )

    year_product = uio_year_age.merge(age_product_rates, on="Vehicle Age", how="left")
    year_product["Full-Year Forecast Incidents"] = year_product["UIO"] * year_product["Weighted_Incident_Rate"]
    year_product["Full-Year Forecast Piece Demand"] = year_product["UIO"] * year_product["Weighted_Piece_Rate"]
    year_product["Forecast Incidents"] = year_product["Full-Year Forecast Incidents"] * year_product["Demand Year Factor"]
    year_product["Forecast Piece Demand"] = year_product["Full-Year Forecast Piece Demand"] * year_product["Demand Year Factor"]

    demand_by_year_product = (
        year_product.groupby(["Forecast Year", PRODUCT_COL], as_index=False)
        .agg(
            Demand_Year_Factor=("Demand Year Factor", "max"),
            Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
            Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )
    )

    # -----------------------------
    # Demand by location
    # -----------------------------
    uio_location_year_age = (
        uio.groupby(["CBSA", "Forecast Year", "Vehicle Age", "Demand Year Factor"], as_index=False)
        .agg(UIO=("UIO", "sum"))
    )

    location_year_age_product = uio_location_year_age.merge(age_product_rates, on="Vehicle Age", how="left")
    location_year_age_product["Full-Year Forecast Incidents"] = (
        location_year_age_product["UIO"] * location_year_age_product["Weighted_Incident_Rate"]
    )
    location_year_age_product["Full-Year Forecast Piece Demand"] = (
        location_year_age_product["UIO"] * location_year_age_product["Weighted_Piece_Rate"]
    )
    location_year_age_product["Forecast Incidents"] = (
        location_year_age_product["Full-Year Forecast Incidents"] * location_year_age_product["Demand Year Factor"]
    )
    location_year_age_product["Forecast Piece Demand"] = (
        location_year_age_product["Full-Year Forecast Piece Demand"] * location_year_age_product["Demand Year Factor"]
    )

    demand_by_location = (
        location_year_age_product.groupby(["CBSA", "Forecast Year"], as_index=False)
        .agg(
            Demand_Year_Factor=("Demand Year Factor", "max"),
            Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
            Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )
    )

    demand_by_location_product = None
    if make_location_product_output:
        demand_by_location_product = (
            location_year_age_product.groupby(["CBSA", "Forecast Year", PRODUCT_COL], as_index=False)
            .agg(
                Demand_Year_Factor=("Demand Year Factor", "max"),
                Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
                Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
                Forecast_Incidents=("Forecast Incidents", "sum"),
                Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
            )
        )

    demand_by_location_age = None
    if make_location_age_output:
        demand_by_location_age = (
            location_year_age_product.groupby(["CBSA", "Forecast Year", "Vehicle Age"], as_index=False)
            .agg(
                Demand_Year_Factor=("Demand Year Factor", "max"),
                Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
                Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
                Forecast_Incidents=("Forecast Incidents", "sum"),
                Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
            )
        )

    # Demand by age/product
    age_product_detail = uio[["Forecast Year", "Vehicle Age", "UIO", "Demand Year Factor"]].merge(
        age_product_rates,
        on="Vehicle Age",
        how="left"
    )
    age_product_detail["Full-Year Forecast Incidents"] = age_product_detail["UIO"] * age_product_detail["Weighted_Incident_Rate"]
    age_product_detail["Full-Year Forecast Piece Demand"] = age_product_detail["UIO"] * age_product_detail["Weighted_Piece_Rate"]
    age_product_detail["Forecast Incidents"] = age_product_detail["Full-Year Forecast Incidents"] * age_product_detail["Demand Year Factor"]
    age_product_detail["Forecast Piece Demand"] = age_product_detail["Full-Year Forecast Piece Demand"] * age_product_detail["Demand Year Factor"]

    demand_by_age_product = (
        age_product_detail.groupby(["Vehicle Age", PRODUCT_COL], as_index=False)
        .agg(
            Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
            Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )
    )

    validation_log = pd.DataFrame(logs)

    diagnostics = {
        "uio_rows": len(uio),
        "location_count": uio["CBSA"].nunique(),
        "input_product_count": len(incidence_products),
        "modeled_product_count": len(valid_products),
        "excluded_product_count": len(incidence_products) - len(valid_products),
        "age_product_rate_rows": len(age_product_rates),
        "total_piece_demand": demand_by_year_product["Forecast_Piece_Demand"].sum(),
        "full_year_piece_demand": demand_by_year_product["Full_Year_Forecast_Piece_Demand"].sum(),
        "above_max_age_vmt_rows": int(age_vmt["Above Max Band Flag"].sum()),
        "max_mileage_band": max_band_label,
        "max_mileage_band_miles": int(max_band_miles),
        "partial_years_count": int((year_factors["Demand Year Factor"] < 1).sum()),
        "warnings": int((validation_log["Severity"] == "Warning").sum()) if not validation_log.empty else 0,
        "errors": int((validation_log["Severity"] == "Error").sum()) if not validation_log.empty else 0,
    }

    return {
        "diagnostics": diagnostics,
        "validation_log": validation_log,
        "year_factors": year_factors,
        "age_vmt_bridge": age_vmt.drop(columns=[c for c in ["_key"] if c in age_vmt.columns]),
        "age_product_rates": age_product_rates,
        "demand_by_year_product": demand_by_year_product,
        "demand_by_location": demand_by_location,
        "demand_by_location_product": demand_by_location_product,
        "demand_by_location_age": demand_by_location_age,
        "demand_by_age_product": demand_by_age_product,
    }


# -----------------------------
# UI
# -----------------------------
st.subheader("1. Upload files")

c1, c2 = st.columns(2)
with c1:
    uio_upload = st.file_uploader("UIO_Age_Summary.csv", type=["csv"])
    incidence_upload = st.file_uploader("Incidence_By_Mileage.csv", type=["csv"])
with c2:
    units_upload = st.file_uploader("Units_Per_Repair.csv", type=["csv"])
    vmt_upload = st.file_uploader("VMT_Distribution.csv optional", type=["csv"])

default_vmt = pd.DataFrame(
    {"Annual VMT": [8000, 10000, 12000, 15000], "VMT Probability": [0.15, 0.25, 0.40, 0.20]}
)

try:
    vmt_df = read_csv_cached(vmt_upload.getvalue()) if vmt_upload else default_vmt
except Exception as e:
    st.error(f"Could not read VMT file: {e}")
    st.stop()

uio_preview = None
if uio_upload:
    try:
        uio_preview = read_csv_cached(uio_upload.getvalue())
    except Exception as e:
        st.error(f"Could not read UIO file: {e}")
        st.stop()

st.subheader("2. VMT assumptions")
st.write("Edit the assumptions below. The probabilities must sum to 100%.")

edited_vmt = st.data_editor(
    vmt_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Annual VMT": st.column_config.NumberColumn("Annual VMT", min_value=0, step=500),
        "VMT Probability": st.column_config.NumberColumn("VMT Probability", min_value=0.0, step=0.01, format="%.2f"),
    },
)

edited_vmt["Annual VMT"] = clean_numeric(edited_vmt["Annual VMT"])
edited_vmt["VMT Probability"] = normalize_probability(edited_vmt["VMT Probability"])
prob_sum = edited_vmt["VMT Probability"].sum()

st.write(f"Probability sum: **{prob_sum:.2%}**")

st.subheader("3. Partial-year demand factors")
st.write(
    "Set a factor below 1.0 for any forecast year where you only want part of the year. "
    "Example: 0.50 = half-year. This scales demand only, not cumulative mileage."
)

default_year_factors = make_default_partial_years(uio_preview)

if default_year_factors.empty:
    st.info("Upload the UIO file to populate forecast years for partial-year factors.")
    edited_year_factors = pd.DataFrame({"Forecast Year": [], "Demand Year Factor": []})
else:
    edited_year_factors = st.data_editor(
        default_year_factors,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Forecast Year": st.column_config.NumberColumn("Forecast Year", step=1, disabled=True),
            "Demand Year Factor": st.column_config.NumberColumn(
                "Demand Year Factor",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                help="1.0 = full year, 0.5 = half year, 0.25 = quarter year."
            ),
        },
        key="partial_year_editor"
    )

st.sidebar.header("Settings")
age0_factor = st.sidebar.slider("Age 0 mileage factor", 0.0, 1.0, 0.5, 0.05)

cap_to_max_band = st.sidebar.checkbox(
    "Cap mileage above max band to highest band",
    value=True,
    help="Recommended. If cumulative mileage exceeds your highest rate band, use the highest available band."
)

strict_missing_rates = st.sidebar.checkbox(
    "Strict missing-rate handling",
    value=False,
    help="If on, the model stops when categories or mileage-band rates are missing. If off, invalid/missing items are excluded and logged."
)

make_location_product_output = st.sidebar.checkbox(
    "Create location/product output",
    value=True
)

make_location_age_output = st.sidebar.checkbox(
    "Create location/age output",
    value=False
)

if not np.isclose(prob_sum, 1.0, atol=0.0001):
    st.warning("Fix the VMT probabilities before running.")
    st.stop()

st.subheader("4. Run model")

if not all([uio_upload, incidence_upload, units_upload]):
    st.info("Upload the UIO, incidence, and units-per-repair files to run.")
    st.stop()

if st.button("Run demand forecast", type="primary"):
    try:
        with st.spinner("Running forecast and validation checks..."):
            uio_df = read_csv_cached(uio_upload.getvalue())
            incidence_df = read_csv_cached(incidence_upload.getvalue())
            units_df = read_csv_cached(units_upload.getvalue())

            result = run_forecast(
                uio_df=uio_df,
                incidence_wide_raw=incidence_df,
                units_wide_raw=units_df,
                vmt_df=edited_vmt,
                year_factor_df=edited_year_factors,
                age0_factor=age0_factor,
                make_location_product_output=make_location_product_output,
                make_location_age_output=make_location_age_output,
                cap_to_max_band=cap_to_max_band,
                strict_missing_rates=strict_missing_rates,
            )

        st.success("Forecast complete.")

        d = result["diagnostics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("UIO Rows", f"{d['uio_rows']:,}")
        m2.metric("Locations", f"{d['location_count']:,}")
        m3.metric("Modeled Products", f"{d['modeled_product_count']:,}")
        m4.metric("Excluded Products", f"{d['excluded_product_count']:,}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Full-Year Piece Demand", f"{d['full_year_piece_demand']:,.0f}")
        m6.metric("Adjusted Piece Demand", f"{d['total_piece_demand']:,.0f}")
        m7.metric("Warnings", f"{d['warnings']:,}")
        m8.metric("Errors Logged", f"{d['errors']:,}")

        if d["warnings"] or d["errors"]:
            st.warning("Review the Validation Log tab before using the results.")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Validation Log",
            "Year/Product Demand",
            "Location Demand",
            "Partial-Year Factors",
            "Age/Product Rates",
            "Other Outputs"
        ])

        with tab1:
            st.subheader("Validation Log")
            st.caption("Shows removed rows, excluded categories, capped mileage outcomes, missing-rate issues, and other data quality events.")
            st.dataframe(result["validation_log"], use_container_width=True, height=520)

        with tab2:
            st.subheader("Demand by Forecast Year and Product Category")
            st.caption("Forecast columns include the partial-year factor. Full-year columns show the unadjusted annual forecast.")
            st.dataframe(result["demand_by_year_product"], use_container_width=True, height=520)

        with tab3:
            st.subheader("Demand by Location")
            st.dataframe(result["demand_by_location"], use_container_width=True, height=420)

            if result["demand_by_location_product"] is not None:
                st.subheader("Demand by Location and Product Category")
                st.dataframe(result["demand_by_location_product"], use_container_width=True, height=520)

        with tab4:
            st.subheader("Partial-Year Demand Factors")
            st.dataframe(result["year_factors"], use_container_width=True, height=320)

        with tab5:
            st.subheader("Age/Product Weighted Rates")
            st.dataframe(result["age_product_rates"], use_container_width=True, height=520)

        with tab6:
            st.subheader("Demand by Age and Product")
            st.dataframe(result["demand_by_age_product"], use_container_width=True, height=420)

            if result["demand_by_location_age"] is not None:
                st.subheader("Demand by Location and Vehicle Age")
                st.dataframe(result["demand_by_location_age"], use_container_width=True, height=420)

            st.subheader("Age/VMT Bridge")
            st.dataframe(result["age_vmt_bridge"], use_container_width=True, height=320)

        st.subheader("Downloads")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                "Download validation log CSV",
                data=csv_bytes(result["validation_log"]),
                file_name="validation_log.csv",
                mime="text/csv",
            )

        with c2:
            st.download_button(
                "Download year/product demand CSV",
                data=csv_bytes(result["demand_by_year_product"]),
                file_name="demand_by_year_product.csv",
                mime="text/csv",
            )

        with c3:
            st.download_button(
                "Download location demand CSV",
                data=csv_bytes(result["demand_by_location"]),
                file_name="demand_by_location.csv",
                mime="text/csv",
            )

        if result["demand_by_location_product"] is not None:
            st.download_button(
                "Download location/product demand CSV",
                data=csv_bytes(result["demand_by_location_product"]),
                file_name="demand_by_location_product.csv",
                mime="text/csv",
            )

        sheets = {
            "Validation_Log": result["validation_log"],
            "Demand_Year_Product": result["demand_by_year_product"],
            "Demand_Location": result["demand_by_location"],
            "Demand_Age_Product": result["demand_by_age_product"],
            "Partial_Year_Factors": result["year_factors"],
            "Age_Product_Rates": result["age_product_rates"],
            "Age_VMT_Bridge": result["age_vmt_bridge"],
        }

        if result["demand_by_location_product"] is not None:
            sheets["Demand_Location_Product"] = result["demand_by_location_product"]

        if result["demand_by_location_age"] is not None:
            sheets["Demand_Location_Age"] = result["demand_by_location_age"]

        st.download_button(
            "Download all Excel outputs",
            data=excel_bytes(sheets),
            file_name="demand_forecast_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(str(e))
        st.caption("Tip: turn off strict missing-rate handling to let the app exclude invalid items and log them instead of stopping.")
        st.stop()
else:
    st.caption("Model will not process the data until you click the run button.")
