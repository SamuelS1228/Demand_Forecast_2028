import io
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UIO Demand Forecast Model", layout="wide")

st.title("UIO Demand Forecast Model")
st.caption(
    "Calculates compact age/product weighted rates first, then applies them to UIO. "
    "Includes demand by year, product, location, and optional partial-year factors."
)

REQUIRED_UIO_COLS = ["CBSA", "Forecast Year", "Vehicle Age", "Calibrated Retained UIO Rounded"]
REQUIRED_VMT_COLS = ["Annual VMT", "VMT Probability"]
PRODUCT_COL = "Product Category"


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
    if s.dropna().max() > 1:
        s = s / 100
    return s


def parse_mileage_band(label: str) -> Optional[Tuple[int, int]]:
    nums = re.findall(r"\d[\d,]*", str(label))
    if len(nums) < 2:
        return None
    return int(nums[0].replace(",", "")), int(nums[1].replace(",", ""))


def build_mileage_bands_from_headers(headers) -> pd.DataFrame:
    rows = []
    for h in headers:
        parsed = parse_mileage_band(h)
        if parsed:
            rows.append({"Min Miles": parsed[0], "Max Miles": parsed[1], "Mileage Band": str(h).strip()})
    bands = pd.DataFrame(rows).drop_duplicates().sort_values("Min Miles").reset_index(drop=True)
    if bands.empty:
        raise ValueError("No mileage band headers could be parsed. Expected headers like '0 To 9,999'.")
    return bands


def assign_mileage_band(miles: pd.Series, bands: pd.DataFrame, cap_to_max_band: bool = True) -> pd.Series:
    bands = bands.sort_values("Min Miles").reset_index(drop=True)
    min_miles = bands["Min Miles"].iloc[0]
    max_miles = bands["Max Miles"].iloc[-1]
    max_band = bands["Mileage Band"].iloc[-1]
    min_band = bands["Mileage Band"].iloc[0]
    bins = list(bands["Min Miles"]) + [max_miles + 1]
    labels = list(bands["Mileage Band"])
    assigned = pd.cut(miles, bins=bins, labels=labels, right=False, include_lowest=True).astype("object")
    if cap_to_max_band:
        assigned = assigned.where(miles <= max_miles, max_band)
        assigned = assigned.where(miles >= min_miles, min_band)
        assigned = assigned.fillna(max_band)
    else:
        assigned = assigned.fillna("Above Max Mileage")
    return assigned


def normalize_rate_table(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if PRODUCT_COL not in df.columns:
        raise ValueError(f"Missing required column: {PRODUCT_COL}")
    mileage_cols = [c for c in df.columns if c != PRODUCT_COL]
    long_df = df.melt(id_vars=[PRODUCT_COL], value_vars=mileage_cols, var_name="Mileage Band", value_name=value_name)
    long_df[PRODUCT_COL] = long_df[PRODUCT_COL].astype(str).str.strip()
    long_df["Mileage Band"] = long_df["Mileage Band"].astype(str).str.strip()
    long_df[value_name] = clean_numeric(long_df[value_name])
    if value_name == "Incidence Rate" and long_df[value_name].dropna().max() > 1:
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
                ws.set_column(idx, idx, min(max(len(str(col)) + 2, 12), 42))
    return output.getvalue()


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def make_default_partial_years(uio_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if uio_df is None or uio_df.empty or "Forecast Year" not in uio_df.columns:
        return pd.DataFrame({"Forecast Year": [], "Demand Year Factor": []})
    years = pd.to_numeric(uio_df["Forecast Year"], errors="coerce").dropna().astype(int).drop_duplicates().sort_values()
    return pd.DataFrame({"Forecast Year": years, "Demand Year Factor": 1.0})


def clean_year_factors(year_factor_df: pd.DataFrame, available_years: pd.Series) -> pd.DataFrame:
    yf = year_factor_df.copy()
    yf.columns = [str(c).strip() for c in yf.columns]
    if "Forecast Year" not in yf.columns:
        raise ValueError("Partial-year table must include 'Forecast Year'.")
    if "Demand Year Factor" not in yf.columns:
        raise ValueError("Partial-year table must include 'Demand Year Factor'.")
    yf["Forecast Year"] = clean_numeric(yf["Forecast Year"]).astype("Int64")
    yf["Demand Year Factor"] = clean_numeric(yf["Demand Year Factor"])
    if yf["Demand Year Factor"].dropna().max() > 1:
        yf["Demand Year Factor"] = yf["Demand Year Factor"] / 100
    yf = yf.dropna(subset=["Forecast Year"])
    yf["Forecast Year"] = yf["Forecast Year"].astype(int)
    yf["Demand Year Factor"] = yf["Demand Year Factor"].fillna(1.0).clip(lower=0.0, upper=1.0)
    base = pd.DataFrame({"Forecast Year": sorted(pd.Series(available_years).dropna().astype(int).unique())})
    out = base.merge(yf[["Forecast Year", "Demand Year Factor"]], on="Forecast Year", how="left")
    out["Demand Year Factor"] = out["Demand Year Factor"].fillna(1.0)
    return out


@st.cache_data(show_spinner=False)
def run_forecast(uio_df, incidence_wide, units_wide, vmt_df, year_factor_df, age0_factor,
                 make_location_product_output, make_location_age_output, cap_to_max_band):
    uio = uio_df.copy()
    uio.columns = [str(c).strip() for c in uio.columns]
    missing = [c for c in REQUIRED_UIO_COLS if c not in uio.columns]
    if missing:
        raise ValueError(f"UIO file is missing required columns: {missing}")
    uio["CBSA"] = uio["CBSA"].astype(str).str.strip()
    uio["Forecast Year"] = clean_numeric(uio["Forecast Year"]).astype("Int64")
    uio["Vehicle Age"] = clean_numeric(uio["Vehicle Age"]).astype("Int64")
    uio["UIO"] = clean_numeric(uio["Calibrated Retained UIO Rounded"]).fillna(0)
    uio = uio.dropna(subset=["Forecast Year", "Vehicle Age"])
    uio["Forecast Year"] = uio["Forecast Year"].astype(int)
    uio["Vehicle Age"] = uio["Vehicle Age"].astype(int)

    year_factors = clean_year_factors(year_factor_df, uio["Forecast Year"])
    uio = uio.merge(year_factors, on="Forecast Year", how="left")
    uio["Demand Year Factor"] = uio["Demand Year Factor"].fillna(1.0)

    vmt = vmt_df.copy()
    vmt.columns = [str(c).strip() for c in vmt.columns]
    missing = [c for c in REQUIRED_VMT_COLS if c not in vmt.columns]
    if missing:
        raise ValueError(f"VMT file is missing required columns: {missing}")
    vmt["Annual VMT"] = clean_numeric(vmt["Annual VMT"])
    vmt["VMT Probability"] = normalize_probability(vmt["VMT Probability"])
    vmt = vmt.dropna(subset=["Annual VMT", "VMT Probability"])
    if not np.isclose(vmt["VMT Probability"].sum(), 1.0, atol=0.0001):
        raise ValueError(f"VMT probabilities must sum to 100%. Current sum: {vmt['VMT Probability'].sum():.2%}")

    incidence_wide = incidence_wide.copy()
    incidence_wide.columns = [str(c).strip() for c in incidence_wide.columns]
    units_wide = units_wide.copy()
    units_wide.columns = [str(c).strip() for c in units_wide.columns]
    if PRODUCT_COL not in incidence_wide.columns:
        raise ValueError(f"Incidence table missing column: {PRODUCT_COL}")
    if PRODUCT_COL not in units_wide.columns:
        raise ValueError(f"Units table missing column: {PRODUCT_COL}")

    bands = build_mileage_bands_from_headers([c for c in incidence_wide.columns if c != PRODUCT_COL])
    inc_long = normalize_rate_table(incidence_wide, "Incidence Rate")
    units_long = normalize_rate_table(units_wide, "Units Per Repair")

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
    age_vmt["Mileage Band"] = assign_mileage_band(age_vmt["Cumulative Miles"], bands, cap_to_max_band)

    products = pd.DataFrame({PRODUCT_COL: sorted(incidence_wide[PRODUCT_COL].astype(str).str.strip().unique())})
    age_vmt["_key"] = 1
    products["_key"] = 1
    bridge = age_vmt.merge(products, on="_key").drop(columns="_key")
    bridge = bridge.merge(inc_long, on=[PRODUCT_COL, "Mileage Band"], how="left")
    bridge = bridge.merge(units_long, on=[PRODUCT_COL, "Mileage Band"], how="left")
    missing_inc = int(bridge["Incidence Rate"].isna().sum())
    missing_units = int(bridge["Units Per Repair"].isna().sum())
    if missing_inc or missing_units:
        sample = bridge.loc[bridge["Incidence Rate"].isna() | bridge["Units Per Repair"].isna(),
                            ["Vehicle Age", "Annual VMT", "Cumulative Miles", "Mileage Band", PRODUCT_COL]].head(10).to_string(index=False)
        raise ValueError(f"Missing rate data. Missing incidence rows: {missing_inc:,}. Missing units rows: {missing_units:,}. Sample missing rows:\n{sample}")

    bridge["Weighted Incident Rate Component"] = bridge["VMT Probability"] * bridge["Incidence Rate"]
    bridge["Weighted Piece Rate Component"] = bridge["VMT Probability"] * bridge["Incidence Rate"] * bridge["Units Per Repair"]
    age_product_rates = bridge.groupby(["Vehicle Age", PRODUCT_COL], as_index=False).agg(
        Weighted_Incident_Rate=("Weighted Incident Rate Component", "sum"),
        Weighted_Piece_Rate=("Weighted Piece Rate Component", "sum"),
    )

    def apply_rates(group_cols):
        base = uio.groupby(group_cols + ["Vehicle Age", "Demand Year Factor"], as_index=False).agg(UIO=("UIO", "sum"))
        out = base.merge(age_product_rates, on="Vehicle Age", how="left")
        out["Full-Year Forecast Incidents"] = out["UIO"] * out["Weighted_Incident_Rate"]
        out["Full-Year Forecast Piece Demand"] = out["UIO"] * out["Weighted_Piece_Rate"]
        out["Forecast Incidents"] = out["Full-Year Forecast Incidents"] * out["Demand Year Factor"]
        out["Forecast Piece Demand"] = out["Full-Year Forecast Piece Demand"] * out["Demand Year Factor"]
        return out

    year_product_detail = apply_rates(["Forecast Year"])
    demand_by_year_product = year_product_detail.groupby(["Forecast Year", PRODUCT_COL], as_index=False).agg(
        Demand_Year_Factor=("Demand Year Factor", "max"),
        Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
        Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
        Forecast_Incidents=("Forecast Incidents", "sum"),
        Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
    )

    loc_detail = apply_rates(["CBSA", "Forecast Year"])
    demand_by_location = loc_detail.groupby(["CBSA", "Forecast Year"], as_index=False).agg(
        Demand_Year_Factor=("Demand Year Factor", "max"),
        Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
        Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
        Forecast_Incidents=("Forecast Incidents", "sum"),
        Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
    )

    demand_by_location_product = None
    if make_location_product_output:
        demand_by_location_product = loc_detail.groupby(["CBSA", "Forecast Year", PRODUCT_COL], as_index=False).agg(
            Demand_Year_Factor=("Demand Year Factor", "max"),
            Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
            Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )

    demand_by_location_age = None
    if make_location_age_output:
        demand_by_location_age = loc_detail.groupby(["CBSA", "Forecast Year", "Vehicle Age"], as_index=False).agg(
            Demand_Year_Factor=("Demand Year Factor", "max"),
            Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
            Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
            Forecast_Incidents=("Forecast Incidents", "sum"),
            Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
        )

    age_detail = apply_rates([])
    demand_by_age_product = age_detail.groupby(["Vehicle Age", PRODUCT_COL], as_index=False).agg(
        Full_Year_Forecast_Incidents=("Full-Year Forecast Incidents", "sum"),
        Full_Year_Forecast_Piece_Demand=("Full-Year Forecast Piece Demand", "sum"),
        Forecast_Incidents=("Forecast Incidents", "sum"),
        Forecast_Piece_Demand=("Forecast Piece Demand", "sum"),
    )

    diagnostics = {
        "uio_rows": len(uio),
        "location_count": uio["CBSA"].nunique(),
        "product_count": products[PRODUCT_COL].nunique(),
        "total_piece_demand": demand_by_year_product["Forecast_Piece_Demand"].sum(),
        "full_year_piece_demand": demand_by_year_product["Full_Year_Forecast_Piece_Demand"].sum(),
        "above_max_age_vmt_rows": int(age_vmt["Above Max Band Flag"].sum()),
        "max_mileage_band": max_band_label,
        "max_mileage_band_miles": int(max_band_miles),
        "partial_years_count": int((year_factors["Demand Year Factor"] < 1).sum()),
    }
    return {
        "diagnostics": diagnostics,
        "year_factors": year_factors,
        "age_vmt_bridge": age_vmt.drop(columns=[c for c in ["_key"] if c in age_vmt.columns]),
        "age_product_rates": age_product_rates,
        "demand_by_year_product": demand_by_year_product,
        "demand_by_location": demand_by_location,
        "demand_by_location_product": demand_by_location_product,
        "demand_by_location_age": demand_by_location_age,
        "demand_by_age_product": demand_by_age_product,
    }


st.subheader("1. Upload files")
c1, c2 = st.columns(2)
with c1:
    uio_upload = st.file_uploader("UIO_Age_Summary.csv", type=["csv"])
    incidence_upload = st.file_uploader("Incidence_By_Mileage.csv", type=["csv"])
with c2:
    units_upload = st.file_uploader("Units_Per_Repair.csv", type=["csv"])
    vmt_upload = st.file_uploader("VMT_Distribution.csv optional", type=["csv"])

default_vmt = pd.DataFrame({"Annual VMT": [8000, 10000, 12000, 15000], "VMT Probability": [0.15, 0.25, 0.40, 0.20]})
vmt_df = read_csv_cached(vmt_upload.getvalue()) if vmt_upload else default_vmt
uio_preview = read_csv_cached(uio_upload.getvalue()) if uio_upload else None

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
st.write("Set a factor below 1.0 for any forecast year where you only want to calculate part of the year. Example: 0.50 = half year. This scales demand only, not cumulative mileage.")
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
            "Demand Year Factor": st.column_config.NumberColumn("Demand Year Factor", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
        },
        key="partial_year_editor",
    )

st.sidebar.header("Settings")
age0_factor = st.sidebar.slider("Age 0 mileage factor", 0.0, 1.0, 0.5, 0.05)
cap_to_max_band = st.sidebar.checkbox("Cap mileage above max band to highest band", value=True)
make_location_product_output = st.sidebar.checkbox("Create location/product output", value=True)
make_location_age_output = st.sidebar.checkbox("Create location/age output", value=False)

if not np.isclose(prob_sum, 1.0, atol=0.0001):
    st.warning("Fix the VMT probabilities before running.")
    st.stop()

st.subheader("4. Run model")
if not all([uio_upload, incidence_upload, units_upload]):
    st.info("Upload the UIO, incidence, and units-per-repair files to run.")
    st.stop()

if st.button("Run demand forecast", type="primary"):
    try:
        with st.spinner("Running forecast..."):
            result = run_forecast(
                uio_df=read_csv_cached(uio_upload.getvalue()),
                incidence_wide=read_csv_cached(incidence_upload.getvalue()),
                units_wide=read_csv_cached(units_upload.getvalue()),
                vmt_df=edited_vmt,
                year_factor_df=edited_year_factors,
                age0_factor=age0_factor,
                make_location_product_output=make_location_product_output,
                make_location_age_output=make_location_age_output,
                cap_to_max_band=cap_to_max_band,
            )
        st.success("Forecast complete.")
        d = result["diagnostics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("UIO Rows", f"{d['uio_rows']:,}")
        m2.metric("Locations", f"{d['location_count']:,}")
        m3.metric("Product Categories", f"{d['product_count']:,}")
        m4.metric("Partial Years", f"{d['partial_years_count']:,}")
        m5, m6 = st.columns(2)
        m5.metric("Full-Year Piece Demand", f"{d['full_year_piece_demand']:,.0f}")
        m6.metric("Adjusted Piece Demand", f"{d['total_piece_demand']:,.0f}")
        if d["above_max_age_vmt_rows"] > 0 and cap_to_max_band:
            st.warning(f"{d['above_max_age_vmt_rows']:,} age/VMT outcomes exceeded the highest mileage band ({d['max_mileage_band']}, max {d['max_mileage_band_miles']:,} miles). They were capped to the highest available mileage band.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Year/Product Demand", "Location Demand", "Partial-Year Factors", "Age/Product Rates", "Other Outputs"])
        with tab1:
            st.subheader("Demand by Forecast Year and Product Category")
            st.caption("Forecast columns include the partial-year factor. Full-year columns show the unadjusted annual forecast.")
            st.dataframe(result["demand_by_year_product"], use_container_width=True, height=520)
        with tab2:
            st.subheader("Demand by Location")
            st.dataframe(result["demand_by_location"], use_container_width=True, height=420)
            if result["demand_by_location_product"] is not None:
                st.subheader("Demand by Location and Product Category")
                st.dataframe(result["demand_by_location_product"], use_container_width=True, height=520)
        with tab3:
            st.subheader("Partial-Year Demand Factors")
            st.dataframe(result["year_factors"], use_container_width=True, height=320)
        with tab4:
            st.subheader("Age/Product Weighted Rates")
            st.dataframe(result["age_product_rates"], use_container_width=True, height=520)
        with tab5:
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
            st.download_button("Download year/product demand CSV", data=csv_bytes(result["demand_by_year_product"]), file_name="demand_by_year_product.csv", mime="text/csv")
        with c2:
            st.download_button("Download location demand CSV", data=csv_bytes(result["demand_by_location"]), file_name="demand_by_location.csv", mime="text/csv")
        with c3:
            if result["demand_by_location_product"] is not None:
                st.download_button("Download location/product demand CSV", data=csv_bytes(result["demand_by_location_product"]), file_name="demand_by_location_product.csv", mime="text/csv")
        sheets = {
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
        st.download_button("Download all Excel outputs", data=excel_bytes(sheets), file_name="demand_forecast_outputs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    st.caption("Model will not process the data until you click the run button.")
