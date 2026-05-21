# UIO Demand Forecast Streamlit App

This version includes:

- Demand by year/product
- Demand by location
- Demand by location/product
- Optional partial-year demand factors by forecast year
- High-mileage capping to the highest available mileage band

## Partial-year demand factors

Use `Demand Year Factor` to scale demand for forecast years that are not full years.

Examples:

- `1.00` = full year
- `0.50` = half year
- `0.25` = quarter year

The factor scales final demand. It does not change cumulative mileage.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
