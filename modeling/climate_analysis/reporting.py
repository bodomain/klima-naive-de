"""Output table and Markdown report writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import ModelBundle

def write_report(
    output: Path,
    data: pd.DataFrame,
    station_data: pd.DataFrame,
    adf: pd.DataFrame,
    johansen: pd.DataFrame,
    granger: pd.DataFrame,
    bundle: ModelBundle,
    forecast: pd.DataFrame,
    external_co2: pd.Series,
    use_daily_max: bool,
    temp_aggregation: str,
    all_stations: bool,
    temperature_source: str,
    hyras_aggregation: str,
    co2_scenario: str,
    co2_lookback: int,
    confidence_level: float,
    co2_feature: str = "dlog",
    interval_method: str = "bootstrap",
    bootstrap_sims: int = 2000,
    forecast_start: str = "trend",
    trend_window: int = 11,
) -> None:
    first_year, last_year = int(data.index.min()), int(data.index.max())
    forecast_end = int(forecast.index.max())
    temp_change = forecast.loc[forecast_end, "temp"] - data.loc[last_year, "temp"]
    sun_change = forecast.loc[forecast_end, "sunshine"] - data.loc[last_year, "sunshine"]
    precip_change = forecast.loc[forecast_end, "precip"] - data.loc[last_year, "precip"]
    co2_change = forecast.loc[forecast_end, "co2"] - data.loc[last_year, "co2"]
    station_records = (
        station_data[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_id")
        .to_dict("records")
    )
    stations = station_records if len(station_records) <= 12 else f"{len(station_records)} stations; see dwd_station_data_long.csv"
    source_note = (
        "annual maximum of daily maximum temperature TXK and annual sunshine sum from daily KL"
        if use_daily_max
        else "annual KL station values; temperature uses JA_MX_TX, sunshine JA_SD_S, precipitation JA_RR"
    )
    station_scope = "all available DWD annual KL stations" if all_stations else "selected representative stations"
    temp_source_note = (
        f"HYRAS-DE gridded daily tasmax ({hyras_aggregation})"
        if temperature_source == "hyras"
        else f"DWD station annual {temp_aggregation} across loaded station-level yearly maxima"
    )

    co2_driver_name = "d_co2" if "d_co2" in granger["causing"].astype(str).values else "co2"
    if bundle.model_type == "EXOGENOUS_CO2_VARX":
        feature_labels = {
            "dlog": "annual log CO2 differences, log(CO2_t) - log(CO2_{t-1})",
            "dppm": "annual absolute CO2 ppm differences, CO2_t - CO2_{t-1}",
            "forcing": "logarithmic CO2 level, log(CO2_t / initial CO2)",
        }
        co2_evidence = (
            "CO2 is imposed as an external driver in this structural model. "
            f"The climate equations are estimated conditional on {feature_labels.get(co2_feature, co2_feature)} "
            "and lagged climate variables; "
            "no reverse CO2 equation is estimated."
        )
    else:
        significant = granger[
            (granger["causing"].eq(co2_driver_name)) & (granger["reject_5pct"].eq(True))
        ]
        co2_evidence = (
            f"The stationary VAR Granger tests reject at 5% for at least one {co2_driver_name} -> climate-variable direction."
            if not significant.empty
            else f"The stationary VAR Granger tests do not reject at 5% for {co2_driver_name} -> the selected German climate variables."
        )
    johansen_note = (
        f"- Johansen cointegration rank at 5% trace-test rule: {bundle.coint_rank}"
        if bundle.model_type == "VECM"
        else "- Johansen/VECM skipped because the model is structural/exogenous or the estimated integration orders are mixed."
    )
    if co2_scenario in {"linear", "exponential"}:
        co2_path_note = (
            f"a {co2_scenario} continuation of the last {co2_lookback} historical CO2 years"
        )
        scenario_note = ""
    else:
        scenario_prefix = (
            "high-CO2 worst-case stress scenario "
            if co2_scenario == "ssp585"
            else ""
        )
        co2_path_note = (
            f"the downloaded CMIP6/ScenarioMIP {scenario_prefix}{co2_scenario.upper()} CO2 concentration pathway "
            "from Meinshausen et al. (2020), ratio-anchored to the final historical NOAA CO2 value"
        )
        scenario_note = (
            "\n- Scenario framing: SSP5-8.5 is used here as a high-CO2 worst-case stress test, "
            "not as a central or most likely forecast."
            if co2_scenario == "ssp585"
            else ""
        )

    report_title = (
        "Structural VARX Climate Forecast Report"
        if bundle.model_type == "EXOGENOUS_CO2_VARX"
        else "Climate VAR/VECM Forecast Report"
    )
    sensitivity_label = "VARX" if bundle.model_type == "EXOGENOUS_CO2_VARX" else "VAR/VECM"

    text = f"""# {report_title}

## Data

- Period used: {first_year}-{last_year}
- DWD stations: {stations}
- DWD climate definition: {source_note}
- DWD station scope: {station_scope}
- Temperature source: {temp_source_note}
- CO2 source: NOAA GML Mauna Loa annual mean. Years before the Mauna Loa start are linearly backcast only for alignment if needed.

## Model

- Selected framework: {bundle.model_type}
- Lag setting: {bundle.lag_order}
- Estimated integration orders from ADF rule: {bundle.integration_order}
{johansen_note}

## Statistical summary

{co2_evidence}

ADF and Johansen/model-selection details are written to CSV files in the output directory. Level correlations should not be interpreted causally. In `EXOGENOUS_CO2_VARX`, the direction CO2 -> climate is an explicit identifying assumption, not a result of symmetric VAR causality testing.

## Naive forecast to {forecast_end}

The model baseline forecast is adjusted with an external CO2 path: {co2_path_note}. Under that assumption:

- CO2 changes by about {co2_change:.1f} ppm from {last_year} to {forecast_end}.
- Annual maximum temperature changes by about {temp_change:.2f} deg C from {last_year} to {forecast_end}.
- Sunshine duration changes by about {sun_change:.1f} h/year from {last_year} to {forecast_end}.
- Annual precipitation changes by about {precip_change:.1f} mm/year from {last_year} to {forecast_end}.
- Forecast uncertainty band: {confidence_level:.0%} confidence interval.
- Forecast interval method: {interval_method}{f" with {bootstrap_sims} residual bootstrap paths" if interval_method == "bootstrap" else ""}.
- Forecast start: {forecast_start}{f" ({trend_window}-year trailing smoothed climate level)" if forecast_start == "trend" else " (final observed year)"}.
{scenario_note}
- CO2 chart note: the dotted exponential CO2 trend, when shown, is documentary only and is not used as model input.
- Scenario comparison: `05_scenario_comparison.png` compares SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5 with the same structural VARX specification.

## Limitations

- This is a statistical extrapolation, not a process-based climate model.
- Mauna Loa CO2 is a global proxy, not a Germany-specific emissions or concentration series.
- DWD station averages are not an official Germany-area mean.
- {sensitivity_label} long-run forecasts are sensitive to lag length, trend specification, structural breaks, station choice, and non-stationarity.
- Policy scenarios, aerosols, land-use change, circulation shifts, volcanic forcing, solar variability, and internal climate variability are not explicitly modeled.
- The CO2 scenario is naive. For decision support, replace it with SSP/RCP concentration pathways and use physical climate-model ensembles.
"""
    output.write_text(text, encoding="utf-8")

def save_tables(outdir: Path, **tables: pd.DataFrame) -> None:
    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv", index=True)
