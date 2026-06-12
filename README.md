# HYRAS Climate Tools

This repository contains two related but separate pieces:

- `data_viewer/`: a FastAPI + React app for exploring HYRAS grid data and nearby DWD station observations.
- `modeling/`: statistical climate analysis and forecasting scripts for HYRAS, DWD station data, and atmospheric CO2.

## What It Does

The modelling entry point `modeling/climate_var_vecm_forecast.py` downloads and combines the HYRAS-focused dataset:

- HYRAS-DE gridded daily maximum air temperature (`tasmax`) from DWD
- DWD annual station data for sunshine duration (`JA_SD_S`)
- DWD annual station data for precipitation (`JA_RR`)
- NOAA Mauna Loa annual CO2 concentration
- CMIP6/ScenarioMIP SSP CO2 concentration pathways from Meinshausen et al. (2020), downloaded from the GMD supplement when selected

It then:

- aligns annual time series
- runs ADF stationarity tests
- supports a preferred structural VARX model with CO2 as exogenous driver, plus symmetric VAR/VECM benchmark options
- runs Granger causality tests, IRF, and FEVD where they are meaningful for the selected model
- creates a naive forecast to 2100 using an external CO2 path
- writes plots, CSV tables, and a Markdown report

## Repository Layout

- `data_viewer/backend/`: FastAPI API for HYRAS point queries, station lookup, and preprocessing.
- `data_viewer/frontend/`: Vite/React data viewer.
- `data_viewer/start_app.sh`: starts the backend and frontend together.
- `modeling/climate_var_vecm_forecast.py`: modelling CLI and workflow orchestration.
- `modeling/structural_time_series_forecast.py`: separate structural time-series scenario runner.
- `modeling/climate_analysis/`: reusable modelling package.
- `docs/`: report source and generated report files.
- `data_cache/`: downloaded HYRAS/DWD/CO2 data cache, ignored by Git.
- `outputs*/`: generated modelling outputs, ignored by Git.

## Data Viewer App

Start the FastAPI backend and Vite frontend together:

```bash
./start_app.sh
```

Then open `http://127.0.0.1:5173`. Press `Ctrl-C` in the terminal to stop both servers.

To refresh station coordinates in an existing local station database:

```bash
python3 data_viewer/backend/preprocessing.py --station-coordinates-only
```

## Main Command

Preferred structural HYRAS run with exponential CO2 continuation:

```bash
python3 modeling/climate_var_vecm_forecast.py \
  --temperature-source hyras \
  --hyras-aggregation annual_spatial_max \
  --all-stations \
  --temp-aggregation max \
  --model-mode exogenous-co2 \
  --co2-feature dlog \
  --co2-scenario exponential \
  --confidence-level 0.90 \
  --interval-method bootstrap \
  --bootstrap-sims 2000 \
  --outdir outputs_full_hyras_exogco2_exponential \
  --cache-dir data_cache
```

Preferred structural HYRAS run with an IPCC/CMIP6 high-CO2 worst-case pathway:

```bash
python3 modeling/climate_var_vecm_forecast.py \
  --temperature-source hyras \
  --hyras-aggregation annual_spatial_max \
  --all-stations \
  --temp-aggregation max \
  --model-mode exogenous-co2 \
  --co2-feature dlog \
  --co2-scenario ssp585 \
  --confidence-level 0.90 \
  --interval-method bootstrap \
  --bootstrap-sims 2000 \
  --outdir outputs_full_hyras_exogco2_ssp585 \
  --cache-dir data_cache
```

Available SSP CO2 pathways: `ssp119`, `ssp126`, `ssp245`, `ssp370`, `ssp370-lowntcf`, `ssp434`, `ssp460`, `ssp534-over`, `ssp585`.

The documented `ssp585` run is intentionally a high-CO2 worst-case stress test. It should not be read as the central or most likely emissions pathway.

The scenario comparison output adds the five main CMIP6/ScenarioMIP CO2 pathways:

- `ssp119`: SSP1-1.9
- `ssp126`: SSP1-2.6
- `ssp245`: SSP2-4.5
- `ssp370`: SSP3-7.0
- `ssp585`: SSP5-8.5 high-CO2 worst-case

It writes `outputs_full_hyras_exogco2_ssp585/05_scenario_comparison.png` and `scenario_comparison_summary.csv`.

## Structural Time-Series Scenario Model

The structural time-series runner is the more interpretable scenario model. It fits HYRAS annual maximum temperature with a Harvey-style state-space specification:

- local temperature level
- AR(1) annual weather noise
- logarithmic CO2 forcing, `log(CO2 / CO2_1951)`

It uses CMIP6/ScenarioMIP CO2 concentration pathways as external scenario inputs and does not estimate a reverse climate-to-CO2 equation.

```bash
python3 modeling/structural_time_series_forecast.py \
  --input outputs_full_hyras_exogco2_ssp585/climate_co2_aligned.csv \
  --cache-dir data_cache \
  --outdir outputs_structural_time_series \
  --confidence-level 0.90
```

This writes `outputs_structural_time_series/01_sts_scenario_forecasts.png`, per-scenario forecast CSVs, `sts_forecast_summary.csv`, and `report.md`.

Symmetric benchmark model with linear CO2 continuation:

```bash
python3 modeling/climate_var_vecm_forecast.py \
  --temperature-source hyras \
  --hyras-aggregation annual_spatial_max \
  --all-stations \
  --temp-aggregation max \
  --co2-scenario linear \
  --outdir outputs_full_hyras_precip_best \
  --cache-dir data_cache
```

Symmetric benchmark model with exponential CO2 continuation:

```bash
python3 modeling/climate_var_vecm_forecast.py \
  --temperature-source hyras \
  --hyras-aggregation annual_spatial_max \
  --all-stations \
  --temp-aggregation max \
  --co2-scenario exponential \
  --outdir outputs_full_hyras_precip_exponential_co2 \
  --cache-dir data_cache
```

## Notes

- HYRAS downloads are large: about 5.8 GB for 1951-2024.
- Downloaded data and generated outputs are excluded via `.gitignore`.
- The forecast is illustrative and statistical, not a physical climate projection.
- Forecast interval width can be changed with `--confidence-level`, for example `--confidence-level 0.90` for broad uncertainty bands.
- CO2 paths can be simple extrapolations (`linear`, `exponential`) or downloaded SSP pathways such as `ssp245`, `ssp370`, and `ssp585`.
- `ssp585` is treated and labeled as the high-CO2 worst-case stress scenario.
- In the SSP5-8.5 output chart, the CO2 panel also shows a dotted documentary exponential CO2 trend for comparison. That reference line is not used as model input.
- `05_scenario_comparison.png` shows the main SSP CO2 pathways and their corresponding structural VARX temperature forecasts in one figure.
- `outputs_structural_time_series/01_sts_scenario_forecasts.png` shows the same main SSP CO2 pathways in the structural time-series model.
- Downloaded SSP paths are ratio-anchored to the final historical NOAA CO2 value before forecasting, avoiding an artificial one-year jump at the history/forecast boundary.
- The preferred structural VARX specification uses `--co2-feature dlog`, i.e. annual log differences of CO2. This is more appropriate for exponentially growing CO2 series than absolute ppm differences.
- The preferred VARX uncertainty bands use `--interval-method bootstrap`, which simulates forecast paths by resampling fitted residual vectors. The old analytic fallback is still available with `--interval-method analytic`.
- The current preferred model for the HYRAS run is `--model-mode exogenous-co2`, because it encodes the identifying assumption that atmospheric CO2 is an external driver of the German climate variables.
- In the structural model, no reverse CO2 equation is estimated. Effects from temperature, sunshine, or precipitation back onto CO2 are therefore excluded by assumption, not statistically disproven.
- The structural time-series model uses CO2 levels through logarithmic forcing. It is useful as a scenario model, but it is still statistical and should not be confused with a physical climate model.
- The preferred output folder is `outputs_full_hyras_exogco2_exponential/`.
