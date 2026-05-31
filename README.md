# HYRAS Structural VARX Climate Forecast Germany

Python workflow for a naive structural VARX time-series analysis of German HYRAS maximum temperature, DWD sunshine and precipitation, and atmospheric CO2.

## What It Does

The command-line entry point `climate_var_vecm_forecast.py` downloads and combines the HYRAS-focused dataset:

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

## Code Layout

- `climate_var_vecm_forecast.py`: CLI and workflow orchestration
- `climate_analysis/data.py`: DWD/NOAA/HYRAS download, parsing, and annual alignment
- `climate_analysis/models.py`: ADF/Johansen tests, VAR/VECM/VARX estimation, Granger tests, and forecast logic
- `climate_analysis/plots.py`: raw trend, diagnostic, and forecast plots
- `climate_analysis/reporting.py`: CSV table writing and Markdown report generation
- `climate_analysis/config.py`: shared variables, labels, URLs, and default stations

## Main Command

Preferred structural HYRAS run with exponential CO2 continuation:

```bash
python3 climate_var_vecm_forecast.py \
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
python3 climate_var_vecm_forecast.py \
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

Symmetric benchmark model with linear CO2 continuation:

```bash
python3 climate_var_vecm_forecast.py \
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
python3 climate_var_vecm_forecast.py \
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
- Downloaded SSP paths are ratio-anchored to the final historical NOAA CO2 value before forecasting, avoiding an artificial one-year jump at the history/forecast boundary.
- The preferred structural VARX specification uses `--co2-feature dlog`, i.e. annual log differences of CO2. This is more appropriate for exponentially growing CO2 series than absolute ppm differences.
- The preferred VARX uncertainty bands use `--interval-method bootstrap`, which simulates forecast paths by resampling fitted residual vectors. The old analytic fallback is still available with `--interval-method analytic`.
- The current preferred model for the HYRAS run is `--model-mode exogenous-co2`, because it encodes the identifying assumption that atmospheric CO2 is an external driver of the German climate variables.
- In the structural model, no reverse CO2 equation is estimated. Effects from temperature, sunshine, or precipitation back onto CO2 are therefore excluded by assumption, not statistically disproven.
- The preferred output folder is `outputs_full_hyras_exogco2_exponential/`.
