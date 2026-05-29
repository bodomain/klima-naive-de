# Climate VAR/VECM Forecast Germany

Python workflow for a naive multivariate time-series analysis of German climate indicators and atmospheric CO2.

## What It Does

The script `climate_var_vecm_forecast.py` downloads and combines:

- HYRAS-DE gridded daily maximum air temperature (`tasmax`) from DWD
- DWD annual station data for sunshine duration (`JA_SD_S`)
- DWD annual station data for precipitation (`JA_RR`)
- NOAA Mauna Loa annual CO2 concentration

It then:

- aligns annual time series
- runs ADF stationarity tests
- chooses between VECM, differenced VAR, or mixed stationary VAR
- runs Granger causality tests, IRF, and FEVD
- creates a naive forecast to 2100 using a linear external CO2 path
- writes plots, CSV tables, and a Markdown report

## Main Command

```bash
python3 climate_var_vecm_forecast.py \
  --temperature-source hyras \
  --hyras-aggregation annual_spatial_max \
  --all-stations \
  --temp-aggregation max \
  --outdir outputs_full_hyras_precip_best \
  --cache-dir data_cache
```

## Notes

- HYRAS downloads are large: about 5.8 GB for 1951-2024.
- Downloaded data and generated outputs are excluded via `.gitignore`.
- The forecast is illustrative and statistical, not a physical climate projection.
- The current preferred model for the HYRAS run is a mixed VAR using climate levels and `d_CO2`, because the integration orders are mixed.
