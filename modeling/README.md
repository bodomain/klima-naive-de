# Statistical Modeling

This folder contains the HYRAS/DWD/CO2 statistical modelling workflow.

Primary scripts:

- `climate_var_vecm_forecast.py`: downloads/aligns HYRAS, DWD station data, CO2 paths, fits VAR/VECM/VARX-style models, and writes outputs.
- `structural_time_series_forecast.py`: runs the structural time-series and energy-balance style scenario models.
- `climate_analysis/`: shared package used by both scripts.

Run commands from the repository root so default paths such as `data_cache/` and `outputs*/` stay in the expected place:

```bash
python3 modeling/climate_var_vecm_forecast.py --help
python3 modeling/structural_time_series_forecast.py --help
```
