"""Shared constants for the HYRAS climate analysis workflow."""

RANDOM_SEED = 42

CLIMATE_VARIABLES = ["temp", "sunshine", "precip"]
MODEL_VARIABLES = CLIMATE_VARIABLES + ["co2"]
VARIABLE_LABELS = {
    "temp": "Annual maximum temperature (deg C)",
    "sunshine": "Sunshine duration (h/year)",
    "precip": "Annual precipitation (mm)",
    "co2": "CO2 Mauna Loa (ppm)",
    "d_co2": "Annual CO2 change (ppm/year)",
}
VARIABLE_COLORS = {
    "temp": "#9b2c2c",
    "sunshine": "#c47f00",
    "precip": "#0b4fbd",
    "co2": "#245c8a",
    "d_co2": "#245c8a",
}
SOURCE_NOTE = "Data sources: DWD HYRAS-DE tasmax; DWD annual station sunshine/precipitation; NOAA GML Mauna Loa CO2"

DWD_ANNUAL_KL_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/annual/kl/historical/"
)
DWD_DAILY_KL_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/daily/kl/historical/"
)
HYRAS_TASMAX_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "grids_germany/daily/hyras_de/air_temperature_max/"
)
NOAA_CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"
GMD_SSP_SUPPLEMENT_URL = "https://gmd.copernicus.org/articles/13/3571/2020/gmd-13-3571-2020-supplement.zip"

SSP_SCENARIOS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp370-lowntcf": "SSP3-7.0-lowNTCF",
    "ssp434": "SSP4-3.4",
    "ssp460": "SSP4-6.0",
    "ssp534-over": "SSP5-3.4-over",
    "ssp585": "SSP5-8.5",
}

DEFAULT_STATIONS = {
    "03987": "Potsdam",
    "02290": "Hohenpeissenberg",
    "00433": "Berlin-Dahlem",
    "01048": "Dresden-Klotzsche",
}
