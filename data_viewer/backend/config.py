"""Backend configuration."""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_CACHE = WORKSPACE / "data_cache"
ZARR_DIR = DATA_CACHE / "hyras_zarr"
DB_PATH = DATA_CACHE / "stations.db"

VARIABLES = {
    "tasmax": {
        "label": "Air Temperature Maximum",
        "unit": "°C",
        "stats": {
            "annual_max": {"label": "Annual Maximum", "unit": "°C"},
            "annual_mean": {"label": "Annual Mean", "unit": "°C"},
        },
    },
    "tas": {
        "label": "Air Temperature Mean",
        "unit": "°C",
        "stats": {
            "annual_mean": {"label": "Annual Mean", "unit": "°C"},
        },
    },
    "pr": {
        "label": "Precipitation",
        "unit": "mm",
        "stats": {
            "annual_sum": {"label": "Annual Sum", "unit": "mm"},
            "annual_mean": {"label": "Annual Mean", "unit": "mm/day"},
        },
    },
}

STATION_VARIABLES = {
    "temp": {"label": "Temperature", "unit": "°C"},
    "sunshine": {"label": "Sunshine Duration", "unit": "h"},
    "precip": {"label": "Precipitation", "unit": "mm"},
}
