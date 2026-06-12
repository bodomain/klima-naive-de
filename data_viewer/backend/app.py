"""FastAPI backend for HYRAS Data Viewer."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .config import ZARR_DIR, DB_PATH, VARIABLES, STATION_VARIABLES, WORKSPACE
from .data_access import HyrasDataStore, StationDataStore

app = FastAPI(title="HYRAS Data Viewer API")

# CORS for local dev (FastAPI serves static build in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy data stores
_hyras: HyrasDataStore | None = None
_stations: StationDataStore | None = None


def _get_hyras() -> HyrasDataStore:
    global _hyras
    if _hyras is None:
        _hyras = HyrasDataStore()
    return _hyras


def _get_stations() -> StationDataStore:
    global _stations
    if _stations is None:
        _stations = StationDataStore()
    return _stations


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/variables")
def list_variables() -> dict[str, Any]:
    return {
        "hyras": {
            key: {
                "label": v["label"],
                "unit": v["unit"],
                "stats": {
                    s: {"label": meta["label"], "unit": meta["unit"]}
                    for s, meta in v["stats"].items()
                },
            }
            for key, v in VARIABLES.items()
        },
        "stations": {
            key: {"label": v["label"], "unit": v["unit"]}
            for key, v in STATION_VARIABLES.items()
        },
    }


@app.get("/api/hyras/point")
def hyras_point(
    lat: float,
    lon: float,
    variable: str,
    stat: str,
) -> dict[str, Any]:
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable: {variable}")
    if stat not in VARIABLES[variable]["stats"]:
        raise HTTPException(status_code=400, detail=f"Unknown stat: {stat}")

    try:
        series = _get_hyras().get_point_timeseries(variable, stat, lat, lon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cell = _get_hyras().get_cell_coordinates(variable, lat, lon)
    return {
        "variable": variable,
        "stat": stat,
        "requested": {"lat": lat, "lon": lon},
        "cell": cell,
        "timeseries": {
            "years": series.index.tolist(),
            "values": series.values.tolist(),
        },
    }


@app.get("/api/stations/nearest")
def nearest_station(lat: float, lon: float, variable: str | None = None) -> dict[str, Any]:
    if variable is not None and variable not in STATION_VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable: {variable}")
    try:
        info = _get_stations().get_nearest(lat, lon, variable)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return info


@app.get("/api/stations/{station_id}/timeseries")
def station_timeseries(station_id: str, variable: str) -> dict[str, Any]:
    if variable not in STATION_VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable: {variable}")
    try:
        series = _get_stations().get_timeseries(station_id, variable)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "station_id": station_id,
        "variable": variable,
        "timeseries": {
            "years": series.index.tolist(),
            "values": series.values.tolist(),
        },
    }


@app.get("/api/germany")
def germany_boundary() -> dict[str, Any]:
    """Return a rough Germany GeoJSON polygon from HYRAS grid bounds."""
    # Generate from the first available Zarr store
    try:
        for var in VARIABLES:
            zarr_path = ZARR_DIR / f"{var}_annual.zarr"
            if zarr_path.exists():
                ds = _get_hyras()._load(var)
                lat = np.asarray(ds["lat"])
                lon = np.asarray(ds["lon"])
                # Outline: top edge, right edge, bottom edge, left edge
                top = list(zip(lat[0, :], lon[0, :]))
                right = list(zip(lat[1:, -1], lon[1:, -1]))
                bottom = list(zip(lat[-1, ::-1], lon[-1, ::-1]))
                left = list(zip(lat[-2::-1, 0], lon[-2::-1, 0]))
                coords = top + right + bottom + left
                # Close the polygon
                coords.append(coords[0])
                # Convert to [lon, lat] order for GeoJSON
                geojson_coords = [[round(float(lon), 4), round(float(lat), 4)] for lat, lon in coords]
                return {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_coords],
                    },
                    "properties": {"name": "Germany", "source": "HYRAS grid boundary"},
                }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not generate boundary: {exc}")
    raise HTTPException(status_code=503, detail="No HYRAS data available")


# ---------------------------------------------------------------------------
# Static files (React build output)
# ---------------------------------------------------------------------------

static_dir = WORKSPACE / "data_viewer" / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
