"""Data access layer for HYRAS Zarr and DWD stations SQLite."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlite3
import xarray as xr

from .config import ZARR_DIR, DB_PATH

# ---------------------------------------------------------------------------
# HYRAS Zarr
# ---------------------------------------------------------------------------

class HyrasDataStore:
    """Lazy-loaded HYRAS annual Zarr stores with point queries."""

    def __init__(self, zarr_dir: Path = ZARR_DIR):
        self.zarr_dir = zarr_dir
        self._cache: dict[str, xr.Dataset] = {}
        self._latlon: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load(self, variable: str) -> xr.Dataset:
        if variable not in self._cache:
            path = self.zarr_dir / f"{variable}_annual.zarr"
            if not path.exists():
                raise FileNotFoundError(f"Zarr store not found: {path}")
            ds = xr.open_zarr(path, consolidated=False)
            self._cache[variable] = ds
            # Pre-extract lat/lon grids for fast nearest lookup
            lat = np.asarray(ds["lat"])
            lon = np.asarray(ds["lon"])
            self._latlon[variable] = (lat, lon)
        return self._cache[variable]

    def _nearest_index(self, variable: str, lat: float, lon: float) -> tuple[int, int]:
        lat_grid, lon_grid = self._latlon[variable]
        # Use a simple Euclidean distance in degrees (good enough for Germany)
        dist_sq = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
        y_idx, x_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        return int(y_idx), int(x_idx)

    def get_point_timeseries(
        self, variable: str, stat: str, lat: float, lon: float
    ) -> pd.Series:
        ds = self._load(variable)
        if stat not in ds.data_vars:
            raise ValueError(f"Statistic '{stat}' not available for {variable}")
        y_idx, x_idx = self._nearest_index(variable, lat, lon)
        da = ds[stat].isel(y=y_idx, x=x_idx)
        series = da.to_series()
        series.index = series.index.astype(int)
        series.name = f"{variable}_{stat}"
        return series

    def get_cell_coordinates(self, variable: str, lat: float, lon: float) -> dict[str, float]:
        """Return the actual lat/lon of the nearest grid cell."""
        lat_grid, lon_grid = self._latlon[variable]
        y_idx, x_idx = self._nearest_index(variable, lat, lon)
        return {"lat": float(lat_grid[y_idx, x_idx]), "lon": float(lon_grid[y_idx, x_idx])}

    def close(self) -> None:
        for ds in self._cache.values():
            ds.close()
        self._cache.clear()
        self._latlon.clear()


# ---------------------------------------------------------------------------
# Stations SQLite
# ---------------------------------------------------------------------------

class StationDataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def list_stations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM stations")
            return [dict(row) for row in cur.fetchall()]

    def get_nearest(self, lat: float, lon: float, variable: str | None = None) -> dict[str, Any]:
        # Haversine distance across all stations
        if variable is not None and variable not in {"temp", "sunshine", "precip"}:
            raise ValueError(f"Unknown station variable: {variable}")

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if variable:
                cur = conn.execute(
                    f"""
                    SELECT *
                    FROM stations
                    WHERE EXISTS (
                        SELECT 1
                        FROM timeseries
                        WHERE timeseries.station_id = stations.id
                          AND timeseries.{variable} IS NOT NULL
                    )
                    """
                )
            else:
                cur = conn.execute("SELECT * FROM stations")
            rows = cur.fetchall()
            if not rows:
                detail = f"No stations with {variable} data in database" if variable else "No stations in database"
                raise ValueError(detail)

        best = None
        best_dist = float("inf")
        skipped_without_coordinates = 0
        for row in rows:
            row_dict = dict(row)
            s_lat = row_dict.get("lat")
            s_lon = row_dict.get("lon")
            if s_lat is None or s_lon is None:
                skipped_without_coordinates += 1
                continue
            dist = haversine(lat, lon, s_lat, s_lon)
            if dist < best_dist:
                best_dist = dist
                best = row_dict

        if best is None:
            if skipped_without_coordinates:
                raise ValueError(
                    "Station coordinates are missing. Run python3 data_viewer/backend/preprocessing.py --station-coordinates-only."
                )
            raise ValueError("Could not find nearest station")
        best["distance_km"] = round(best_dist, 2)
        return best

    def get_timeseries(self, station_id: str, variable: str) -> pd.Series:
        col = variable
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT year, {col} FROM timeseries WHERE station_id = ? AND {col} IS NOT NULL ORDER BY year",
                (station_id,),
            )
            rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No data for station {station_id} variable {variable}")
        years, values = zip(*rows)
        return pd.Series(values, index=pd.Index(years, name="year"), name=variable)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)
