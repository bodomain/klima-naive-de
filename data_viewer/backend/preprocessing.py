#!/usr/bin/env python3
"""Preprocess HYRAS NetCDFs and DWD stations into queryable formats."""

from __future__ import annotations

import io
import argparse
import logging
import re
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import xarray as xr

# Add parent package for reuse
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "modeling"))
from climate_analysis.config import (
    DWD_ANNUAL_KL_URL,
    HYRAS_TASMAX_URL,
    DEFAULT_STATIONS,
)
from climate_analysis.data import (
    http_get,
    list_dwd_zip_urls,
    station_id_from_zip_name,
    open_product_table_from_zip,
    clean_dwd_numeric,
    find_first_existing,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CACHE = WORKSPACE / "data_cache"
ZARR_DIR = DATA_CACHE / "hyras_zarr"
DB_PATH = DATA_CACHE / "stations.db"

HYRAS_VARS = {
    "tasmax": {
        "url_base": HYRAS_TASMAX_URL,
        "cache_dir": DATA_CACHE / "hyras_tasmax",
        "stats": ["annual_max", "annual_mean"],
    },
    "tas": {
        "url_base": (
            "https://opendata.dwd.de/climate_environment/CDC/"
            "grids_germany/daily/hyras_de/air_temperature_mean/"
        ),
        "cache_dir": DATA_CACHE / "hyras_tas",
        "stats": ["annual_mean"],
    },
    "pr": {
        "url_base": (
            "https://opendata.dwd.de/climate_environment/CDC/"
            "grids_germany/daily/hyras_de/precipitation/"
        ),
        "cache_dir": DATA_CACHE / "hyras_pr",
        "stats": ["annual_sum", "annual_mean"],
    },
}

# ---------------------------------------------------------------------------
# HYRAS helpers
# ---------------------------------------------------------------------------

def list_hyras_yearly_files(url_base: str, version: str = "v6-1") -> dict[int, str]:
    html = http_get(url_base).decode("utf-8", errors="replace")
    # file pattern example: tasmax_hyras_1_2024_v6-1_de.nc
    pattern = rf'href="([^"]+_{re.escape(version)}_de\.nc)"'
    files: dict[int, str] = {}
    for name in re.findall(pattern, html, flags=re.I):
        m = re.search(r"_(\d{4})_", name)
        if m:
            files[int(m.group(1))] = url_base + name
    return dict(sorted(files.items()))


def station_metadata_url(base_url: str) -> str | None:
    """Return the DWD station description file URL from a CDC directory."""
    html = http_get(base_url).decode("utf-8", errors="replace")
    matches = re.findall(r'href="([^"]*Beschreibung_Stationen[^"]*\.txt)"', html, flags=re.I)
    if not matches:
        return None
    # Prefer the KL-specific description if present.
    matches = sorted(set(matches), key=lambda name: ("KL" not in name.upper(), name))
    name = matches[0]
    return name if name.startswith("http") else base_url + name


def parse_station_metadata(text: str) -> dict[str, dict[str, float | str]]:
    """Parse DWD station descriptions into coordinates keyed by station id."""
    stations: dict[str, dict[str, float | str]] = {}
    for line in text.splitlines():
        match = re.match(
            r"^\s*(\d{1,5})\s+\d{8}\s+\d{8}\s+"
            r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(.+?)\s{2,}.+$",
            line,
        )
        if not match:
            continue
        station_id, elevation, lat, lon, name = match.groups()
        stations[station_id.zfill(5)] = {
            "elevation": float(elevation),
            "lat": float(lat),
            "lon": float(lon),
            "name": name.strip(),
        }
    return stations


def load_station_metadata() -> dict[str, dict[str, float | str]]:
    url = station_metadata_url(DWD_ANNUAL_KL_URL)
    if url is None:
        logging.warning("No DWD station description file found")
        return {}
    try:
        text = http_get(url).decode("latin1", errors="replace")
    except Exception as exc:
        logging.warning("Could not load DWD station metadata: %s", exc)
        return {}
    metadata = parse_station_metadata(text)
    logging.info("Loaded coordinates for %d DWD stations", len(metadata))
    return metadata


def update_existing_station_coordinates() -> None:
    """Populate coordinates in an existing stations.db without rebuilding time series."""
    metadata = load_station_metadata()
    if not metadata:
        raise RuntimeError("No station metadata available")
    with sqlite3.connect(DB_PATH) as conn:
        updated = 0
        for sid, info in metadata.items():
            cur = conn.execute(
                """
                UPDATE stations
                SET name = CASE WHEN name IS NULL OR name = '' THEN ? ELSE name END,
                    lat = ?,
                    lon = ?,
                    elevation = ?
                WHERE id = ?
                """,
                (info["name"], info["lat"], info["lon"], info["elevation"], sid),
            )
            updated += cur.rowcount
        conn.commit()
    logging.info("Updated coordinates for %d stations in %s", updated, DB_PATH)


def download_to_cache(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(url).name
    if path.exists() and path.stat().st_size > 0:
        return path
    logging.info("Downloading %s", Path(url).name)
    headers = {"User-Agent": "hyras-data-viewer/1.0"}
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, headers=headers, timeout=300, stream=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def infer_hyras_variable(ds: xr.Dataset) -> str:
    candidates = [v for v in ds.data_vars if v not in ds.coords]
    if not candidates:
        raise RuntimeError("No data variables found")
    # Prefer known names
    preferred = ["tasmax", "tas", "pr", "hurs", "rsds"]
    for p in preferred:
        for c in candidates:
            if p.lower() in c.lower():
                return c
    return candidates[0]


def compute_annual_stats(ds: xr.Dataset, var_name: str, stats: list[str]) -> xr.Dataset:
    """Given a single-year daily dataset, compute requested annual stats."""
    da = ds[var_name]
    out_vars: dict[str, xr.DataArray] = {}
    for stat in stats:
        if stat == "annual_max":
            out_vars[stat] = da.max(dim="time", skipna=True)
        elif stat == "annual_mean":
            out_vars[stat] = da.mean(dim="time", skipna=True)
        elif stat == "annual_sum":
            out_vars[stat] = da.sum(dim="time", skipna=True)
        else:
            raise ValueError(f"Unknown stat: {stat}")
    out = xr.Dataset(out_vars)
    # Keep spatial coordinates
    for coord in ["y", "x", "lat", "lon", "crs"]:
        if coord in ds.coords:
            out = out.assign_coords({coord: ds.coords[coord]})
    return out


def process_hyras_variable(var_key: str) -> None:
    cfg = HYRAS_VARS[var_key]
    files = list_hyras_yearly_files(cfg["url_base"])
    logging.info("%s: found %d years (%s-%s)", var_key, len(files), min(files), max(files))

    # Download all missing files first
    paths: dict[int, Path] = {}
    for year, url in files.items():
        paths[year] = download_to_cache(url, cfg["cache_dir"])

    # Build annual datasets year by year
    annual_datasets: list[xr.Dataset] = []
    for year in sorted(paths):
        path = paths[year]
        try:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                var_name = infer_hyras_variable(ds)
                annual_ds = compute_annual_stats(ds, var_name, cfg["stats"])
                annual_ds = annual_ds.expand_dims(year=[year])
                annual_datasets.append(annual_ds)
                logging.info("%s %s processed", var_key, year)
        except Exception as exc:
            logging.warning("Failed to process %s %s: %s", var_key, year, exc)

    if not annual_datasets:
        raise RuntimeError(f"No annual datasets built for {var_key}")

    combined = xr.concat(annual_datasets, dim="year")
    zarr_path = ZARR_DIR / f"{var_key}_annual.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    # Remove existing zarr if present to avoid conflicts
    if any(zarr_path.iterdir()):
        import shutil
        shutil.rmtree(zarr_path)
    combined.to_zarr(zarr_path, mode="w")
    logging.info("%s: Zarr written to %s", var_key, zarr_path)


# ---------------------------------------------------------------------------
# Stations database
# ---------------------------------------------------------------------------

def build_stations_db() -> None:
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE stations (
            id TEXT PRIMARY KEY,
            name TEXT,
            lat REAL,
            lon REAL,
            elevation REAL,
            start_year INTEGER,
            end_year INTEGER,
            has_temp INTEGER,
            has_sunshine INTEGER,
            has_precip INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE timeseries (
            station_id TEXT,
            year INTEGER,
            temp REAL,
            sunshine REAL,
            precip REAL,
            temp_source TEXT,
            sun_source TEXT,
            precip_source TEXT,
            PRIMARY KEY (station_id, year)
        )
        """
    )

    zip_urls = list_dwd_zip_urls(DWD_ANNUAL_KL_URL)
    logging.info("Stations: found %d ZIP files", len(zip_urls))
    station_metadata = load_station_metadata()

    inserted_stations = 0
    inserted_rows = 0

    for url in zip_urls:
        sid = station_id_from_zip_name(url)
        if not sid:
            continue
        try:
            raw = clean_dwd_numeric(open_product_table_from_zip(http_get(url)))
        except Exception as exc:
            logging.warning("Skipping station %s: %s", sid, exc)
            continue

        date_col = find_first_existing(
            raw.columns, ["MESS_DATUM", "MESS_DATUM_BEGINN", "MESS_DATUM_ENDE", "JAHR"]
        )
        temp_col = find_first_existing(
            raw.columns, ["JA_MX_TX", "MX_TX", "JA_TX", "TXK_JAHR", "TXK", "JA_TT", "TTK_JAHR", "TTK"]
        )
        sun_col = find_first_existing(
            raw.columns, ["JA_SD_S", "SDK_JAHR", "SDK", "SD_JAHR", "JA_SDK"]
        )
        precip_col = find_first_existing(
            raw.columns, ["JA_RR", "RRK_JAHR", "RRK", "JA_NIEDERSCHLAG"]
        )

        if date_col is None:
            continue

        years = pd.to_numeric(raw[date_col], errors="coerce").astype("Int64")
        years = (years // 10000).where(years > 9999, years).astype("Int64")

        has_temp = int(temp_col is not None)
        has_sun = int(sun_col is not None)
        has_precip = int(precip_col is not None)

        valid_years = years.dropna()
        if valid_years.empty:
            continue

        start_year = int(valid_years.min())
        end_year = int(valid_years.max())
        meta = station_metadata.get(sid, {})
        name = DEFAULT_STATIONS.get(sid, str(meta.get("name", "")))
        lat = meta.get("lat")
        lon = meta.get("lon")
        elevation = meta.get("elevation")

        conn.execute(
            "INSERT INTO stations VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, name, lat, lon, elevation, start_year, end_year, has_temp, has_sun, has_precip),
        )
        inserted_stations += 1

        for _, row in raw.iterrows():
            y = int(row[date_col]) if pd.notna(row[date_col]) else None
            if y is None:
                continue
            if y > 9999:
                y = y // 10000
            t = float(row[temp_col]) if temp_col and pd.notna(row.get(temp_col)) else None
            s = float(row[sun_col]) if sun_col and pd.notna(row.get(sun_col)) else None
            p = float(row[precip_col]) if precip_col and pd.notna(row.get(precip_col)) else None
            if t == -999 or t == -9999:
                t = None
            if s == -999 or s == -9999:
                s = None
            if p == -999 or p == -9999:
                p = None
            conn.execute(
                "INSERT OR REPLACE INTO timeseries VALUES (?,?,?,?,?,?,?,?)",
                (sid, y, t, s, p, temp_col, sun_col, precip_col),
            )
            inserted_rows += 1

    conn.commit()
    conn.close()
    logging.info(
        "Stations DB built: %s stations, %s rows", inserted_stations, inserted_rows
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-coordinates-only",
        action="store_true",
        help="Update station lat/lon/elevation in the existing SQLite database and exit.",
    )
    args = parser.parse_args()

    if args.station_coordinates_only:
        update_existing_station_coordinates()
        return

    ZARR_DIR.mkdir(parents=True, exist_ok=True)

    for var_key in ["tasmax", "tas", "pr"]:
        logging.info("=== Processing %s ===", var_key)
        try:
            process_hyras_variable(var_key)
        except Exception:
            logging.exception("Failed to process %s", var_key)

    logging.info("=== Building stations DB ===")
    build_stations_db()
    logging.info("=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
