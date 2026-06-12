"""Data download, parsing, and preprocessing."""

from __future__ import annotations

import io
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy import stats

from .config import (
    CLIMATE_VARIABLES,
    DEFAULT_STATIONS,
    DWD_ANNUAL_KL_URL,
    DWD_DAILY_KL_URL,
    GMD_SSP_SUPPLEMENT_URL,
    HYRAS_TASMAX_URL,
    NOAA_CO2_URL,
    SSP_SCENARIOS,
)

def http_get(url: str, timeout: int = 60) -> bytes:
    """GET helper with clear errors and a browser-like user agent."""
    headers = {"User-Agent": "climate-var-vecm-forecast/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    return response.content

def list_dwd_zip_urls(base_url: str) -> list[str]:
    html = http_get(base_url).decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'href="([^"]+\.zip)"', html, flags=re.I)))
    return [base_url + name for name in names]

def list_hyras_tasmax_urls(version: str = "v6-1") -> dict[int, str]:
    html = http_get(HYRAS_TASMAX_URL).decode("utf-8", errors="replace")
    pattern = rf'href="(tasmax_hyras_1_(\d{{4}})_{re.escape(version)}_de\.nc)"'
    files = {}
    for name, year in re.findall(pattern, html, flags=re.I):
        files[int(year)] = HYRAS_TASMAX_URL + name
    if not files:
        raise RuntimeError(f"No HYRAS tasmax NetCDF files found for version {version}.")
    return dict(sorted(files.items()))

def station_id_from_zip_name(url: str) -> str | None:
    # Common DWD pattern: jahreswerte_KL_03987_...
    match = re.search(r"_(\d{5})_", Path(url).name)
    return match.group(1) if match else None

def open_product_table_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        product_names = [
            name
            for name in zf.namelist()
            if Path(name).name.lower().startswith("produkt_") and name.lower().endswith(".txt")
        ]
        if not product_names:
            raise ValueError("No DWD product TXT found in ZIP.")
        klima_products = [
            name
            for name in product_names
            if "klima" in Path(name).name.lower() or "jahreswerte" in Path(name).name.lower()
        ]
        if klima_products:
            product_names = klima_products
        # Prefer the largest matching product file if multiple files exist.
        product_name = max(product_names, key=lambda name: zf.getinfo(name).file_size)
        with zf.open(product_name) as fh:
            text = fh.read().decode("latin1", errors="replace")

    first_line = text.splitlines()[0]
    sep = ";" if ";" in first_line else r"\s+"
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df.columns = [str(col).strip() for col in df.columns]
    return df

def clean_dwd_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.upper() in {"STATIONS_ID", "MESS_DATUM", "QN_4", "QN_J"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            continue
        try:
            out[col] = pd.to_numeric(out[col])
        except (TypeError, ValueError):
            pass
    out = out.replace([-999, -999.0, -9999, -9999.0], np.nan)
    return out

def find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    col_map = {col.upper(): col for col in columns}
    for cand in candidates:
        if cand.upper() in col_map:
            return col_map[cand.upper()]
    return None

def parse_dwd_annual_station(zip_url: str) -> pd.DataFrame:
    station_id = station_id_from_zip_name(zip_url)
    raw = clean_dwd_numeric(open_product_table_from_zip(http_get(zip_url)))

    date_col = find_first_existing(raw.columns, ["MESS_DATUM", "MESS_DATUM_BEGINN", "MESS_DATUM_ENDE", "JAHR"])
    if date_col is None:
        raise ValueError(f"No date column found in {zip_url}")

    # Annual KL files usually contain JA_MX_TX (annual absolute maximum
    # temperature), JA_TX (annual mean of daily maximum temperature), JA_TT
    # (annual mean air temperature), and JA_SD_S (annual sunshine duration).
    temp_col = find_first_existing(
        raw.columns,
        ["JA_MX_TX", "MX_TX", "JA_TX", "TXK_JAHR", "TXK", "JA_TT", "TTK_JAHR", "TTK"],
    )
    sun_col = find_first_existing(
        raw.columns,
        ["JA_SD_S", "SDK_JAHR", "SDK", "SD_JAHR", "JA_SDK"],
    )
    precip_col = find_first_existing(
        raw.columns,
        ["JA_RR", "RRK_JAHR", "RRK", "JA_NIEDERSCHLAG"],
    )
    if temp_col is None or sun_col is None or precip_col is None:
        raise ValueError(
            f"Required temperature/sunshine/precipitation columns missing in {zip_url}. "
            f"Available: {list(raw.columns)}"
        )

    years = pd.to_numeric(raw[date_col], errors="coerce").astype("Int64")
    # MESS_DATUM may be YYYY or YYYYMMDD in DWD products.
    years = (years // 10000).where(years > 9999, years).astype("Int64")

    parsed = pd.DataFrame(
        {
            "year": years,
            "temp": pd.to_numeric(raw[temp_col], errors="coerce"),
            "sunshine": pd.to_numeric(raw[sun_col], errors="coerce"),
            "precip": pd.to_numeric(raw[precip_col], errors="coerce"),
            "station_id": station_id,
            "station_name": DEFAULT_STATIONS.get(station_id or "", station_id),
            "temp_source_column": temp_col,
            "sun_source_column": sun_col,
            "precip_source_column": precip_col,
        }
    )
    return parsed.dropna(subset=["year"]).set_index("year").sort_index()

def parse_dwd_daily_station_to_annual(zip_url: str) -> pd.DataFrame:
    """Optional path: annual maximum of daily TXK and annual sunshine sum."""
    station_id = station_id_from_zip_name(zip_url)
    raw = clean_dwd_numeric(open_product_table_from_zip(http_get(zip_url)))
    date_col = find_first_existing(raw.columns, ["MESS_DATUM"])
    tx_col = find_first_existing(raw.columns, ["TXK"])
    sun_col = find_first_existing(raw.columns, ["SDK"])
    precip_col = find_first_existing(raw.columns, ["RSK", "RR"])
    if date_col is None or tx_col is None or sun_col is None or precip_col is None:
        raise ValueError(f"Daily TXK/SDK/RSK columns missing in {zip_url}")

    date = pd.to_datetime(raw[date_col].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    daily = pd.DataFrame(
        {
            "date": date,
            "temp": pd.to_numeric(raw[tx_col], errors="coerce"),
            "sunshine": pd.to_numeric(raw[sun_col], errors="coerce"),
            "precip": pd.to_numeric(raw[precip_col], errors="coerce"),
        }
    ).dropna(subset=["date"])
    annual = daily.groupby(daily["date"].dt.year).agg(
        temp=("temp", "max"),
        sunshine=("sunshine", "sum"),
        precip=("precip", "sum"),
    )
    annual["station_id"] = station_id
    annual["station_name"] = DEFAULT_STATIONS.get(station_id or "", station_id)
    annual["temp_source_column"] = tx_col
    annual["sun_source_column"] = sun_col
    annual["precip_source_column"] = precip_col
    annual.index.name = "year"
    return annual

def fetch_dwd_climate(
    station_ids: list[str] | None,
    use_daily_max: bool = False,
    min_year: int = 1950,
    temp_aggregation: str = "max",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and average representative DWD stations."""
    base_url = DWD_DAILY_KL_URL if use_daily_max else DWD_ANNUAL_KL_URL
    zip_urls = list_dwd_zip_urls(base_url)
    by_station = {station_id_from_zip_name(url): url for url in zip_urls}
    selected_station_ids = sorted(sid for sid in by_station if sid) if station_ids is None else station_ids

    station_frames = []
    for station_id in selected_station_ids:
        url = by_station.get(station_id.zfill(5))
        if not url:
            logging.warning("DWD station %s not found in %s", station_id, base_url)
            continue
        try:
            frame = (
                parse_dwd_daily_station_to_annual(url)
                if use_daily_max
                else parse_dwd_annual_station(url)
            )
            frame = frame.loc[frame.index >= min_year]
            if frame[CLIMATE_VARIABLES].dropna().empty:
                logging.warning("DWD station %s has no usable rows after %s", station_id, min_year)
                continue
            station_frames.append(frame)
            logging.info(
                "Loaded DWD station %s (%s), %s-%s",
                station_id,
                DEFAULT_STATIONS.get(station_id, "unknown"),
                int(frame.index.min()),
                int(frame.index.max()),
            )
        except Exception as exc:
            logging.warning("Skipping DWD station %s: %s", station_id, exc)

    if not station_frames:
        raise RuntimeError("No DWD station could be loaded. Try different --stations.")

    station_data = pd.concat(station_frames, axis=0)
    grouped = station_data.groupby(level=0)
    climate = grouped[["sunshine", "precip"]].mean().sort_index()
    if temp_aggregation == "max":
        climate["temp"] = grouped["temp"].max()
    elif temp_aggregation == "mean":
        climate["temp"] = grouped["temp"].mean()
    else:
        raise ValueError(f"Unsupported temp_aggregation: {temp_aggregation}")
    climate = climate[CLIMATE_VARIABLES].sort_index()
    climate.index.name = "year"
    return climate, station_data

def require_xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "HYRAS NetCDF support requires xarray plus a NetCDF backend. "
            "Install for example: python3 -m pip install xarray h5netcdf"
        ) from exc
    return xr

def download_to_cache(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(url).name
    if path.exists() and path.stat().st_size > 0:
        return path
    logging.info("Downloading %s", Path(url).name)
    headers = {"User-Agent": "climate-var-vecm-forecast/1.0"}
    try:
        with requests.get(url, headers=headers, timeout=120, stream=True) as response:
            response.raise_for_status()
            tmp_path = path.with_suffix(path.suffix + ".part")
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            tmp_path.replace(path)
    except requests.RequestException as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    return path


def xlsx_shared_strings(xlsx: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", ns):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))
    return strings


def xlsx_cell_value(cell, shared_strings: list[str]) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    value = cell.find("a:v", ns)
    if value is None:
        return ""
    text = value.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(text)]
    return text


def xlsx_sheet_paths_by_name(xlsx: zipfile.ZipFile) -> dict[str, str]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib["Id"]: "xl/" + rel.attrib["Target"].lstrip("/")
        for rel in rels.findall("rel:Relationship", ns)
    }
    return {
        sheet.attrib["name"]: targets[sheet.attrib[f"{{{ns['r']}}}id"]]
        for sheet in workbook.find("a:sheets", ns)
    }


def normalize_scenario_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_world_co2_from_ssp_sheet(xlsx: zipfile.ZipFile, sheet_path: str) -> pd.Series:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared_strings = xlsx_shared_strings(xlsx)
    root = ET.fromstring(xlsx.read(sheet_path))
    rows = {}
    for row in root.findall(".//a:sheetData/a:row", ns):
        row_idx = int(row.attrib["r"])
        rows[row_idx] = {
            re.match(r"[A-Z]+", cell.attrib["r"]).group(0): xlsx_cell_value(cell, shared_strings)
            for cell in row.findall("a:c", ns)
        }

    gas_row, unit_row, region_row = rows[9], rows[10], rows[11]
    co2_col = None
    for col, gas in gas_row.items():
        if gas == "CO2" and unit_row.get(col) == "ppm" and region_row.get(col) == "World":
            co2_col = col
            break
    if co2_col is None:
        raise RuntimeError(f"Could not find world CO2 ppm column in SSP sheet {sheet_path}.")

    values = {}
    for row_idx in sorted(idx for idx in rows if idx >= 13):
        row = rows[row_idx]
        if not row.get("A") or not row.get(co2_col):
            continue
        try:
            year = int(float(row["A"]))
            value = float(row[co2_col])
        except ValueError:
            continue
        values[year] = value
    if not values:
        raise RuntimeError(f"No CO2 values found in SSP sheet {sheet_path}.")
    series = pd.Series(values, name="co2").sort_index()
    series.index.name = "year"
    return series


def fetch_ssp_co2_path(
    scenario: str,
    cache_dir: Path,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """Fetch CMIP6/ScenarioMIP CO2 concentrations from Meinshausen et al. 2020."""
    scenario_key = scenario.lower()
    if scenario_key not in SSP_SCENARIOS:
        raise ValueError(f"Unsupported SSP scenario: {scenario}")

    supplement_zip = download_to_cache(GMD_SSP_SUPPLEMENT_URL, cache_dir)
    with zipfile.ZipFile(supplement_zip) as supplement:
        xlsx_names = [name for name in supplement.namelist() if name.endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError("GMD SSP supplement contains no XLSX data table.")
        xlsx_bytes = supplement.read(xlsx_names[0])

    target = normalize_scenario_name(SSP_SCENARIOS[scenario_key])
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as xlsx:
        sheet_paths = xlsx_sheet_paths_by_name(xlsx)
        matches = [
            (sheet_name, path)
            for sheet_name, path in sheet_paths.items()
            if target in normalize_scenario_name(sheet_name)
        ]
        if not matches:
            raise RuntimeError(f"Could not find sheet for {SSP_SCENARIOS[scenario_key]} in GMD SSP supplement.")
        co2 = extract_world_co2_from_ssp_sheet(xlsx, matches[0][1])

    requested_years = pd.Index(range(start_year, end_year + 1), name="year")
    co2 = co2.reindex(co2.index.union(requested_years)).interpolate(method="linear")
    co2 = co2.reindex(requested_years)
    if co2.isna().any():
        raise RuntimeError(
            f"SSP scenario {scenario} does not cover requested years {start_year}-{end_year}."
        )
    return co2

def infer_hyras_tasmax_variable(dataset) -> str:
    candidates = [name for name in dataset.data_vars if "tasmax" in name.lower()]
    if not candidates:
        candidates = [name for name in dataset.data_vars if "temperature" in name.lower()]
    if not candidates:
        candidates = list(dataset.data_vars)
    if not candidates:
        raise RuntimeError("HYRAS NetCDF file contains no data variables.")
    return candidates[0]

def maybe_kelvin_to_celsius(value: float, units: str) -> float:
    units_lower = units.lower()
    if units_lower in {"k", "kelvin"} or value > 100:
        return value - 273.15
    return value

def fetch_hyras_tasmax(
    start_year: int,
    end_year: int | None,
    cache_dir: Path,
    version: str = "v6-1",
    aggregation: str = "annual_spatial_max",
) -> pd.Series:
    """Fetch HYRAS daily gridded tasmax and aggregate to annual values."""
    xr = require_xarray()
    urls = list_hyras_tasmax_urls(version=version)
    if end_year is None:
        end_year = max(urls)
    years = [year for year in sorted(urls) if max(start_year, 1951) <= year <= end_year]
    if not years:
        raise RuntimeError(f"No HYRAS tasmax files available for requested period {start_year}-{end_year}.")

    values = {}
    for year in years:
        path = download_to_cache(urls[year], cache_dir)
        try:
            with xr.open_dataset(path) as ds:
                var_name = infer_hyras_tasmax_variable(ds)
                tasmax = ds[var_name]
                if aggregation == "annual_spatial_max":
                    value = float(tasmax.max(skipna=True).values)
                elif aggregation == "annual_mean_daily_spatial_mean":
                    value = float(tasmax.mean(skipna=True).values)
                else:
                    raise ValueError(f"Unsupported HYRAS aggregation: {aggregation}")
                values[year] = maybe_kelvin_to_celsius(value, str(tasmax.attrs.get("units", "")))
        except Exception as exc:
            raise RuntimeError(f"Could not read HYRAS file {path}: {exc}") from exc
        logging.info("HYRAS tasmax %s = %.2f deg C", year, values[year])

    series = pd.Series(values, name="temp").sort_index()
    series.index.name = "year"
    return series

def fetch_noaa_co2(min_year: int = 1950) -> pd.Series:
    raw = http_get(NOAA_CO2_URL).decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append(line)
    df = pd.read_csv(io.StringIO("\n".join(rows)))
    df.columns = [col.strip().lower() for col in df.columns]

    year_col = find_first_existing(df.columns, ["year"])
    mean_col = find_first_existing(df.columns, ["mean", "co2", "annmean"])
    if year_col is None or mean_col is None:
        raise RuntimeError(f"Unexpected NOAA CO2 columns: {list(df.columns)}")

    co2 = pd.Series(
        pd.to_numeric(df[mean_col], errors="coerce").values,
        index=pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
        name="co2",
    ).dropna()
    co2 = co2[co2 > 0].sort_index()

    if co2.index.min() > min_year:
        # Mauna Loa starts in 1959. For alignment from 1950, backcast with the
        # earliest local linear trend. This is a documented proxy, not a
        # reconstruction.
        first_years = co2.iloc[: min(10, len(co2))]
        slope, intercept, *_ = stats.linregress(first_years.index.astype(float), first_years.values)
        extra_years = np.arange(min_year, int(co2.index.min()))
        extra = pd.Series(intercept + slope * extra_years, index=extra_years, name="co2")
        co2 = pd.concat([extra, co2])

    return co2.loc[co2.index >= min_year].sort_index()

def align_and_clean(climate: pd.DataFrame, co2: pd.Series, start_year: int, end_year: int | None) -> pd.DataFrame:
    data = climate.join(co2, how="outer")
    if end_year is None:
        observed_max_years = [
            int(data[col].dropna().index.max())
            for col in data.columns
            if not data[col].dropna().empty
        ]
        end_year = int(min(min(observed_max_years), pd.Timestamp.today().year - 1))
    data = data.loc[(data.index >= start_year) & (data.index <= end_year)]
    full_index = pd.Index(range(start_year, end_year + 1), name="year")
    data = data.reindex(full_index)

    missing_before = data.isna().sum()
    data = data.interpolate(method="linear", limit=3, limit_area="inside")
    data = data.dropna()
    missing_after = data.isna().sum()
    logging.info("Missing values before interpolation: %s", missing_before.to_dict())
    logging.info("Missing values after cleaning: %s", missing_after.to_dict())
    return data
