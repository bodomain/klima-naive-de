#!/usr/bin/env python3
"""
Naive multivariate climate time-series analysis for Germany.

Data:
  - DWD CDC annual climate observations, representative long-running stations
  - NOAA GML Mauna Loa annual mean atmospheric CO2

Outputs:
  - Cleaned annual data CSV
  - Trend plots
  - ADF and Johansen tables
  - IRF and FEVD plots
  - Naive forecast to 2100 with uncertainty bands
  - Markdown report

This script is intentionally conservative: it documents proxy choices and uses
simple, reproducible assumptions. The long-run forecast is illustrative, not a
physical climate projection.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

try:
    import seaborn as sns
except ImportError:  # Optional styling dependency.
    sns = None


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

DEFAULT_STATIONS = {
    "03987": "Potsdam",
    "02290": "Hohenpeissenberg",
    "00433": "Berlin-Dahlem",
    "01048": "Dresden-Klotzsche",
}


@dataclass
class ModelBundle:
    model_type: str
    result: object
    data_used: pd.DataFrame
    lag_order: int
    coint_rank: int
    integration_order: dict[str, int]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
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
    logging.info("Downloading HYRAS file %s", Path(url).name)
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


def rolling_trend(series: pd.Series, window: int = 11) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=max(3, window // 2)).mean()


def add_source_note(fig, note: str = SOURCE_NOTE) -> None:
    fig.text(0.5, 0.965, note, ha="center", va="top", fontsize=9, color="#333333")


def plot_raw_trends(data: pd.DataFrame, output: Path) -> None:
    plot_cols = [col for col in MODEL_VARIABLES if col in data.columns]
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, 2.8 * len(plot_cols)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, plot_cols):
        ax.plot(data.index, data[col], color=VARIABLE_COLORS[col], lw=1.2, label="Annual")
        ax.plot(data.index, rolling_trend(data[col]), color="black", lw=2, label="11-year moving average")
        ax.set_ylabel(VARIABLE_LABELS[col])
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Historical climate series and NOAA CO2", y=0.995)
    add_source_note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def adf_table(data: pd.DataFrame, max_diff: int = 2) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    integration_order = {}
    for col in data.columns:
        current = data[col].dropna()
        order = None
        for diff in range(max_diff + 1):
            test_series = current if diff == 0 else current.diff(diff).dropna()
            result = adfuller(test_series, autolag="AIC")
            rows.append(
                {
                    "variable": col,
                    "difference_order": diff,
                    "adf_statistic": result[0],
                    "p_value": result[1],
                    "used_lags": result[2],
                    "n_obs": result[3],
                    "stationary_5pct": bool(result[1] < 0.05),
                }
            )
            if order is None and result[1] < 0.05:
                order = diff
        integration_order[col] = max_diff + 1 if order is None else order
    return pd.DataFrame(rows), integration_order


def johansen_table(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> tuple[pd.DataFrame, int]:
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    rows = []
    rank = 0
    for i, trace_stat in enumerate(result.lr1):
        crit90, crit95, crit99 = result.cvt[i]
        rows.append(
            {
                "rank_null_r<=": i,
                "trace_statistic": trace_stat,
                "crit_90": crit90,
                "crit_95": crit95,
                "crit_99": crit99,
                "reject_95": bool(trace_stat > crit95),
            }
        )
        if trace_stat > crit95:
            rank = i + 1
    rank = min(rank, data.shape[1] - 1)
    return pd.DataFrame(rows), rank


def prepare_mixed_stationary_data(data: pd.DataFrame) -> pd.DataFrame:
    """Stationary mixed specification for climate levels plus CO2 growth."""
    mixed = pd.DataFrame(index=data.index)
    for col in CLIMATE_VARIABLES:
        if col in data.columns:
            mixed[col] = data[col]
    mixed["d_co2"] = data["co2"].diff()
    return mixed.dropna()


def choose_lag_order(data: pd.DataFrame, maxlags: int = 6) -> int:
    selected = VAR(data).select_order(maxlags=maxlags)
    lag = selected.selected_orders.get("aic")
    if lag is None or lag < 1:
        lag = 1
    return int(lag)


def fit_model(
    data: pd.DataFrame,
    coint_rank: int,
    integration_order: dict[str, int],
    maxlags: int = 6,
) -> ModelBundle:
    all_i1 = all(order == 1 for order in integration_order.values())
    if all_i1 and coint_rank > 0:
        order = select_order(data, maxlags=maxlags, deterministic="ci")
        k_ar_diff = order.selected_orders.get("aic")
        if k_ar_diff is None:
            k_ar_diff = 1
        k_ar_diff = max(1, int(k_ar_diff))
        result = VECM(
            data,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic="ci",
        ).fit()
        logging.info("Fitted VECM with k_ar_diff=%s and rank=%s", k_ar_diff, coint_rank)
        return ModelBundle("VECM", result, data, k_ar_diff, coint_rank, integration_order)

    if not all_i1:
        mixed = prepare_mixed_stationary_data(data)
        lag = choose_lag_order(mixed, maxlags=maxlags)
        result = VAR(mixed).fit(lag)
        logging.info(
            "Fitted MIXED_VAR with climate levels and d_co2, lag=%s; integration orders=%s",
            lag,
            integration_order,
        )
        return ModelBundle("MIXED_VAR_LEVELS_DCO2", result, mixed, lag, 0, integration_order)

    diff = data.diff().dropna()
    lag = choose_lag_order(diff, maxlags=maxlags)
    result = VAR(diff).fit(lag)
    logging.info("Fitted VAR on first differences with lag=%s", lag)
    return ModelBundle("VAR_DIFF", result, diff, lag, 0, integration_order)


def save_irf_plot(bundle: ModelBundle, output: Path, periods: int = 20) -> None:
    try:
        irf = bundle.result.irf(periods)
        fig = irf.plot(orth=False)
        fig.set_size_inches(12, 9)
        fig.suptitle(f"Impulse response functions ({bundle.model_type})", y=0.995)
        add_source_note(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(output, dpi=160)
        plt.close(fig)
    except Exception as exc:
        logging.warning("Could not create IRF plot: %s", exc)


def fit_auxiliary_var_levels(data: pd.DataFrame, maxlags: int = 6):
    lag = choose_lag_order(data, maxlags=maxlags)
    return VAR(data).fit(lag)


def save_fevd_plot(
    data: pd.DataFrame,
    output: Path,
    periods: int = 20,
    maxlags: int = 6,
    stationary_data: pd.DataFrame | None = None,
) -> None:
    try:
        var_res = fit_auxiliary_var_levels(stationary_data if stationary_data is not None else data, maxlags=maxlags)
        fevd = var_res.fevd(periods)
        fig = fevd.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle("Forecast error variance decomposition", y=0.995)
        add_source_note(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(output, dpi=160)
        plt.close(fig)
    except Exception as exc:
        logging.warning("Could not create FEVD plot: %s", exc)


def granger_tests(
    data: pd.DataFrame,
    integration_order: dict[str, int],
    maxlags: int = 6,
) -> pd.DataFrame:
    all_i1 = all(order == 1 for order in integration_order.values())
    if all_i1:
        test_data = data.diff().dropna()
        co2_name = "co2"
    else:
        test_data = prepare_mixed_stationary_data(data)
        co2_name = "d_co2"
    lag = choose_lag_order(test_data, maxlags=maxlags)
    res = VAR(test_data).fit(lag)
    tests = []
    test_specs = []
    climate_cols = [col for col in CLIMATE_VARIABLES if col in test_data.columns]
    for climate_col in climate_cols:
        test_specs.append((climate_col, [co2_name]))
        test_specs.append((co2_name, [climate_col]))
    for caused in climate_cols:
        for causing in climate_cols:
            if caused != causing:
                test_specs.append((caused, [causing]))

    for caused, causing in test_specs:
        try:
            test = res.test_causality(caused=caused, causing=causing, kind="f")
            tests.append(
                {
                    "caused": caused,
                    "causing": ",".join(causing),
                    "test_statistic": test.test_statistic,
                    "p_value": test.pvalue,
                    "df": str(test.df),
                    "reject_5pct": bool(test.pvalue < 0.05),
                }
            )
        except Exception as exc:
            tests.append(
                {
                    "caused": caused,
                    "causing": ",".join(causing),
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "df": "",
                    "reject_5pct": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(tests)


def scenario_co2_path(
    history: pd.Series,
    future_years: np.ndarray,
    lookback: int = 10,
    scenario: str = "linear",
) -> pd.Series:
    tail = history.dropna().iloc[-lookback:]
    years = tail.index.astype(float)
    if scenario == "linear":
        slope, intercept, *_ = stats.linregress(years, tail.values)
        path = intercept + slope * future_years
        # Prevent implausible negative increments if the selected period is unusual.
        if slope < 0:
            path = history.iloc[-1] + np.arange(1, len(future_years) + 1) * 0.0
    elif scenario == "exponential":
        slope, intercept, *_ = stats.linregress(years, np.log(tail.values))
        path = np.exp(intercept + slope * future_years)
        if slope < 0:
            path = history.iloc[-1] + np.arange(1, len(future_years) + 1) * 0.0
    else:
        raise ValueError(f"Unsupported CO2 scenario: {scenario}")
    return pd.Series(path, index=future_years, name="co2")


def historical_co2_sensitivities(data: pd.DataFrame) -> dict[str, float]:
    """Simple long-run OLS slopes used for scenario adjustment."""
    sensitivities = {}
    x = data["co2"].values
    for col in CLIMATE_VARIABLES:
        if col not in data.columns:
            continue
        slope, *_ = stats.linregress(x, data[col].values)
        sensitivities[col] = slope
    return sensitivities


def forecast_bundle(
    bundle: ModelBundle,
    levels_data: pd.DataFrame,
    horizon_end: int = 2100,
    alpha: float = 0.10,
    co2_scenario: str = "linear",
    co2_lookback: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    last_year = int(levels_data.index.max())
    future_years = np.arange(last_year + 1, horizon_end + 1)
    steps = len(future_years)
    columns = list(levels_data.columns)

    if bundle.model_type == "VECM":
        try:
            pred, lower, upper = bundle.result.predict(steps=steps, alpha=alpha)
        except Exception:
            pred = bundle.result.predict(steps=steps)
            # Fallback uncertainty from historical residual spread.
            resid_std = pd.DataFrame(bundle.result.resid, columns=columns).std().values
            z = stats.norm.ppf(1 - alpha / 2)
            lower = pred - z * np.sqrt(np.arange(1, steps + 1))[:, None] * resid_std
            upper = pred + z * np.sqrt(np.arange(1, steps + 1))[:, None] * resid_std
    elif bundle.model_type == "MIXED_VAR_LEVELS_DCO2":
        res = bundle.result
        pred_mixed, lower_mixed, upper_mixed = res.forecast_interval(
            bundle.data_used.values[-res.k_ar :],
            steps=steps,
            alpha=alpha,
        )
        mixed_columns = list(bundle.data_used.columns)
        mixed_forecast = pd.DataFrame(pred_mixed, index=future_years, columns=mixed_columns)
        mixed_lower = pd.DataFrame(lower_mixed, index=future_years, columns=mixed_columns)
        mixed_upper = pd.DataFrame(upper_mixed, index=future_years, columns=mixed_columns)

        pred_df = pd.DataFrame(index=future_years, columns=columns, dtype=float)
        lower_df = pd.DataFrame(index=future_years, columns=columns, dtype=float)
        upper_df = pd.DataFrame(index=future_years, columns=columns, dtype=float)
        for col in CLIMATE_VARIABLES:
            if col not in mixed_forecast.columns:
                continue
            pred_df[col] = mixed_forecast[col]
            lower_df[col] = mixed_lower[col]
            upper_df[col] = mixed_upper[col]

        co2_model_path = levels_data["co2"].iloc[-1] + mixed_forecast["d_co2"].cumsum()
        co2_lower_path = levels_data["co2"].iloc[-1] + mixed_lower["d_co2"].cumsum()
        co2_upper_path = levels_data["co2"].iloc[-1] + mixed_upper["d_co2"].cumsum()
        pred_df["co2"] = co2_model_path
        lower_df["co2"] = co2_lower_path
        upper_df["co2"] = co2_upper_path

        pred = pred_df[columns].values
        lower = lower_df[columns].values
        upper = upper_df[columns].values
    else:
        res = bundle.result
        diff_forecast, diff_lower, diff_upper = res.forecast_interval(
            bundle.data_used.values[-res.k_ar :],
            steps=steps,
            alpha=alpha,
        )
        last_level = levels_data.iloc[-1].values
        pred = last_level + np.cumsum(diff_forecast, axis=0)
        lower = last_level + np.cumsum(diff_lower, axis=0)
        upper = last_level + np.cumsum(diff_upper, axis=0)

    forecast = pd.DataFrame(pred, index=future_years, columns=columns)
    lower_df = pd.DataFrame(lower, index=future_years, columns=columns)
    upper_df = pd.DataFrame(upper, index=future_years, columns=columns)

    external_co2 = scenario_co2_path(
        levels_data["co2"],
        future_years,
        lookback=co2_lookback,
        scenario=co2_scenario,
    )
    co2_delta = external_co2 - forecast["co2"]
    sensitivities = historical_co2_sensitivities(levels_data)

    adjusted = forecast.copy()
    adjusted["co2"] = external_co2
    for col in CLIMATE_VARIABLES:
        if col not in adjusted.columns:
            continue
        adjusted[col] = adjusted[col] + sensitivities[col] * co2_delta
        lower_df[col] = lower_df[col] + sensitivities[col] * co2_delta
        upper_df[col] = upper_df[col] + sensitivities[col] * co2_delta
    lower_df["co2"] = external_co2 - (forecast["co2"] - lower_df["co2"]).abs()
    upper_df["co2"] = external_co2 + (upper_df["co2"] - forecast["co2"]).abs()

    return adjusted, lower_df, upper_df, external_co2


def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    output: Path,
) -> None:
    plot_cols = [col for col in MODEL_VARIABLES if col in history.columns]
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(13, 2.9 * len(plot_cols)), sharex=True)
    axes = np.atleast_1d(axes)
    transition = int(history.index.max())

    for ax, col in zip(axes, plot_cols):
        ax.plot(history.index, history[col], color=VARIABLE_COLORS[col], lw=1.5, label="Historical data")
        ax.plot(forecast.index, forecast[col], color=VARIABLE_COLORS[col], lw=1.8, ls="--", label="Naive forecast")
        ax.fill_between(
            forecast.index,
            lower[col].astype(float).values,
            upper[col].astype(float).values,
            color=VARIABLE_COLORS[col],
            alpha=0.18,
            label="90% confidence interval",
        )
        ax.axvline(transition, color="black", lw=1, alpha=0.6)
        ax.set_ylabel(VARIABLE_LABELS[col])
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Historical data and naive VAR/VECM forecast to 2100", y=0.995)
    add_source_note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_report(
    output: Path,
    data: pd.DataFrame,
    station_data: pd.DataFrame,
    adf: pd.DataFrame,
    johansen: pd.DataFrame,
    granger: pd.DataFrame,
    bundle: ModelBundle,
    forecast: pd.DataFrame,
    external_co2: pd.Series,
    use_daily_max: bool,
    temp_aggregation: str,
    all_stations: bool,
    temperature_source: str,
    hyras_aggregation: str,
    co2_scenario: str,
    co2_lookback: int,
) -> None:
    first_year, last_year = int(data.index.min()), int(data.index.max())
    forecast_end = int(forecast.index.max())
    temp_change = forecast.loc[forecast_end, "temp"] - data.loc[last_year, "temp"]
    sun_change = forecast.loc[forecast_end, "sunshine"] - data.loc[last_year, "sunshine"]
    precip_change = forecast.loc[forecast_end, "precip"] - data.loc[last_year, "precip"]
    co2_change = forecast.loc[forecast_end, "co2"] - data.loc[last_year, "co2"]
    station_records = (
        station_data[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_id")
        .to_dict("records")
    )
    stations = station_records if len(station_records) <= 12 else f"{len(station_records)} stations; see dwd_station_data_long.csv"
    source_note = (
        "annual maximum of daily maximum temperature TXK and annual sunshine sum from daily KL"
        if use_daily_max
        else "annual KL station values; temperature uses JA_MX_TX, sunshine JA_SD_S, precipitation JA_RR"
    )
    station_scope = "all available DWD annual KL stations" if all_stations else "selected representative stations"
    temp_source_note = (
        f"HYRAS-DE gridded daily tasmax ({hyras_aggregation})"
        if temperature_source == "hyras"
        else f"DWD station annual {temp_aggregation} across loaded station-level yearly maxima"
    )

    co2_driver_name = "d_co2" if "d_co2" in granger["causing"].astype(str).values else "co2"
    significant = granger[
        (granger["causing"].eq(co2_driver_name)) & (granger["reject_5pct"].eq(True))
    ]
    co2_evidence = (
        f"The stationary VAR Granger tests reject at 5% for at least one {co2_driver_name} -> climate-variable direction."
        if not significant.empty
        else f"The stationary VAR Granger tests do not reject at 5% for {co2_driver_name} -> the selected German climate variables."
    )
    johansen_note = (
        f"- Johansen cointegration rank at 5% trace-test rule: {bundle.coint_rank}"
        if bundle.model_type == "VECM"
        else "- Johansen/VECM skipped because the estimated integration orders are mixed."
    )

    text = f"""# Climate VAR/VECM Report

## Data

- Period used: {first_year}-{last_year}
- DWD stations: {stations}
- DWD climate definition: {source_note}
- DWD station scope: {station_scope}
- Temperature source: {temp_source_note}
- CO2 source: NOAA GML Mauna Loa annual mean. Years before the Mauna Loa start are linearly backcast only for alignment if needed.

## Model

- Selected framework: {bundle.model_type}
- Lag setting: {bundle.lag_order}
- Estimated integration orders from ADF rule: {bundle.integration_order}
{johansen_note}

## Statistical summary

{co2_evidence}

ADF and Johansen/model-selection details are written to CSV files in the output directory. Level correlations should not be interpreted causally. If the integration orders are mixed, the reported Granger tests use stationary climate levels and CO2 changes (`d_co2`) rather than a VECM on levels.

## Naive forecast to {forecast_end}

The model baseline forecast is adjusted with an external CO2 path: a {co2_scenario} continuation of the last {co2_lookback} historical CO2 years. Under that assumption:

- CO2 changes by about {co2_change:.1f} ppm from {last_year} to {forecast_end}.
- Annual maximum temperature changes by about {temp_change:.2f} deg C from {last_year} to {forecast_end}.
- Sunshine duration changes by about {sun_change:.1f} h/year from {last_year} to {forecast_end}.
- Annual precipitation changes by about {precip_change:.1f} mm/year from {last_year} to {forecast_end}.

## Limitations

- This is a statistical extrapolation, not a process-based climate model.
- Mauna Loa CO2 is a global proxy, not a Germany-specific emissions or concentration series.
- DWD station averages are not an official Germany-area mean.
- VAR/VECM long-run forecasts are sensitive to lag length, trend specification, structural breaks, station choice, and non-stationarity.
- Policy scenarios, aerosols, land-use change, circulation shifts, volcanic forcing, solar variability, and internal climate variability are not explicitly modeled.
- The CO2 scenario is naive. For decision support, replace it with SSP/RCP concentration pathways and use physical climate-model ensembles.
"""
    output.write_text(text, encoding="utf-8")


def save_tables(outdir: Path, **tables: pd.DataFrame) -> None:
    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv", index=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--forecast-end", type=int, default=2100)
    parser.add_argument("--stations", nargs="+", default=list(DEFAULT_STATIONS.keys()))
    parser.add_argument("--all-stations", action="store_true", help="Load all DWD station ZIP files instead of --stations.")
    parser.add_argument(
        "--temp-aggregation",
        choices=["max", "mean"],
        default="max",
        help="Aggregate yearly station maximum temperatures by max or mean.",
    )
    parser.add_argument(
        "--temperature-source",
        choices=["stations", "hyras"],
        default="stations",
        help="Use DWD station maxima or HYRAS-DE gridded daily tasmax for temperature.",
    )
    parser.add_argument(
        "--hyras-aggregation",
        choices=["annual_spatial_max", "annual_mean_daily_spatial_mean"],
        default="annual_spatial_max",
        help="Aggregation for HYRAS tasmax NetCDF files.",
    )
    parser.add_argument("--hyras-version", default="v6-1")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--co2-scenario",
        choices=["linear", "exponential"],
        default="linear",
        help="External CO2 path used for the long-run forecast.",
    )
    parser.add_argument("--co2-lookback", type=int, default=10, help="Years used to fit the external CO2 path.")
    parser.add_argument("--use-daily-max", action="store_true", help="Use daily KL TXK aggregation instead of annual KL.")
    parser.add_argument("--maxlags", type=int, default=6)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(args.verbose)
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        fallback_style = "seaborn-v0_8-whitegrid"
        plt.style.use(fallback_style if fallback_style in plt.style.available else "default")

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing outputs to %s", args.outdir.resolve())

    climate, station_data = fetch_dwd_climate(
        station_ids=None if args.all_stations else [station.zfill(5) for station in args.stations],
        use_daily_max=args.use_daily_max,
        min_year=args.start_year,
        temp_aggregation=args.temp_aggregation,
    )
    if args.temperature_source == "hyras":
        hyras_end_year = args.end_year
        if hyras_end_year is None:
            climate_support = climate[["sunshine", "precip"]].dropna(how="any")
            hyras_end_year = int(climate_support.index.max())
        hyras_temp = fetch_hyras_tasmax(
            start_year=args.start_year,
            end_year=hyras_end_year,
            cache_dir=args.cache_dir / "hyras_tasmax",
            version=args.hyras_version,
            aggregation=args.hyras_aggregation,
        )
        climate = climate.copy()
        climate["temp"] = hyras_temp
    co2 = fetch_noaa_co2(min_year=args.start_year)
    data = align_and_clean(climate, co2, args.start_year, args.end_year)

    if len(data) < 30:
        raise RuntimeError(f"Too few aligned observations after cleaning: {len(data)}")

    data.to_csv(args.outdir / "climate_co2_aligned.csv")
    station_data.to_csv(args.outdir / "dwd_station_data_long.csv")
    plot_raw_trends(data, args.outdir / "01_raw_trends.png")

    adf, integration_order = adf_table(data)
    logging.info("Estimated integration orders: %s", integration_order)

    all_i1 = all(order == 1 for order in integration_order.values())
    if all_i1:
        joh_lag = max(1, min(args.maxlags, choose_lag_order(data, maxlags=args.maxlags)) - 1)
        johansen, rank = johansen_table(data, det_order=0, k_ar_diff=joh_lag)
    else:
        johansen = pd.DataFrame(
            [
                {
                    "note": (
                        "Johansen/VECM skipped because variables are not all I(1); "
                        f"estimated integration orders: {integration_order}"
                    )
                }
            ]
        )
        rank = 0
    bundle = fit_model(data, coint_rank=rank, integration_order=integration_order, maxlags=args.maxlags)

    granger = granger_tests(data, integration_order=integration_order, maxlags=args.maxlags)
    save_tables(
        args.outdir,
        adf_results=adf,
        johansen_results=johansen,
        granger_results=granger,
    )

    save_irf_plot(bundle, args.outdir / "02_irf.png", periods=20)
    save_fevd_plot(
        data,
        args.outdir / "03_fevd.png",
        periods=20,
        maxlags=args.maxlags,
        stationary_data=bundle.data_used if bundle.model_type == "MIXED_VAR_LEVELS_DCO2" else None,
    )

    forecast, lower, upper, external_co2 = forecast_bundle(
        bundle,
        levels_data=data,
        horizon_end=args.forecast_end,
        alpha=0.10,
        co2_scenario=args.co2_scenario,
        co2_lookback=args.co2_lookback,
    )
    forecast.to_csv(args.outdir / "forecast_to_2100.csv")
    lower.to_csv(args.outdir / "forecast_lower_90.csv")
    upper.to_csv(args.outdir / "forecast_upper_90.csv")
    external_co2.to_csv(args.outdir / "external_co2_path.csv")
    plot_forecast(data, forecast, lower, upper, args.outdir / "04_forecast_to_2100.png")

    write_report(
        args.outdir / "report.md",
        data=data,
        station_data=station_data,
        adf=adf,
        johansen=johansen,
        granger=granger,
        bundle=bundle,
        forecast=forecast,
        external_co2=external_co2,
        use_daily_max=args.use_daily_max,
        temp_aggregation=args.temp_aggregation,
        all_stations=args.all_stations,
        temperature_source=args.temperature_source,
        hyras_aggregation=args.hyras_aggregation,
        co2_scenario=args.co2_scenario,
        co2_lookback=args.co2_lookback,
    )

    logging.info("Done. Key output: %s", (args.outdir / "report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
