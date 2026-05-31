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
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # Optional styling dependency.
    sns = None

from climate_analysis.config import DEFAULT_STATIONS, RANDOM_SEED, SSP_SCENARIOS
from climate_analysis.data import (
    align_and_clean,
    fetch_dwd_climate,
    fetch_hyras_tasmax,
    fetch_noaa_co2,
    fetch_ssp_co2_path,
)
from climate_analysis.models import (
    adf_table,
    choose_lag_order,
    fit_model,
    forecast_bundle,
    granger_tests,
    johansen_table,
    scenario_co2_path,
)
from climate_analysis.plots import plot_forecast, plot_raw_trends, save_fevd_plot, save_irf_plot
from climate_analysis.reporting import save_tables, write_report

np.random.seed(RANDOM_SEED)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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
        choices=["linear", "exponential", *SSP_SCENARIOS.keys()],
        default="linear",
        help="External CO2 path used for the long-run forecast.",
    )
    parser.add_argument("--co2-lookback", type=int, default=10, help="Years used to fit the external CO2 path.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.90,
        help="Forecast interval level. Use 0.66 or 66 for a 66%% interval.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["auto", "exogenous-co2"],
        default="auto",
        help="auto uses VAR/VECM selection; exogenous-co2 treats d_CO2 as an imposed external driver.",
    )
    parser.add_argument(
        "--co2-feature",
        choices=["dlog", "dppm", "forcing"],
        default="dlog",
        help=(
            "CO2 driver used in --model-mode exogenous-co2: dlog uses annual log differences, "
            "dppm uses annual ppm differences, forcing uses log(CO2 / initial CO2)."
        ),
    )
    parser.add_argument(
        "--interval-method",
        choices=["bootstrap", "analytic"],
        default="bootstrap",
        help="Forecast interval method for exogenous VARX. Bootstrap resamples fitted residual vectors.",
    )
    parser.add_argument("--bootstrap-sims", type=int, default=2000, help="Residual bootstrap paths for VARX intervals.")
    parser.add_argument(
        "--forecast-start",
        choices=["trend", "last"],
        default="trend",
        help="Initial climate state for VARX forecasts. Trend starts from a trailing smoothed level; last uses the final observed year.",
    )
    parser.add_argument("--trend-window", type=int, default=11, help="Trailing window for --forecast-start trend.")
    parser.add_argument("--use-daily-max", action="store_true", help="Use daily KL TXK aggregation instead of annual KL.")
    parser.add_argument("--maxlags", type=int, default=6)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(args.verbose)
    if args.confidence_level > 1:
        args.confidence_level = args.confidence_level / 100.0
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must be between 0 and 1, or between 0 and 100 as a percentage.")
    interval_tag = f"{int(round(args.confidence_level * 100)):02d}"
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
    if args.model_mode == "exogenous-co2":
        johansen = pd.DataFrame(
            [{"note": "Johansen/VECM skipped because --model-mode exogenous-co2 imposes CO2 as an external driver."}]
        )
        rank = 0
    elif all_i1:
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
    bundle = fit_model(
        data,
        coint_rank=rank,
        integration_order=integration_order,
        maxlags=args.maxlags,
        model_mode=args.model_mode,
        co2_feature=args.co2_feature,
    )

    granger = granger_tests(
        data,
        integration_order=integration_order,
        maxlags=args.maxlags,
        co2_feature=args.co2_feature if args.model_mode == "exogenous-co2" else "dppm",
    )
    save_tables(
        args.outdir,
        adf_results=adf,
        johansen_results=johansen,
        granger_results=granger,
    )

    save_irf_plot(bundle, args.outdir / "02_irf.png", periods=20)
    if bundle.model_type != "EXOGENOUS_CO2_VARX":
        save_fevd_plot(
            data,
            args.outdir / "03_fevd.png",
            periods=20,
            maxlags=args.maxlags,
            stationary_data=bundle.data_used if bundle.model_type == "MIXED_VAR_LEVELS_DCO2" else None,
        )
    else:
        logging.info("Skipping FEVD plot for EXOGENOUS_CO2_VARX.")

    future_start = int(data.index.max()) + 1
    external_co2_path = None
    if args.co2_scenario not in {"linear", "exponential"}:
        external_co2_path = fetch_ssp_co2_path(
            scenario=args.co2_scenario,
            cache_dir=args.cache_dir / "ssp_co2",
            start_year=int(data.index.max()),
            end_year=args.forecast_end,
        )

    forecast, lower, upper, external_co2 = forecast_bundle(
        bundle,
        levels_data=data,
        horizon_end=args.forecast_end,
        alpha=1.0 - args.confidence_level,
        co2_scenario=args.co2_scenario,
        co2_lookback=args.co2_lookback,
        external_co2_path=external_co2_path,
        interval_method=args.interval_method,
        bootstrap_sims=args.bootstrap_sims,
        random_seed=RANDOM_SEED,
        forecast_start=args.forecast_start,
        trend_window=args.trend_window,
    )
    forecast.to_csv(args.outdir / "forecast_to_2100.csv")
    lower.to_csv(args.outdir / f"forecast_lower_{interval_tag}.csv")
    upper.to_csv(args.outdir / f"forecast_upper_{interval_tag}.csv")
    external_co2.to_csv(args.outdir / "external_co2_path.csv")
    reference_paths = None
    if args.co2_scenario == "ssp585":
        reference_paths = {
            "co2": scenario_co2_path(
                data["co2"],
                forecast.index.to_numpy(),
                lookback=args.co2_lookback,
                scenario="exponential",
            )
        }

    plot_forecast(
        data,
        forecast,
        lower,
        upper,
        args.outdir / "04_forecast_to_2100.png",
        confidence_level=args.confidence_level,
        model_label="structural VARX" if bundle.model_type == "EXOGENOUS_CO2_VARX" else bundle.model_type,
        scenario_label="SSP5-8.5 high-CO2 worst-case" if args.co2_scenario == "ssp585" else args.co2_scenario,
        co2_scenario_label="CO2 path: SSP5-8.5 high-CO2 worst-case" if args.co2_scenario == "ssp585" else None,
        forecast_label=(
            "Structural VARX forecast under SSP5-8.5"
            if args.co2_scenario == "ssp585"
            else "Naive forecast"
        ),
        co2_forecast_label=(
            "SSP5-8.5 CO2 path (model input)"
            if args.co2_scenario == "ssp585"
            else None
        ),
        reference_paths=reference_paths,
    )

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
        confidence_level=args.confidence_level,
        co2_feature=args.co2_feature,
        interval_method=args.interval_method,
        bootstrap_sims=args.bootstrap_sims,
        forecast_start=args.forecast_start,
        trend_window=args.trend_window,
    )

    logging.info("Done. Key output: %s", (args.outdir / "report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
