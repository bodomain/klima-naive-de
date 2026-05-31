"""Run Harvey-style structural time-series CO2 scenario forecasts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd

from climate_analysis.structural_time_series import (
    plot_sts_scenarios,
    run_sts_scenarios,
    write_sts_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a structural time-series model for HYRAS maximum temperature and forecast SSP CO2 scenarios."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs_full_hyras_exogco2_ssp585/climate_co2_aligned.csv"),
        help="Aligned historical climate/CO2 CSV produced by climate_var_vecm_forecast.py.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data_cache"),
        help="Cache directory containing or receiving SSP supplement data.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs_structural_time_series"),
        help="Output directory for plots, CSV files, and report.",
    )
    parser.add_argument("--forecast-end", type=int, default=2100)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = 1.0 - args.confidence_level
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, index_col=0)
    data.index = data.index.astype(int)

    sts, forecasts, summary, exp_path = run_sts_scenarios(
        data=data,
        cache_dir=args.cache_dir / "ssp_co2",
        forecast_end=args.forecast_end,
        alpha=alpha,
    )

    for scenario, forecast in forecasts.items():
        forecast.to_csv(args.outdir / f"sts_forecast_{scenario}.csv")
    summary.to_csv(args.outdir / "sts_forecast_summary.csv")
    exp_path.to_csv(args.outdir / "documentary_exponential_co2_path.csv", header=["co2"])

    plot_sts_scenarios(
        data=data,
        forecasts=forecasts,
        exp_path=exp_path,
        output=args.outdir / "01_sts_scenario_forecasts.png",
        confidence_level=args.confidence_level,
    )
    write_sts_report(
        output=args.outdir / "report.md",
        sts=sts,
        summary=summary,
        confidence_level=args.confidence_level,
    )

    print(f"Model: {sts.model_name}")
    print(f"AIC: {sts.result.aic:.2f}")
    print(summary.round(2).to_string())
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
