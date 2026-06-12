"""Run Harvey-style structural time-series CO2 scenario forecasts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd

from climate_analysis.structural_time_series import (
    plot_climate_modulated_ebm,
    plot_sts_scenarios,
    plot_energy_balance_comparison,
    run_climate_modulated_ebm_scenarios,
    run_energy_balance_scenarios,
    run_sts_scenarios,
    write_climate_modulated_ebm_report,
    write_energy_balance_report,
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
    parser.add_argument(
        "--ebm-max-adjustment-rate",
        type=float,
        default=0.12,
        help="Annual adjustment-rate cap for the energy-balance sensitivity model.",
    )
    parser.add_argument(
        "--modulated-plot-scenario",
        default="ssp370",
        help="CO2 scenario to show in the climate-modulated EBM regional-state plot.",
    )
    parser.add_argument("--modulated-co2-prior-weight", type=float, default=20.0)
    parser.add_argument("--modulated-regional-prior-weight", type=float, default=1.0)
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

    ebm, ebm_forecasts, ebm_summary = run_energy_balance_scenarios(
        data=data,
        cache_dir=args.cache_dir / "ssp_co2",
        forecast_end=args.forecast_end,
        confidence_level=args.confidence_level,
        max_adjustment_rate=args.ebm_max_adjustment_rate,
    )
    for scenario, forecast in ebm_forecasts.items():
        forecast.to_csv(args.outdir / f"ebm_forecast_{scenario}.csv")
    ebm_summary.to_csv(args.outdir / "ebm_forecast_summary.csv")
    plot_energy_balance_comparison(
        data=data,
        sts_forecasts=forecasts,
        ebm_forecasts=ebm_forecasts,
        output=args.outdir / "02_energy_balance_sensitivity.png",
        confidence_level=args.confidence_level,
    )
    write_energy_balance_report(
        output=args.outdir / "energy_balance_report.md",
        ebm=ebm,
        summary=ebm_summary,
        confidence_level=args.confidence_level,
    )

    modulated, modulated_forecasts, modulated_summary = run_climate_modulated_ebm_scenarios(
        data=data,
        cache_dir=args.cache_dir / "ssp_co2",
        forecast_end=args.forecast_end,
        confidence_level=args.confidence_level,
        max_adjustment_rate=args.ebm_max_adjustment_rate,
        co2_prior_weight=args.modulated_co2_prior_weight,
        regional_prior_weight=args.modulated_regional_prior_weight,
    )
    for (scenario, state), forecast in modulated_forecasts.items():
        forecast.to_csv(args.outdir / f"modulated_ebm_forecast_{scenario}_{state}.csv")
    modulated_summary.to_csv(args.outdir / "modulated_ebm_forecast_summary.csv")
    plot_climate_modulated_ebm(
        data=data,
        forecasts=modulated_forecasts,
        output=args.outdir / "03_climate_modulated_ebm.png",
        scenario=args.modulated_plot_scenario,
        confidence_level=args.confidence_level,
    )
    write_climate_modulated_ebm_report(
        output=args.outdir / "climate_modulated_ebm_report.md",
        model=modulated,
        summary=modulated_summary,
        confidence_level=args.confidence_level,
    )

    print(f"Model: {sts.model_name}")
    print(f"AIC: {sts.result.aic:.2f}")
    print(summary.round(2).to_string())
    print(f"\nEnergy-balance sensitivity: adjustment_rate={ebm.adjustment_rate:.3f}, sensitivity={ebm.sensitivity:.3f}")
    print(ebm_summary.round(2).to_string())
    print(
        "\nClimate-modulated EBM: "
        f"adjustment_rate={modulated.adjustment_rate:.3f}, "
        f"co2_sensitivity={modulated.co2_sensitivity:.3f}, "
        f"sunshine_effect={modulated.sunshine_effect:.3f}, "
        f"precip_effect={modulated.precip_effect:.3f}"
    )
    print(modulated_summary.round(2).to_string())
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
