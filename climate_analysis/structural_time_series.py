"""Structural time-series models for climate scenario forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import SparseEfficiencyWarning
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .config import SOURCE_NOTE, VARIABLE_LABELS
from .data import fetch_ssp_co2_path
from .models import scenario_co2_path


MAIN_SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5 worst-case",
    "exponential": "Exp. CO2 trend",
}
SCENARIO_COLORS = {
    "ssp119": "#2f9e44",
    "ssp126": "#74b816",
    "ssp245": "#f08c00",
    "ssp370": "#e8590c",
    "ssp585": "#c92a2a",
    "exponential": "#555555",
}


@dataclass
class STSResult:
    result: object
    co2_base: float
    end_year: int
    model_name: str


def co2_forcing(co2: pd.Series, base: float) -> pd.Series:
    forcing = np.log(co2.astype(float) / base)
    forcing.name = "log_co2_ratio"
    return forcing


def fit_temperature_sts(data: pd.DataFrame) -> STSResult:
    """Fit a Harvey-style structural model for HYRAS annual maximum temperature."""
    co2_base = float(data["co2"].iloc[0])
    annual_index = pd.PeriodIndex(data.index.astype(int), freq="Y")
    temp = data["temp"].astype(float).copy()
    co2 = data["co2"].astype(float).copy()
    temp.index = annual_index
    co2.index = annual_index
    exog = co2_forcing(co2, co2_base)
    model = UnobservedComponents(
        temp,
        level="local level",
        autoregressive=1,
        exog=exog,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        result = model.fit(disp=False, maxiter=1000)
    return STSResult(
        result=result,
        co2_base=co2_base,
        end_year=int(data.index.max()),
        model_name="local level + AR(1) weather + log(CO2 / CO2_1951)",
    )


def anchored_ssp_path(
    scenario: str,
    data: pd.DataFrame,
    cache_dir: Path,
    forecast_end: int,
) -> pd.Series:
    raw = fetch_ssp_co2_path(scenario, cache_dir, int(data.index.max()), forecast_end)
    ratio = float(data["co2"].iloc[-1]) / float(raw.loc[int(data.index.max())])
    return raw * ratio


def forecast_temperature_sts(
    sts: STSResult,
    co2_path: pd.Series,
    alpha: float = 0.10,
) -> pd.DataFrame:
    future_years = pd.Index(range(sts.end_year + 1, int(co2_path.index.max()) + 1), name="year")
    future_co2 = co2_path.reindex(future_years).astype(float)
    exog_future = co2_forcing(future_co2, sts.co2_base)
    prediction = sts.result.get_forecast(steps=len(future_years), exog=exog_future)
    frame = prediction.summary_frame(alpha=alpha)
    forecast = pd.DataFrame(
        {
            "temp": frame["mean"].values,
            "temp_lower": frame["mean_ci_lower"].values,
            "temp_upper": frame["mean_ci_upper"].values,
            "co2": future_co2.values,
        },
        index=future_years,
    )
    forecast.index.name = "year"
    return forecast


def run_sts_scenarios(
    data: pd.DataFrame,
    cache_dir: Path,
    forecast_end: int = 2100,
    alpha: float = 0.10,
    scenarios: list[str] | None = None,
) -> tuple[STSResult, dict[str, pd.DataFrame], pd.DataFrame, pd.Series]:
    scenarios = MAIN_SCENARIOS if scenarios is None else scenarios
    sts = fit_temperature_sts(data)
    forecasts = {}
    rows = []
    for scenario in scenarios:
        co2_path = anchored_ssp_path(scenario, data, cache_dir, forecast_end)
        forecast = forecast_temperature_sts(sts, co2_path, alpha=alpha)
        forecasts[scenario] = forecast
        rows.append(
            {
                "scenario": scenario,
                "co2_2100": float(forecast.loc[forecast_end, "co2"]),
                "temp_2100": float(forecast.loc[forecast_end, "temp"]),
                "temp_lower_2100": float(forecast.loc[forecast_end, "temp_lower"]),
                "temp_upper_2100": float(forecast.loc[forecast_end, "temp_upper"]),
            }
        )

    future_years = np.arange(int(data.index.max()) + 1, forecast_end + 1)
    exp_path = scenario_co2_path(data["co2"], future_years, lookback=10, scenario="exponential")
    summary = pd.DataFrame(rows).set_index("scenario")
    return sts, forecasts, summary, exp_path


def plot_sts_scenarios(
    data: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    exp_path: pd.Series,
    output: Path,
    confidence_level: float = 0.90,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    temp_ax, co2_ax = axes
    transition = int(data.index.max())

    temp_ax.plot(data.index, data["temp"], color="#9b2c2c", lw=1.4, label="Historical HYRAS")
    temp_ax.plot(
        data.index,
        data["temp"].rolling(11, center=True, min_periods=6).mean(),
        color="black",
        lw=2,
        label="11-year moving average",
    )
    for scenario, forecast in forecasts.items():
        color = SCENARIO_COLORS.get(scenario)
        temp_ax.plot(
            forecast.index,
            forecast["temp"],
            color=color,
            lw=2 if scenario == "ssp585" else 1.6,
            ls="-" if scenario == "ssp585" else "--",
            label=f"{SCENARIO_LABELS.get(scenario, scenario)} STS",
        )
        if scenario == "ssp585":
            temp_ax.fill_between(
                forecast.index,
                forecast["temp_lower"].values,
                forecast["temp_upper"].values,
                color=color,
                alpha=0.14,
                label=f"{confidence_level:.0%} interval (SSP5-8.5)",
            )
    temp_ax.axvline(transition, color="black", lw=1, alpha=0.6)
    temp_ax.set_ylabel(VARIABLE_LABELS["temp"])
    temp_ax.grid(alpha=0.25)
    temp_ax.legend(loc="best", ncols=2)

    co2_ax.plot(data.index, data["co2"], color="#245c8a", lw=1.4, label="Historical NOAA CO2")
    for scenario, forecast in forecasts.items():
        co2_ax.plot(
            forecast.index,
            forecast["co2"],
            color=SCENARIO_COLORS.get(scenario),
            lw=2 if scenario == "ssp585" else 1.6,
            ls="-" if scenario == "ssp585" else "--",
            label=SCENARIO_LABELS.get(scenario, scenario),
        )
    co2_ax.plot(
        exp_path.index,
        exp_path.values,
        color=SCENARIO_COLORS["exponential"],
        lw=1.5,
        ls=":",
        label="Documentary exp. CO2 trend",
    )
    co2_ax.text(
        0.01,
        0.92,
        "CO2 scenarios: CMIP6/ScenarioMIP pathways",
        transform=co2_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#245c8a",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#245c8a", "alpha": 0.85},
    )
    co2_ax.axvline(transition, color="black", lw=1, alpha=0.6)
    co2_ax.set_ylabel(VARIABLE_LABELS["co2"])
    co2_ax.set_xlabel("Year")
    co2_ax.grid(alpha=0.25)
    co2_ax.legend(loc="best", ncols=2)

    fig.suptitle("Structural time-series temperature forecasts under CMIP6 CO2 scenarios", y=0.995)
    fig.text(0.5, 0.965, SOURCE_NOTE, ha="center", va="top", fontsize=9, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_sts_report(
    output: Path,
    sts: STSResult,
    summary: pd.DataFrame,
    confidence_level: float,
) -> None:
    params = sts.result.params
    beta = params.get("beta.log_co2_ratio", params.get("beta.co2", np.nan))
    table = summary.round(2).reset_index()
    table_md = "\n".join(
        [
            "| scenario | co2_2100 | temp_2100 | temp_lower_2100 | temp_upper_2100 |",
            "|---|---:|---:|---:|---:|",
            *[
                (
                    f"| {row.scenario} | {row.co2_2100:.2f} | {row.temp_2100:.2f} | "
                    f"{row.temp_lower_2100:.2f} | {row.temp_upper_2100:.2f} |"
                )
                for row in table.itertuples(index=False)
            ],
        ]
    )
    text = f"""# Structural Time-Series Climate Forecast Report

## Model

- Framework: Harvey-style structural time-series / state-space model.
- Specification: {sts.model_name}.
- CO2 forcing: `log(CO2 / CO2_1951)`.
- Temperature variable: HYRAS annual spatial maximum of daily maximum temperature.
- Estimated CO2 forcing coefficient: {beta:.3f}.
- AIC: {sts.result.aic:.2f}.

## Scenario Results

Forecasts use CMIP6/ScenarioMIP CO2 concentration pathways, ratio-anchored to the final historical NOAA CO2 value. SSP5-8.5 is treated as a high-CO2 worst-case stress test, not a central forecast.

{table_md}

## Interpretation

This model is less autoregressive than the VARX scenario comparison. It separates a structural temperature component, AR(1) weather noise, and logarithmic CO2 forcing. It is still a naive statistical model, but the scenario response is easier to interpret than a VARX driven by annual CO2 increments.

## Limitations

- This is not a physical climate model.
- The CO2 coefficient and latent trend are difficult to fully separate in annual historical data.
- Forecast intervals reflect state-space model uncertainty, not full climate-system uncertainty.
- Hitzewellen are represented indirectly through annual maximum temperature, not duration or frequency.
"""
    output.write_text(text, encoding="utf-8")
