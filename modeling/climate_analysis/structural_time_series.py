"""Structural time-series models for climate scenario forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
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
    "ssp585": "SSP5-8.5 extreme",
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


@dataclass
class EnergyBalanceResult:
    base_temp: float
    sensitivity: float
    adjustment_rate: float
    max_adjustment_rate: float
    co2_base: float
    end_year: int
    residual_std: float
    model_name: str


@dataclass
class ClimateModulatedEBMResult:
    base_temp: float
    co2_sensitivity: float
    sunshine_effect: float
    precip_effect: float
    adjustment_rate: float
    max_adjustment_rate: float
    co2_base: float
    sunshine_mean: float
    sunshine_std: float
    precip_mean: float
    precip_std: float
    co2_prior_sensitivity: float
    co2_prior_weight: float
    regional_prior_weight: float
    end_year: int
    residual_std: float
    model_name: str


REGIONAL_STATES = {
    "normal": {"label": "normal sunshine/precip", "sunshine_z": 0.0, "precip_z": 0.0, "color": "#333333"},
    "hot_dry": {"label": "sunny-dry (+S, -P)", "sunshine_z": 1.0, "precip_z": -1.0, "color": "#c92a2a"},
    "cloudy_wet": {"label": "cloudy-wet (-S, +P)", "sunshine_z": -1.0, "precip_z": 1.0, "color": "#245c8a"},
}


def co2_forcing(co2: pd.Series, base: float) -> pd.Series:
    forcing = np.log(co2.astype(float) / base)
    forcing.name = "log_co2_ratio"
    return forcing


def radiative_forcing(co2: pd.Series | float, base: float) -> pd.Series | float:
    """CO2 radiative forcing approximation in W/m2."""
    return 5.35 * np.log(np.asarray(co2, dtype=float) / base)


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


def fit_energy_balance_model(data: pd.DataFrame, max_adjustment_rate: float = 0.12) -> EnergyBalanceResult:
    """Fit a one-box energy-balance sensitivity model for annual extreme temperature."""
    temp = data["temp"].astype(float)
    co2_base = float(data["co2"].iloc[0])
    forcing = pd.Series(radiative_forcing(data["co2"], co2_base), index=data.index)
    y = temp.iloc[1:].values
    previous = temp.iloc[:-1].values
    force = forcing.iloc[1:].values

    def unpack(params: np.ndarray) -> tuple[float, float, float]:
        base_temp = float(params[0])
        sensitivity = float(np.exp(params[1]))
        adjustment_rate = float(max_adjustment_rate / (1.0 + np.exp(-params[2])))
        return base_temp, sensitivity, adjustment_rate

    def residuals(params: np.ndarray) -> np.ndarray:
        base_temp, sensitivity, adjustment_rate = unpack(params)
        equilibrium = base_temp + sensitivity * force
        prediction = previous + adjustment_rate * (equilibrium - previous)
        return prediction - y

    initial = np.array([float(temp.iloc[0]), np.log(1.0), 0.0])
    result = least_squares(residuals, initial, max_nfev=20000)
    base_temp, sensitivity, adjustment_rate = unpack(result.x)
    resid = residuals(result.x)
    return EnergyBalanceResult(
        base_temp=base_temp,
        sensitivity=sensitivity,
        adjustment_rate=adjustment_rate,
        max_adjustment_rate=max_adjustment_rate,
        co2_base=co2_base,
        end_year=int(data.index.max()),
        residual_std=float(np.std(resid, ddof=3)),
        model_name=(
            "T_t = T_{t-1} + kappa * "
            "(base + lambda * 5.35 log(CO2/CO2_1951) - T_{t-1})"
        ),
    )


def fit_climate_modulated_ebm(
    data: pd.DataFrame,
    max_adjustment_rate: float = 0.12,
    co2_prior_sensitivity: float | None = None,
    co2_prior_weight: float = 20.0,
    regional_prior_weight: float = 1.0,
) -> ClimateModulatedEBMResult:
    """Fit an EBM-style dynamic regression with sunshine and precipitation modulation."""
    if co2_prior_sensitivity is None:
        co2_prior_sensitivity = fit_energy_balance_model(
            data,
            max_adjustment_rate=max_adjustment_rate,
        ).sensitivity
    temp = data["temp"].astype(float)
    sunshine = data["sunshine"].astype(float)
    precip = data["precip"].astype(float)
    co2_base = float(data["co2"].iloc[0])
    forcing = pd.Series(radiative_forcing(data["co2"], co2_base), index=data.index)

    sunshine_mean = float(sunshine.mean())
    sunshine_std = float(sunshine.std(ddof=1))
    precip_mean = float(precip.mean())
    precip_std = float(precip.std(ddof=1))
    sunshine_z = (sunshine - sunshine_mean) / sunshine_std
    precip_z = (precip - precip_mean) / precip_std

    y = temp.iloc[1:].values
    previous = temp.iloc[:-1].values
    force = forcing.iloc[1:].values
    sun = sunshine_z.iloc[1:].values
    rain = precip_z.iloc[1:].values

    def unpack(params: np.ndarray) -> tuple[float, float, float, float, float]:
        base_temp = float(params[0])
        co2_sensitivity = float(np.exp(params[1]))
        sunshine_effect = float(params[2])
        precip_effect = float(params[3])
        adjustment_rate = float(max_adjustment_rate / (1.0 + np.exp(-params[4])))
        return base_temp, co2_sensitivity, sunshine_effect, precip_effect, adjustment_rate

    def residuals(params: np.ndarray) -> np.ndarray:
        base_temp, co2_sensitivity, sunshine_effect, precip_effect, adjustment_rate = unpack(params)
        equilibrium = base_temp + co2_sensitivity * force + sunshine_effect * sun + precip_effect * rain
        prediction = previous + adjustment_rate * (equilibrium - previous)
        fit_resid = prediction - y
        penalties = np.array(
            [
                np.sqrt(co2_prior_weight) * (co2_sensitivity - float(co2_prior_sensitivity)),
                np.sqrt(regional_prior_weight) * sunshine_effect,
                np.sqrt(regional_prior_weight) * precip_effect,
            ]
        )
        return np.r_[fit_resid, penalties]

    initial = np.array([float(temp.iloc[0]), np.log(0.8), 0.5, -0.3, 0.0])
    result = least_squares(residuals, initial, max_nfev=20000)
    base_temp, co2_sensitivity, sunshine_effect, precip_effect, adjustment_rate = unpack(result.x)
    resid = residuals(result.x)[: len(y)]
    return ClimateModulatedEBMResult(
        base_temp=base_temp,
        co2_sensitivity=co2_sensitivity,
        sunshine_effect=sunshine_effect,
        precip_effect=precip_effect,
        adjustment_rate=adjustment_rate,
        max_adjustment_rate=max_adjustment_rate,
        co2_base=co2_base,
        sunshine_mean=sunshine_mean,
        sunshine_std=sunshine_std,
        precip_mean=precip_mean,
        precip_std=precip_std,
        co2_prior_sensitivity=float(co2_prior_sensitivity),
        co2_prior_weight=co2_prior_weight,
        regional_prior_weight=regional_prior_weight,
        end_year=int(data.index.max()),
        residual_std=float(np.std(resid, ddof=5)),
        model_name=(
            "T_t = T_{t-1} + kappa * "
            "(base + lambda_C F_CO2 + lambda_S z_sunshine + lambda_P z_precip - T_{t-1})"
        ),
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


def forecast_energy_balance(
    ebm: EnergyBalanceResult,
    data: pd.DataFrame,
    co2_path: pd.Series,
    confidence_level: float = 0.90,
    start_window: int = 11,
) -> pd.DataFrame:
    future_years = pd.Index(range(ebm.end_year + 1, int(co2_path.index.max()) + 1), name="year")
    z_value = 1.6448536269514722 if abs(confidence_level - 0.90) < 1e-9 else 1.959963984540054
    current_temp = float(data["temp"].rolling(start_window, min_periods=max(3, start_window // 2)).mean().iloc[-1])
    if np.isnan(current_temp):
        current_temp = float(data["temp"].iloc[-1])

    rows = []
    forecast_variance = 0.0
    for step, year in enumerate(future_years, start=1):
        co2 = float(co2_path.loc[year])
        forcing = float(radiative_forcing(co2, ebm.co2_base))
        equilibrium = ebm.base_temp + ebm.sensitivity * forcing
        current_temp = current_temp + ebm.adjustment_rate * (equilibrium - current_temp)
        forecast_variance = (1.0 - ebm.adjustment_rate) ** 2 * forecast_variance + ebm.residual_std**2
        interval = z_value * np.sqrt(forecast_variance)
        rows.append(
            {
                "temp": current_temp,
                "temp_lower": current_temp - interval,
                "temp_upper": current_temp + interval,
                "co2": co2,
                "forcing_wm2": forcing,
                "equilibrium_temp": equilibrium,
                "step": step,
            }
        )
    forecast = pd.DataFrame(rows, index=future_years)
    forecast.index.name = "year"
    return forecast


def forecast_climate_modulated_ebm(
    model: ClimateModulatedEBMResult,
    data: pd.DataFrame,
    co2_path: pd.Series,
    sunshine_z: float,
    precip_z: float,
    confidence_level: float = 0.90,
    start_window: int = 11,
) -> pd.DataFrame:
    future_years = pd.Index(range(model.end_year + 1, int(co2_path.index.max()) + 1), name="year")
    z_value = 1.6448536269514722 if abs(confidence_level - 0.90) < 1e-9 else 1.959963984540054
    current_temp = float(data["temp"].rolling(start_window, min_periods=max(3, start_window // 2)).mean().iloc[-1])
    if np.isnan(current_temp):
        current_temp = float(data["temp"].iloc[-1])

    rows = []
    forecast_variance = 0.0
    for step, year in enumerate(future_years, start=1):
        co2 = float(co2_path.loc[year])
        forcing = float(radiative_forcing(co2, model.co2_base))
        equilibrium = (
            model.base_temp
            + model.co2_sensitivity * forcing
            + model.sunshine_effect * sunshine_z
            + model.precip_effect * precip_z
        )
        current_temp = current_temp + model.adjustment_rate * (equilibrium - current_temp)
        forecast_variance = (1.0 - model.adjustment_rate) ** 2 * forecast_variance + model.residual_std**2
        interval = z_value * np.sqrt(forecast_variance)
        rows.append(
            {
                "temp": current_temp,
                "temp_lower": current_temp - interval,
                "temp_upper": current_temp + interval,
                "co2": co2,
                "forcing_wm2": forcing,
                "equilibrium_temp": equilibrium,
                "sunshine_z": sunshine_z,
                "precip_z": precip_z,
                "sunshine_hours": model.sunshine_mean + sunshine_z * model.sunshine_std,
                "precip_mm": model.precip_mean + precip_z * model.precip_std,
                "step": step,
            }
        )
    forecast = pd.DataFrame(rows, index=future_years)
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


def run_energy_balance_scenarios(
    data: pd.DataFrame,
    cache_dir: Path,
    forecast_end: int = 2100,
    confidence_level: float = 0.90,
    scenarios: list[str] | None = None,
    max_adjustment_rate: float = 0.12,
) -> tuple[EnergyBalanceResult, dict[str, pd.DataFrame], pd.DataFrame]:
    scenarios = MAIN_SCENARIOS if scenarios is None else scenarios
    ebm = fit_energy_balance_model(data, max_adjustment_rate=max_adjustment_rate)
    forecasts = {}
    rows = []
    for scenario in scenarios:
        co2_path = anchored_ssp_path(scenario, data, cache_dir, forecast_end)
        forecast = forecast_energy_balance(ebm, data, co2_path, confidence_level=confidence_level)
        forecasts[scenario] = forecast
        rows.append(
            {
                "scenario": scenario,
                "co2_2100": float(forecast.loc[forecast_end, "co2"]),
                "forcing_2100_wm2": float(forecast.loc[forecast_end, "forcing_wm2"]),
                "temp_2100": float(forecast.loc[forecast_end, "temp"]),
                "temp_lower_2100": float(forecast.loc[forecast_end, "temp_lower"]),
                "temp_upper_2100": float(forecast.loc[forecast_end, "temp_upper"]),
                "equilibrium_temp_2100": float(forecast.loc[forecast_end, "equilibrium_temp"]),
            }
        )
    summary = pd.DataFrame(rows).set_index("scenario")
    return ebm, forecasts, summary


def run_climate_modulated_ebm_scenarios(
    data: pd.DataFrame,
    cache_dir: Path,
    forecast_end: int = 2100,
    confidence_level: float = 0.90,
    scenarios: list[str] | None = None,
    regional_states: dict[str, dict[str, float | str]] | None = None,
    max_adjustment_rate: float = 0.12,
    co2_prior_weight: float = 20.0,
    regional_prior_weight: float = 1.0,
) -> tuple[ClimateModulatedEBMResult, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    scenarios = MAIN_SCENARIOS if scenarios is None else scenarios
    regional_states = REGIONAL_STATES if regional_states is None else regional_states
    model = fit_climate_modulated_ebm(
        data,
        max_adjustment_rate=max_adjustment_rate,
        co2_prior_weight=co2_prior_weight,
        regional_prior_weight=regional_prior_weight,
    )
    forecasts = {}
    rows = []
    for scenario in scenarios:
        co2_path = anchored_ssp_path(scenario, data, cache_dir, forecast_end)
        for state_name, state in regional_states.items():
            forecast = forecast_climate_modulated_ebm(
                model,
                data,
                co2_path,
                sunshine_z=float(state["sunshine_z"]),
                precip_z=float(state["precip_z"]),
                confidence_level=confidence_level,
            )
            forecasts[(scenario, state_name)] = forecast
            rows.append(
                {
                    "scenario": scenario,
                    "regional_state": state_name,
                    "co2_2100": float(forecast.loc[forecast_end, "co2"]),
                    "forcing_2100_wm2": float(forecast.loc[forecast_end, "forcing_wm2"]),
                    "sunshine_z": float(state["sunshine_z"]),
                    "precip_z": float(state["precip_z"]),
                    "sunshine_hours": float(forecast.loc[forecast_end, "sunshine_hours"]),
                    "precip_mm": float(forecast.loc[forecast_end, "precip_mm"]),
                    "temp_2100": float(forecast.loc[forecast_end, "temp"]),
                    "temp_lower_2100": float(forecast.loc[forecast_end, "temp_lower"]),
                    "temp_upper_2100": float(forecast.loc[forecast_end, "temp_upper"]),
                    "equilibrium_temp_2100": float(forecast.loc[forecast_end, "equilibrium_temp"]),
                }
            )
    summary = pd.DataFrame(rows).set_index(["scenario", "regional_state"])
    return model, forecasts, summary


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


def plot_energy_balance_comparison(
    data: pd.DataFrame,
    sts_forecasts: dict[str, pd.DataFrame],
    ebm_forecasts: dict[str, pd.DataFrame],
    output: Path,
    confidence_level: float = 0.90,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    transition = int(data.index.max())
    ax.plot(data.index, data["temp"], color="#9b2c2c", lw=1.2, alpha=0.8, label="Historical HYRAS")
    ax.plot(
        data.index,
        data["temp"].rolling(11, center=True, min_periods=6).mean(),
        color="black",
        lw=2,
        label="11-year moving average",
    )
    for scenario, forecast in sts_forecasts.items():
        color = SCENARIO_COLORS.get(scenario)
        ax.plot(
            forecast.index,
            forecast["temp"],
            color=color,
            lw=1.1,
            ls=":",
            alpha=0.8,
            label=f"{SCENARIO_LABELS.get(scenario, scenario)} STS",
        )
    for scenario, forecast in ebm_forecasts.items():
        color = SCENARIO_COLORS.get(scenario)
        ax.plot(
            forecast.index,
            forecast["temp"],
            color=color,
            lw=2.1 if scenario == "ssp585" else 1.8,
            ls="-" if scenario == "ssp585" else "--",
            label=f"{SCENARIO_LABELS.get(scenario, scenario)} EBM sensitivity",
        )
        if scenario == "ssp585":
            ax.fill_between(
                forecast.index,
                forecast["temp_lower"].values,
                forecast["temp_upper"].values,
                color=color,
                alpha=0.08,
                label=f"{confidence_level:.0%} EBM interval (SSP5-8.5)",
            )
    ax.axvline(transition, color="black", lw=1, alpha=0.6)
    fig.suptitle("Temperature scenarios with logarithmic CO2 forcing and bounded thermal adjustment", y=0.985)
    fig.text(0.5, 0.945, SOURCE_NOTE, ha="center", va="top", fontsize=9, color="#333333")
    ax.set_xlabel("Year")
    ax.set_ylabel(VARIABLE_LABELS["temp"])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_climate_modulated_ebm(
    data: pd.DataFrame,
    forecasts: dict[tuple[str, str], pd.DataFrame],
    output: Path,
    scenario: str = "ssp370",
    confidence_level: float = 0.90,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    transition = int(data.index.max())
    ax.plot(data.index, data["temp"], color="#9b2c2c", lw=1.2, alpha=0.75, label="Historical HYRAS")
    ax.plot(
        data.index,
        data["temp"].rolling(11, center=True, min_periods=6).mean(),
        color="black",
        lw=2,
        label="11-year moving average",
    )
    for state_name, state in REGIONAL_STATES.items():
        forecast = forecasts[(scenario, state_name)]
        color = str(state["color"])
        label = str(state["label"])
        ax.plot(forecast.index, forecast["temp"], color=color, lw=2.2, label=label)
        if state_name == "normal":
            ax.fill_between(
                forecast.index,
                forecast["temp_lower"].values,
                forecast["temp_upper"].values,
                color=color,
                alpha=0.08,
                label=f"{confidence_level:.0%} model interval (normal)",
            )
    ax.axvline(transition, color="black", lw=1, alpha=0.6)
    fig.suptitle(f"Climate-modulated EBM forecast under {SCENARIO_LABELS.get(scenario, scenario)}", y=0.985)
    fig.text(0.5, 0.945, SOURCE_NOTE, ha="center", va="top", fontsize=9, color="#333333")
    ax.set_xlabel("Year")
    ax.set_ylabel(VARIABLE_LABELS["temp"])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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


def write_energy_balance_report(
    output: Path,
    ebm: EnergyBalanceResult,
    summary: pd.DataFrame,
    confidence_level: float,
) -> None:
    table = summary.round(2).reset_index()
    table_md = "\n".join(
        [
            (
                "| scenario | co2_2100 | forcing_2100_wm2 | temp_2100 | "
                "temp_lower_2100 | temp_upper_2100 | equilibrium_temp_2100 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                (
                    f"| {row.scenario} | {row.co2_2100:.2f} | {row.forcing_2100_wm2:.2f} | "
                    f"{row.temp_2100:.2f} | {row.temp_lower_2100:.2f} | "
                    f"{row.temp_upper_2100:.2f} | {row.equilibrium_temp_2100:.2f} |"
                )
                for row in table.itertuples(index=False)
            ],
        ]
    )
    text = f"""# Energy-Balance Sensitivity Report

## Model

- Framework: one-box energy-balance sensitivity model.
- Specification: `{ebm.model_name}`.
- CO2 forcing: `5.35 * log(CO2 / CO2_1951)`.
- Fitted baseline temperature: {ebm.base_temp:.3f}.
- Fitted sensitivity per W/m2: {ebm.sensitivity:.3f}.
- Fitted annual adjustment rate: {ebm.adjustment_rate:.3f}.
- Adjustment-rate cap: {ebm.max_adjustment_rate:.3f}.
- Residual standard deviation: {ebm.residual_std:.3f}.
- Forecast interval: {confidence_level:.0%}, based on recursive residual propagation.

## Scenario Results

{table_md}

## Interpretation

This is a sensitivity model, not a pure statistical fit. The logarithmic CO2
forcing already implements the standard diminishing marginal forcing of CO2.
The bounded adjustment rate adds thermal inertia. That makes high-CO2 scenarios
less explosive than the unconstrained structural time-series model, but the cap
is an identifying assumption because annual German extreme temperatures cannot
estimate global heat uptake by themselves.
"""
    output.write_text(text, encoding="utf-8")


def write_climate_modulated_ebm_report(
    output: Path,
    model: ClimateModulatedEBMResult,
    summary: pd.DataFrame,
    confidence_level: float,
) -> None:
    table = summary.round(2).reset_index()
    table_md = "\n".join(
        [
            (
                "| scenario | regional_state | co2_2100 | forcing_2100_wm2 | "
                "sunshine_z | precip_z | temp_2100 | temp_lower_2100 | temp_upper_2100 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            *[
                (
                    f"| {row.scenario} | {row.regional_state} | {row.co2_2100:.2f} | "
                    f"{row.forcing_2100_wm2:.2f} | {row.sunshine_z:.2f} | {row.precip_z:.2f} | "
                    f"{row.temp_2100:.2f} | {row.temp_lower_2100:.2f} | {row.temp_upper_2100:.2f} |"
                )
                for row in table.itertuples(index=False)
            ],
        ]
    )
    text = f"""# Climate-Modulated Energy-Balance Report

## Model

- Framework: energy-balance dynamic regression with regional climate modulation.
- Specification: `{model.model_name}`.
- CO2 forcing: `5.35 * log(CO2 / CO2_1951)`.
- Sunshine and precipitation enter as historical z-scores.
- Fitted baseline temperature: {model.base_temp:.3f}.
- Fitted CO2 sensitivity per W/m2: {model.co2_sensitivity:.3f}.
- Fitted sunshine effect per standard deviation: {model.sunshine_effect:.3f}.
- Fitted precipitation effect per standard deviation: {model.precip_effect:.3f}.
- Fitted annual adjustment rate: {model.adjustment_rate:.3f}.
- Adjustment-rate cap: {model.max_adjustment_rate:.3f}.
- CO2 prior sensitivity per W/m2: {model.co2_prior_sensitivity:.3f}.
- CO2 prior weight: {model.co2_prior_weight:.3f}.
- Regional-covariate ridge weight: {model.regional_prior_weight:.3f}.
- Residual standard deviation: {model.residual_std:.3f}.
- Forecast interval: {confidence_level:.0%}, based on recursive residual propagation.

## Regional Scenario States

- `normal`: historical mean sunshine and precipitation.
- `hot_dry`: sunshine one historical standard deviation above normal, precipitation one standard deviation below normal.
- `cloudy_wet`: sunshine one historical standard deviation below normal, precipitation one standard deviation above normal.

## Scenario Results

{table_md}

## Interpretation

This model combines the structural CO2 forcing idea with the VARX insight that
sunshine and precipitation are relevant regional climate covariates. Sunshine
captures shortwave radiation and cloudiness; precipitation is a coarse moisture
and evaporative-cooling proxy. The fit is intentionally regularized: without
that structure, sunshine can absorb the common historical trend and drive the
CO2 coefficient to zero, which is statistically possible but physically not a
credible attribution model. The fitted CO2 effect is therefore interpreted as
the global forcing component conditional on regional radiation and moisture
conditions.
"""
    output.write_text(text, encoding="utf-8")
