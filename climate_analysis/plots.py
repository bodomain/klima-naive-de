"""Plotting helpers for historical data, diagnostics, and forecasts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from .config import MODEL_VARIABLES, SOURCE_NOTE, VARIABLE_COLORS, VARIABLE_LABELS
from .models import ModelBundle, choose_lag_order

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

def save_irf_plot(bundle: ModelBundle, output: Path, periods: int = 20) -> None:
    if bundle.model_type == "EXOGENOUS_CO2_VARX":
        logging.info("Skipping IRF plot for EXOGENOUS_CO2_VARX; use forecast comparison for the imposed CO2 path.")
        return
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
    if stationary_data is not None and "d_co2" not in stationary_data.columns and len(stationary_data.columns) <= 3:
        pass
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

def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    output: Path,
    confidence_level: float = 0.90,
    model_label: str = "naive forecast",
    scenario_label: str | None = None,
    co2_scenario_label: str | None = None,
    forecast_label: str = "Naive forecast",
    co2_forecast_label: str | None = None,
    reference_paths: dict[str, pd.Series] | None = None,
) -> None:
    plot_cols = [col for col in MODEL_VARIABLES if col in history.columns]
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(13, 2.9 * len(plot_cols)), sharex=True)
    axes = np.atleast_1d(axes)
    transition = int(history.index.max())

    for ax, col in zip(axes, plot_cols):
        ax.plot(history.index, history[col], color=VARIABLE_COLORS[col], lw=1.5, label="Historical data")
        line_label = co2_forecast_label if col == "co2" and co2_forecast_label else forecast_label
        ax.plot(forecast.index, forecast[col], color=VARIABLE_COLORS[col], lw=1.8, ls="--", label=line_label)
        if reference_paths and col in reference_paths:
            ref = reference_paths[col].dropna()
            ax.plot(
                ref.index,
                ref.values,
                color="#555555",
                lw=1.5,
                ls=":",
                label="Documentary exp. CO2 trend (not model input)",
            )
        ax.fill_between(
            forecast.index,
            lower[col].astype(float).values,
            upper[col].astype(float).values,
            color=VARIABLE_COLORS[col],
            alpha=0.18,
            label=f"{confidence_level:.0%} confidence interval",
        )
        ax.axvline(transition, color="black", lw=1, alpha=0.6)
        ax.set_ylabel(VARIABLE_LABELS[col])
        if col == "co2" and co2_scenario_label:
            ax.text(
                0.01,
                0.92,
                co2_scenario_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=VARIABLE_COLORS[col],
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": VARIABLE_COLORS[col], "alpha": 0.85},
            )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    title = f"Historical data and {model_label} forecast to 2100"
    if scenario_label:
        title += f" ({scenario_label})"
    fig.suptitle(title, y=0.995)
    add_source_note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=160)
    plt.close(fig)
