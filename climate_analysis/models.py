"""Statistical tests, VAR/VECM/VARX estimation, and forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

from .config import CLIMATE_VARIABLES

@dataclass
class ModelBundle:
    model_type: str
    result: object
    data_used: pd.DataFrame
    lag_order: int
    coint_rank: int
    integration_order: dict[str, int]


@dataclass
class ExogenousCO2Result:
    params: np.ndarray
    climate_cols: list[str]
    co2_feature: str
    co2_feature_col: str
    co2_base: float
    lag_order: int
    resid_std: pd.Series
    fitted_values: pd.DataFrame
    residuals: pd.DataFrame


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

def co2_feature_column(co2_feature: str) -> str:
    if co2_feature == "dppm":
        return "d_co2"
    if co2_feature == "dlog":
        return "dlog_co2"
    if co2_feature == "forcing":
        return "log_co2_ratio"
    raise ValueError(f"Unsupported CO2 feature: {co2_feature}")


def add_co2_feature(data: pd.DataFrame, co2_feature: str = "dppm") -> pd.DataFrame:
    out = data.copy()
    feature_col = co2_feature_column(co2_feature)
    if co2_feature == "dppm":
        out[feature_col] = out["co2"].diff()
    elif co2_feature == "dlog":
        out[feature_col] = np.log(out["co2"]).diff()
    elif co2_feature == "forcing":
        out[feature_col] = np.log(out["co2"] / float(out["co2"].iloc[0]))
    return out


def co2_feature_value(
    current_co2: float,
    previous_co2: float,
    co2_feature: str,
    co2_base: float,
) -> float:
    if co2_feature == "dppm":
        return current_co2 - previous_co2
    if co2_feature == "dlog":
        return float(np.log(current_co2) - np.log(previous_co2))
    if co2_feature == "forcing":
        return float(np.log(current_co2 / co2_base))
    raise ValueError(f"Unsupported CO2 feature: {co2_feature}")


def initial_climate_history(
    levels_data: pd.DataFrame,
    climate_cols: list[str],
    lag_order: int,
    start_mode: str,
    trend_window: int,
) -> list[np.ndarray]:
    raw_history = [row.copy() for row in levels_data[climate_cols].values[-lag_order:]]
    if start_mode == "last":
        return raw_history
    if start_mode != "trend":
        raise ValueError(f"Unsupported forecast start mode: {start_mode}")

    smoothed = (
        levels_data[climate_cols]
        .rolling(trend_window, min_periods=max(3, trend_window // 2))
        .mean()
        .iloc[-1]
        .values
    )
    if np.isnan(smoothed).any():
        return raw_history
    history = raw_history.copy()
    history[-1] = smoothed
    return history


def prepare_mixed_stationary_data(data: pd.DataFrame, co2_feature: str = "dppm") -> pd.DataFrame:
    """Stationary mixed specification for climate levels plus CO2 growth."""
    mixed = pd.DataFrame(index=data.index)
    for col in CLIMATE_VARIABLES:
        if col in data.columns:
            mixed[col] = data[col]
    feature_col = co2_feature_column(co2_feature)
    mixed[feature_col] = add_co2_feature(data, co2_feature)[feature_col]
    return mixed.dropna()

def make_lagged_exog_design(
    climate_data: pd.DataFrame,
    d_co2: pd.Series,
    lag_order: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    rows = []
    y_rows = []
    row_index = []
    climate_values = climate_data.values
    d_co2_values = d_co2.values
    for pos in range(lag_order, len(climate_data)):
        features = [1.0]
        for lag in range(1, lag_order + 1):
            features.extend(climate_values[pos - lag])
        features.append(d_co2_values[pos])
        rows.append(features)
        y_rows.append(climate_values[pos])
        row_index.append(climate_data.index[pos])
    return np.asarray(rows), np.asarray(y_rows), pd.Index(row_index, name=climate_data.index.name)

def fit_exogenous_co2_model(data: pd.DataFrame, maxlags: int = 6, co2_feature: str = "dlog") -> ModelBundle:
    model_data = prepare_mixed_stationary_data(data, co2_feature=co2_feature)
    feature_col = co2_feature_column(co2_feature)
    climate_cols = [col for col in CLIMATE_VARIABLES if col in model_data.columns]
    climate_data = model_data[climate_cols]
    lag = choose_lag_order(climate_data, maxlags=maxlags)
    x, y, fit_index = make_lagged_exog_design(climate_data, model_data[feature_col], lag)
    params, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = pd.DataFrame(x @ params, index=fit_index, columns=climate_cols)
    residuals = pd.DataFrame(y - fitted.values, index=fit_index, columns=climate_cols)
    result = ExogenousCO2Result(
        params=params,
        climate_cols=climate_cols,
        co2_feature=co2_feature,
        co2_feature_col=feature_col,
        co2_base=float(data["co2"].iloc[0]),
        lag_order=lag,
        resid_std=residuals.std(),
        fitted_values=fitted,
        residuals=residuals,
    )
    logging.info(
        "Fitted EXOGENOUS_CO2 dynamic regression with climate lags=%s and CO2 feature=%s",
        lag,
        co2_feature,
    )
    return ModelBundle("EXOGENOUS_CO2_VARX", result, model_data, lag, 0, {})

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
    model_mode: str = "auto",
    co2_feature: str = "dlog",
) -> ModelBundle:
    if model_mode == "exogenous-co2":
        bundle = fit_exogenous_co2_model(data, maxlags=maxlags, co2_feature=co2_feature)
        bundle.integration_order = integration_order
        return bundle

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

def granger_tests(
    data: pd.DataFrame,
    integration_order: dict[str, int],
    maxlags: int = 6,
    co2_feature: str = "dppm",
) -> pd.DataFrame:
    all_i1 = all(order == 1 for order in integration_order.values())
    if all_i1:
        test_data = data.diff().dropna()
        co2_name = "co2"
    else:
        test_data = prepare_mixed_stationary_data(data, co2_feature=co2_feature)
        co2_name = co2_feature_column(co2_feature)
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
    external_co2_path: pd.Series | None = None,
    interval_method: str = "bootstrap",
    bootstrap_sims: int = 2000,
    random_seed: int = 42,
    forecast_start: str = "trend",
    trend_window: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    last_year = int(levels_data.index.max())
    future_years = np.arange(last_year + 1, horizon_end + 1)
    steps = len(future_years)
    columns = list(levels_data.columns)
    if external_co2_path is None:
        external_co2 = scenario_co2_path(
            levels_data["co2"],
            future_years,
            lookback=co2_lookback,
            scenario=co2_scenario,
        )
    else:
        path = external_co2_path.astype(float).copy()
        if last_year in path.index:
            ratio = float(levels_data["co2"].iloc[-1]) / float(path.loc[last_year])
            path = path * ratio
        external_co2 = path.reindex(future_years).astype(float)
        if external_co2.isna().any():
            missing = external_co2[external_co2.isna()].index.tolist()
            raise RuntimeError(f"External CO2 path is missing forecast years: {missing[:5]}")
        external_co2.name = "co2"

    if bundle.model_type == "EXOGENOUS_CO2_VARX":
        res: ExogenousCO2Result = bundle.result
        climate_cols = res.climate_cols
        lag = res.lag_order
        history = initial_climate_history(levels_data, climate_cols, lag, forecast_start, trend_window)
        last_co2 = float(levels_data["co2"].iloc[-1])
        pred_rows = []
        lower_rows = []
        upper_rows = []
        z = stats.norm.ppf(1 - alpha / 2)

        for step_idx, year in enumerate(future_years, start=1):
            co2_driver = co2_feature_value(
                current_co2=float(external_co2.loc[year]),
                previous_co2=last_co2,
                co2_feature=res.co2_feature,
                co2_base=res.co2_base,
            )
            features = [1.0]
            for lag_idx in range(1, lag + 1):
                features.extend(history[-lag_idx])
            features.append(co2_driver)
            pred_climate = np.asarray(features) @ res.params
            history.append(pred_climate)
            last_co2 = float(external_co2.loc[year])

            row = {col: val for col, val in zip(climate_cols, pred_climate)}
            row["co2"] = external_co2.loc[year]
            pred_rows.append(row)

        forecast = pd.DataFrame(pred_rows, index=future_years)[columns]

        if interval_method == "bootstrap":
            rng = np.random.default_rng(random_seed)
            residual_values = res.residuals[climate_cols].values
            simulated = np.empty((bootstrap_sims, steps, len(climate_cols)), dtype=float)
            for sim_idx in range(bootstrap_sims):
                sim_history = initial_climate_history(levels_data, climate_cols, lag, forecast_start, trend_window)
                sim_last_co2 = float(levels_data["co2"].iloc[-1])
                for step_pos, year in enumerate(future_years):
                    co2_driver = co2_feature_value(
                        current_co2=float(external_co2.loc[year]),
                        previous_co2=sim_last_co2,
                        co2_feature=res.co2_feature,
                        co2_base=res.co2_base,
                    )
                    features = [1.0]
                    for lag_idx in range(1, lag + 1):
                        features.extend(sim_history[-lag_idx])
                    features.append(co2_driver)
                    deterministic = np.asarray(features) @ res.params
                    innovation = residual_values[rng.integers(0, len(residual_values))]
                    sim_row = deterministic + innovation
                    simulated[sim_idx, step_pos, :] = sim_row
                    sim_history.append(sim_row)
                    sim_last_co2 = float(external_co2.loc[year])

            lower_q = np.quantile(simulated, alpha / 2, axis=0)
            upper_q = np.quantile(simulated, 1 - alpha / 2, axis=0)
            lower_df = pd.DataFrame(lower_q, index=future_years, columns=climate_cols)
            upper_df = pd.DataFrame(upper_q, index=future_years, columns=climate_cols)
            lower_df["co2"] = external_co2
            upper_df["co2"] = external_co2
            lower_df = lower_df[columns]
            upper_df = upper_df[columns]
        elif interval_method == "analytic":
            for step_idx, year in enumerate(future_years, start=1):
                row = forecast.loc[year]
                interval = z * np.sqrt(step_idx) * res.resid_std
                lower_row = {col: row[col] - interval[col] for col in climate_cols}
                upper_row = {col: row[col] + interval[col] for col in climate_cols}
                lower_row["co2"] = external_co2.loc[year]
                upper_row["co2"] = external_co2.loc[year]
                lower_rows.append(lower_row)
                upper_rows.append(upper_row)
            lower_df = pd.DataFrame(lower_rows, index=future_years)[columns]
            upper_df = pd.DataFrame(upper_rows, index=future_years)[columns]
        else:
            raise ValueError(f"Unsupported interval_method: {interval_method}")
        return forecast, lower_df, upper_df, external_co2

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
