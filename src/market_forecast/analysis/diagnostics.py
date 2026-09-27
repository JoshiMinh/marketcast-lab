from __future__ import annotations

import numpy as np
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def residual_diagnostics(residuals: np.ndarray, lags: int = 10) -> dict[str, object]:
    values = np.asarray(residuals, dtype=float).reshape(-1)
    usable_lags = min(lags, max(1, len(values) // 2 - 1))
    if np.allclose(values, values[0]):
        return {
            "residual_acf": [1.0] + [0.0] * usable_lags,
            "ljung_box_lag": usable_lags,
            "ljung_box_stat": 0.0,
            "ljung_box_pvalue": 1.0,
        }
    correlations = acf(values, nlags=usable_lags, fft=False)
    ljung = acorr_ljungbox(values, lags=[usable_lags], return_df=True)
    return {
        "residual_acf": correlations.tolist(),
        "ljung_box_lag": usable_lags,
        "ljung_box_stat": float(ljung["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(ljung["lb_pvalue"].iloc[0]),
    }


def series_diagnostics(values: np.ndarray, lags: int = 20) -> dict[str, object]:
    array = np.asarray(values, dtype=float).reshape(-1)
    usable = min(lags, len(array) // 2 - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(array, autolag="AIC", result_object=True)
        kpss_result = kpss(array, regression="c", nlags="auto", result_object=True)
    decomposition = seasonal_decompose(array, model="additive", period=7, extrapolate_trend="period")
    rolling = np.convolve(array, np.ones(7) / 7, mode="valid")
    return {
        "adf_stat": float(adf.statistic), "adf_pvalue": float(adf.pvalue),
        "kpss_stat": float(kpss_result.statistic), "kpss_pvalue": float(kpss_result.pvalue),
        "acf": acf(array, nlags=usable, fft=True).tolist(),
        "pacf": pacf(array, nlags=usable).tolist(),
        "rolling_mean_7_latest": float(rolling[-1]),
        "decomposition_period": 7,
        "trend_latest": float(decomposition.trend[-1]),
        "seasonal_latest": float(decomposition.seasonal[-1]),
        "residual_std": float(np.nanstd(decomposition.resid)),
    }
