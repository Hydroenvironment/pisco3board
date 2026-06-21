from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def longest_spell(condition: np.ndarray) -> int:
    condition = np.asarray(condition, dtype=bool)
    max_len = 0
    current = 0
    for value in condition:
        if value:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0
    return int(max_len)


def base_metrics(df: pd.DataFrame, value_col: str = "precipitation", wet_threshold: float = 1.0) -> pd.DataFrame:
    x = pd.to_numeric(df[value_col], errors="coerce").astype(float).to_numpy()
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return pd.DataFrame([{"metric": "valid_observations", "value": 0, "unit": "count"}])
    wet = valid >= wet_threshold
    rows = [
        ("valid_observations", valid.size, "count"),
        ("missing_observations", int(np.sum(~np.isfinite(x))), "count"),
        ("total_precipitation", np.nansum(valid), "mm"),
        ("mean_daily_precipitation", np.nanmean(valid), "mm/day"),
        ("median_daily_precipitation", np.nanmedian(valid), "mm/day"),
        ("standard_deviation", np.nanstd(valid, ddof=1), "mm/day"),
        ("coefficient_of_variation", np.nanstd(valid, ddof=1) / np.nanmean(valid) if np.nanmean(valid) != 0 else np.nan, "dimensionless"),
        ("minimum", np.nanmin(valid), "mm/day"),
        ("maximum", np.nanmax(valid), "mm/day"),
        ("p10", np.nanpercentile(valid, 10), "mm/day"),
        ("p25", np.nanpercentile(valid, 25), "mm/day"),
        ("p50", np.nanpercentile(valid, 50), "mm/day"),
        ("p75", np.nanpercentile(valid, 75), "mm/day"),
        ("p90", np.nanpercentile(valid, 90), "mm/day"),
        ("p95", np.nanpercentile(valid, 95), "mm/day"),
        ("p99", np.nanpercentile(valid, 99), "mm/day"),
        ("wet_days", int(np.sum(wet)), "days"),
        ("dry_days", int(np.sum(valid < wet_threshold)), "days"),
        ("wet_day_fraction", float(np.sum(wet) / valid.size), "fraction"),
        ("simple_daily_intensity_index", np.nansum(valid[wet]) / np.sum(wet) if np.sum(wet) > 0 else np.nan, "mm/wet day"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def etccdi_precipitation_indices(df: pd.DataFrame, value_col: str = "precipitation", wet_threshold: float = 1.0, heavy_threshold: float = 10.0, very_heavy_threshold: float = 20.0, extreme_threshold: float = 50.0) -> pd.DataFrame:
    valid = pd.to_numeric(df[value_col], errors="coerce").astype(float).dropna()
    if valid.empty:
        return pd.DataFrame(columns=["index", "value", "unit", "description"])
    p95 = valid.quantile(0.95)
    p99 = valid.quantile(0.99)
    rolling3 = valid.rolling(3, min_periods=3).sum()
    rolling5 = valid.rolling(5, min_periods=5).sum()
    rolling7 = valid.rolling(7, min_periods=7).sum()
    rows = [
        ("PRCPTOT", valid[valid >= wet_threshold].sum(), "mm", "Total precipitation on wet days."),
        ("SDII", valid[valid >= wet_threshold].sum() / (valid >= wet_threshold).sum() if (valid >= wet_threshold).sum() > 0 else np.nan, "mm/wet day", "Simple daily intensity index."),
        ("Rx1day", valid.max(), "mm", "Maximum 1-day precipitation."),
        ("Rx3day", rolling3.max(), "mm", "Maximum consecutive 3-day precipitation."),
        ("Rx5day", rolling5.max(), "mm", "Maximum consecutive 5-day precipitation."),
        ("Rx7day", rolling7.max(), "mm", "Maximum consecutive 7-day precipitation."),
        ("R10mm", (valid >= heavy_threshold).sum(), "days", f"Days with precipitation >= {heavy_threshold} mm."),
        ("R20mm", (valid >= very_heavy_threshold).sum(), "days", f"Days with precipitation >= {very_heavy_threshold} mm."),
        ("R50mm", (valid >= extreme_threshold).sum(), "days", f"Days with precipitation >= {extreme_threshold} mm."),
        ("R95p", valid[valid > p95].sum(), "mm", "Total precipitation above the local 95th percentile."),
        ("R99p", valid[valid > p99].sum(), "mm", "Total precipitation above the local 99th percentile."),
        ("R95p_days", (valid > p95).sum(), "days", "Number of days above the local 95th percentile."),
        ("R99p_days", (valid > p99).sum(), "days", "Number of days above the local 99th percentile."),
        ("CDD", longest_spell(valid.to_numpy() < wet_threshold), "days", f"Maximum number of consecutive dry days below {wet_threshold} mm."),
        ("CWD", longest_spell(valid.to_numpy() >= wet_threshold), "days", f"Maximum number of consecutive wet days >= {wet_threshold} mm."),
    ]
    return pd.DataFrame(rows, columns=["index", "value", "unit", "description"])


def annual_statistics(df: pd.DataFrame, value_col: str = "precipitation") -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["year"] = out["time"].dt.year
    return out.groupby("year")[value_col].agg(total="sum", mean="mean", max="max", std="std", wet_days=lambda s: (s >= 1.0).sum(), dry_days=lambda s: (s < 1.0).sum()).reset_index()


def monthly_statistics(df: pd.DataFrame, value_col: str = "precipitation") -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    out["month_name"] = out["time"].dt.month_name()
    return out.groupby(["year", "month", "month_name"])[value_col].agg(total="sum", mean="mean", max="max", wet_days=lambda s: (s >= 1.0).sum(), dry_days=lambda s: (s < 1.0).sum()).reset_index()


def mann_kendall_approximation(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return {"tau": np.nan, "p_value": np.nan, "interpretation": "insufficient data"}
    tau, p_value = stats.kendalltau(np.arange(n), y)
    if np.isnan(p_value):
        interp = "undefined"
    elif p_value < 0.05 and tau > 0:
        interp = "significant increasing trend"
    elif p_value < 0.05 and tau < 0:
        interp = "significant decreasing trend"
    else:
        interp = "not significant at alpha=0.05"
    return {"tau": float(tau), "p_value": float(p_value), "interpretation": interp}
