from __future__ import annotations

import pandas as pd


def monthly_climatology(df: pd.DataFrame, value_col: str = "precipitation", base_start=None, base_end=None) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    if base_start:
        out = out[out["time"] >= pd.to_datetime(base_start)]
    if base_end:
        out = out[out["time"] <= pd.to_datetime(base_end)]
    out["month"] = out["time"].dt.month
    out["month_name"] = out["time"].dt.month_name()
    clim = out.groupby(["month", "month_name"])[value_col].agg(
        mean="mean",
        median="median",
        total_mean=lambda s: s.groupby(out.loc[s.index, "time"].dt.to_period("M")).sum().mean(),
        p10=lambda s: s.quantile(0.10),
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
        max="max",
    ).reset_index().sort_values("month")
    return clim


def dayofyear_climatology(df: pd.DataFrame, value_col: str = "precipitation", base_start=None, base_end=None) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    if base_start:
        out = out[out["time"] >= pd.to_datetime(base_start)]
    if base_end:
        out = out[out["time"] <= pd.to_datetime(base_end)]
    out["dayofyear"] = out["time"].dt.dayofyear
    return out.groupby("dayofyear")[value_col].agg(mean="mean", median="median", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90), max="max").reset_index()


def monthly_anomalies(df: pd.DataFrame, value_col: str = "precipitation", base_start=None, base_end=None) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    monthly = out.groupby(["year", "month"])[value_col].sum().reset_index()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    clim_input = out.copy()
    if base_start:
        clim_input = clim_input[clim_input["time"] >= pd.to_datetime(base_start)]
    if base_end:
        clim_input = clim_input[clim_input["time"] <= pd.to_datetime(base_end)]
    clim_input["year"] = clim_input["time"].dt.year
    clim_input["month"] = clim_input["time"].dt.month
    monthly_base = clim_input.groupby(["year", "month"])[value_col].sum().reset_index()
    clim = monthly_base.groupby("month")[value_col].agg(climatological_mean="mean", climatological_std="std").reset_index()
    result = monthly.merge(clim, on="month", how="left")
    result["anomaly_mm"] = result[value_col] - result["climatological_mean"]
    result["standardized_anomaly"] = result["anomaly_mm"] / result["climatological_std"]
    return result
