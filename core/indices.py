from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def standardized_precipitation_anomaly(df: pd.DataFrame, value_col: str = "precipitation") -> pd.Series:
    x = pd.to_numeric(df[value_col], errors="coerce").astype(float)
    mean = x.mean()
    std = x.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=df.index)
    return (x - mean) / std


def spi_gamma_monthly(df: pd.DataFrame, scale_months: int = 3, value_col: str = "precipitation") -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.set_index("time").sort_index()
    monthly = out[value_col].resample("MS").sum(min_count=1).to_frame("monthly_precipitation")
    monthly[f"accumulated_{scale_months}_month"] = monthly["monthly_precipitation"].rolling(scale_months, min_periods=scale_months).sum()
    values = monthly[f"accumulated_{scale_months}_month"]
    spi = pd.Series(index=monthly.index, dtype=float)
    for month in range(1, 13):
        idx = values.index.month == month
        sample = values[idx].dropna()
        if len(sample) < 10:
            continue
        zero_probability = (sample <= 0).mean()
        positive = sample[sample > 0]
        if len(positive) < 8:
            continue
        try:
            shape, loc, scale = stats.gamma.fit(positive, floc=0)
            cdf = pd.Series(np.nan, index=sample.index, dtype=float)
            positive_values = sample[sample > 0]
            cdf.loc[positive_values.index] = zero_probability + (1.0 - zero_probability) * stats.gamma.cdf(positive_values, a=shape, loc=loc, scale=scale)
            cdf.loc[sample <= 0] = zero_probability / 2.0
            cdf = cdf.clip(1e-6, 1.0 - 1e-6)
            spi.loc[cdf.index] = stats.norm.ppf(cdf)
        except Exception:
            z = (sample - sample.mean()) / sample.std(ddof=1)
            spi.loc[z.index] = z
    monthly[f"SPI_{scale_months}"] = spi
    monthly = monthly.reset_index().rename(columns={"time": "month"})
    return monthly


def classify_spi(value: float) -> str:
    if pd.isna(value):
        return "No data"
    if value <= -2.0:
        return "Extremely dry"
    if value <= -1.5:
        return "Severely dry"
    if value <= -1.0:
        return "Moderately dry"
    if value < 1.0:
        return "Near normal"
    if value < 1.5:
        return "Moderately wet"
    if value < 2.0:
        return "Very wet"
    return "Extremely wet"
