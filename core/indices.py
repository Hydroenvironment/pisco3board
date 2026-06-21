import numpy as np
import pandas as pd
from scipy.stats import norm

def hydro_metrics(ts):
    return {
        "mean": float(np.nanmean(ts)),
        "max": float(np.nanmax(ts)),
        "min": float(np.nanmin(ts)),
        "std": float(np.nanstd(ts)),
        "total": float(np.nansum(ts))
    }

def extreme_indices(ts):
    ts = pd.Series(ts)
    return {
        "Rx1day": float(ts.max()),
        "Rx5day": float(ts.rolling(5).sum().max()),
        "DryDays": int((ts < 1).sum()),
        "WetDays": int((ts > 10).sum())
    }

def spi_like(ts):
    ts = np.array(ts)
    z = (ts - np.nanmean(ts)) / np.nanstd(ts)
    return norm.cdf(z)
