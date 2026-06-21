from __future__ import annotations

import pandas as pd
import xarray as xr


def build_time_index(n: int, start_date: str = "1981-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, periods=n, freq="D")


def assign_datetime_coordinate(da: xr.DataArray, time_dim: str, start_date: str = "1981-01-01") -> xr.DataArray:
    if time_dim not in da.dims:
        return da
    n = da.sizes[time_dim]
    time_index = build_time_index(n, start_date=start_date)
    da = da.assign_coords({time_dim: time_index})
    return da.rename({time_dim: "time"})
