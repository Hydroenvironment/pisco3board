from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def spatial_kmeans_from_mean_field(da: xr.DataArray, time_name: str = "time", lat_name: str = "latitude", lon_name: str = "longitude", n_clusters: int = 5) -> xr.DataArray:
    if time_name in da.dims:
        field = da.mean(dim=time_name, skipna=True).compute()
    else:
        field = da.compute()
    values = field.values
    flat = values.reshape(-1, 1)
    valid = np.isfinite(flat[:, 0])
    labels = np.full(flat.shape[0], np.nan)
    if valid.sum() < n_clusters:
        raise ValueError("No hay suficientes celdas válidas para el número de clusters solicitado.")
    X = StandardScaler().fit_transform(flat[valid])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels[valid] = km.fit_predict(X) + 1
    label_grid = labels.reshape(values.shape)
    return xr.DataArray(label_grid, dims=(lat_name, lon_name), coords={lat_name: field[lat_name].values, lon_name: field[lon_name].values}, name="spatial_cluster")
