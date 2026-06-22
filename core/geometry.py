from __future__ import annotations

import zipfile
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
import rasterio.features as rfeatures
from rasterio.transform import from_origin


def grid_transform_from_centers(lon_values: np.ndarray, lat_values: np.ndarray):
    lon_values = np.asarray(lon_values)
    lat_values = np.asarray(lat_values)
    dx = float(abs(np.nanmedian(np.diff(lon_values))))
    dy = float(abs(np.nanmedian(np.diff(lat_values))))
    west = float(np.nanmin(lon_values) - dx / 2.0)
    north = float(np.nanmax(lat_values) + dy / 2.0)
    return from_origin(west, north, dx, dy)


def geometry_mask_dataarray(geometry, lat_values, lon_values, lat_name="latitude", lon_name="longitude") -> xr.DataArray:
    lat_values = np.asarray(lat_values)
    lon_values = np.asarray(lon_values)
    if len(lat_values) >= 2 and lat_values[1] > lat_values[0]:
        lat_values_for_mask = lat_values[::-1]
        flipped = True
    else:
        lat_values_for_mask = lat_values
        flipped = False
    transform = grid_transform_from_centers(lon_values=lon_values, lat_values=lat_values_for_mask)
    mask = rfeatures.geometry_mask(
        [mapping(geometry)],
        out_shape=(len(lat_values_for_mask), len(lon_values)),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    if flipped:
        mask = mask[::-1, :]
    return xr.DataArray(mask, dims=(lat_name, lon_name), coords={lat_name: lat_values, lon_name: lon_values}, name="polygon_mask")


def load_vector_from_zip(uploaded_file) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "uploaded_vector.zip"
        zip_path.write_bytes(uploaded_file.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        candidates = list(tmpdir.rglob("*.shp")) + list(tmpdir.rglob("*.gpkg")) + list(tmpdir.rglob("*.geojson")) + list(tmpdir.rglob("*.json"))
        if not candidates:
            raise ValueError("El ZIP no contiene .shp, .gpkg, .geojson o .json.")
        gdf = gpd.read_file(candidates[0])
    if gdf.empty:
        raise ValueError("El archivo vectorial no contiene geometrías.")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    if gdf.empty:
        raise ValueError("El archivo vectorial no contiene geometrías válidas.")
    return gdf


def dissolve_geometry(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.geometry.unary_union
