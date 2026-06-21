from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict

import requests
import xarray as xr

GOOGLE_DRIVE_CONFIRM_URL = "https://drive.google.com/uc?export=download"


def parse_google_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    patterns = [r"[?&]id=([A-Za-z0-9_-]+)", r"/file/d/([A-Za-z0-9_-]+)", r"/d/([A-Za-z0-9_-]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url.strip()):
        return url.strip()
    return None


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    text = response.text or ""
    match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
    if match:
        return match.group(1)
    match = re.search(r'name="confirm"\s+value="([^"]+)"', text)
    if match:
        return match.group(1)
    return None


def _looks_like_html(path: Path, n: int = 512) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(n).lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"text/html" in head[:200]
    except Exception:
        return False


def download_google_drive_file(url: str, cache_dir: str = "/tmp/hydroclimate_dashboard_cache", filename: str = "dataset.nc") -> str:
    file_id = parse_google_drive_file_id(url)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    local_path = Path(cache_dir) / f"{key}_{filename}"

    if local_path.exists() and local_path.stat().st_size > 10_000_000 and not _looks_like_html(local_path):
        return str(local_path)

    if file_id is None:
        raise ValueError("No pude extraer el FILE_ID de Google Drive. Usa https://drive.google.com/uc?id=FILE_ID")

    session = requests.Session()
    response = session.get(GOOGLE_DRIVE_CONFIRM_URL, params={"id": file_id}, stream=True, timeout=(20, 180))
    token = _get_confirm_token(response)
    if token:
        response = session.get(GOOGLE_DRIVE_CONFIRM_URL, params={"id": file_id, "confirm": token}, stream=True, timeout=(20, 180))
    response.raise_for_status()

    tmp_path = local_path.with_suffix(".part")
    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp_path.replace(local_path)

    if _looks_like_html(local_path):
        try:
            preview = local_path.read_text(errors="ignore")[:300]
        except Exception:
            preview = ""
        raise RuntimeError("Google Drive devolvió HTML, no NetCDF. El archivo quizá no está público, no es descargable o Drive bloqueó la descarga. Vista previa: " + preview)

    if local_path.stat().st_size < 10_000_000:
        raise RuntimeError(f"El archivo descargado pesa solo {local_path.stat().st_size} bytes. No parece ser el NetCDF esperado.")
    return str(local_path)


def open_netcdf_local(path: str, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
    chunks = chunks or {"Z1": 365, "latitude": 50, "longitude": 50}
    errors = []
    for engine in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(path, engine=engine, chunks=chunks, decode_times=False, mask_and_scale=True)
        except Exception as exc:
            errors.append(f"{engine}: {repr(exc)}")
    raise RuntimeError("No se pudo abrir el NetCDF local. Errores: " + " | ".join(errors))


def load_dataset_from_google_drive(url: str, cache_dir: str = "/tmp/hydroclimate_dashboard_cache") -> Tuple[xr.Dataset, str]:
    local_path = download_google_drive_file(url=url, cache_dir=cache_dir, filename="precipitation.nc")
    ds = open_netcdf_local(local_path)
    return ds, local_path


def infer_dataset_roles(ds: xr.Dataset) -> Dict[str, str]:
    all_names = list(ds.variables)
    lat_candidates = [n for n in all_names if n.lower() in ("lat", "latitude", "y")]
    lon_candidates = [n for n in all_names if n.lower() in ("lon", "longitude", "x")]
    time_candidates = [n for n in all_names if n.lower() in ("time", "z1", "date", "day")]
    lat_name = lat_candidates[0] if lat_candidates else "latitude"
    lon_name = lon_candidates[0] if lon_candidates else "longitude"
    time_name = time_candidates[0] if time_candidates else "Z1"

    if "precipitation" in ds.data_vars:
        var_name = "precipitation"
    else:
        data_vars = []
        for name, da in ds.data_vars.items():
            if name.lower() in ("crs", "spatial_ref"):
                continue
            if lat_name in da.dims and lon_name in da.dims:
                data_vars.append(name)
        if not data_vars:
            raise ValueError("No se encontró variable con dimensiones de latitud y longitud.")
        var_name = data_vars[0]
    return {"variable": var_name, "lat": lat_name, "lon": lon_name, "time": time_name}


def clean_precipitation_da(da: xr.DataArray) -> xr.DataArray:
    fill_value = da.attrs.get("_FillValue", None)
    if fill_value is not None:
        da = da.where(da != fill_value)
    da = da.where(da > -1e20)
    da = da.where(da >= 0)
    return da


def ensure_latitude_descending(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    lat_values = da[lat_name].values
    if len(lat_values) >= 2 and lat_values[1] > lat_values[0]:
        return da.sortby(lat_name, ascending=False)
    return da
