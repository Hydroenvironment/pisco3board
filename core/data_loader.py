from __future__ import annotations

import re
import html
import hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple
from urllib.parse import urljoin

import requests
import xarray as xr

GOOGLE_DRIVE_UC_URL = "https://drive.google.com/uc?export=download"


def parse_google_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    patterns = [
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/file/d/([A-Za-z0-9_-]+)",
        r"/d/([A-Za-z0-9_-]+)",
        r"/open\?id=([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url):
        return url
    return None


def _looks_like_html(path: Path, n: int = 4096) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    with path.open("rb") as f:
        head = f.read(n).lstrip().lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or b"<title>google drive" in head
        or b"virus scan warning" in head
        or b"text/html" in head[:500]
    )


def _looks_like_netcdf_or_hdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as f:
        head = f.read(16)
    return head.startswith(b"CDF") or head.startswith(b"\x89HDF")


def _write_stream_to_file(response: requests.Response, path: Path, chunk_size: int = 1024 * 1024) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    tmp.replace(path)


def _extract_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    text = response.text or ""
    for pattern in [
        r"confirm=([0-9A-Za-z_\-]+)",
        r'name="confirm"\s+value="([^"]+)"',
        r'"confirm"\s*:\s*"([^"]+)"',
    ]:
        match = re.search(pattern, text)
        if match:
            return html.unescape(match.group(1))
    return None


def _extract_download_form(response_text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Extrae el formulario actual de Google Drive para archivos grandes:
    <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
    """
    text = response_text or ""
    form_match = re.search(
        r'<form[^>]+id=["\']download-form["\'][^>]*>(.*?)</form>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not form_match:
        return None
    form_html = form_match.group(0)
    action_match = re.search(r'action=["\']([^"\']+)["\']', form_html, flags=re.IGNORECASE)
    if not action_match:
        return None
    action = urljoin("https://drive.google.com", html.unescape(action_match.group(1)))
    params: Dict[str, str] = {}
    for input_match in re.finditer(r'<input[^>]+>', form_html, flags=re.IGNORECASE):
        tag = input_match.group(0)
        name_match = re.search(r'name=["\']([^"\']+)["\']', tag, flags=re.IGNORECASE)
        value_match = re.search(r'value=["\']([^"\']*)["\']', tag, flags=re.IGNORECASE)
        if name_match:
            name = html.unescape(name_match.group(1))
            value = html.unescape(value_match.group(1)) if value_match else ""
            params[name] = value
    return action, params


def _diagnose_html(path: Path, max_chars: int = 1200) -> str:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return ""
    low = text.lower()
    if "too many users have viewed or downloaded this file recently" in low:
        return "Google Drive bloqueó la descarga por cuota: too many users have viewed or downloaded this file recently."
    if "download quota exceeded" in low:
        return "Google Drive bloqueó la descarga por cuota: download quota exceeded."
    if "virus scan warning" in low:
        return "Google Drive devolvió la página de advertencia de virus y no entregó el binario."
    if "sign in" in low or "login" in low:
        return "Google Drive está pidiendo inicio de sesión. El archivo no está realmente público."
    return text[:max_chars]


def _download_with_gdown(file_id: str, output_path: Path) -> bool:
    try:
        import gdown
    except Exception:
        return False
    try:
        if output_path.exists():
            output_path.unlink()
        gdown.download(
            id=file_id,
            output=str(output_path),
            quiet=False,
            fuzzy=False,
            use_cookies=False,
        )
        return _looks_like_netcdf_or_hdf(output_path) and not _looks_like_html(output_path)
    except Exception:
        return False


def _download_with_requests(file_id: str, output_path: Path) -> bool:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 hydroclimate-dashboard/3.0"}

    response = session.get(
        GOOGLE_DRIVE_UC_URL,
        params={"id": file_id},
        headers=headers,
        stream=True,
        timeout=(30, 180),
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        _write_stream_to_file(response, output_path)
        return _looks_like_netcdf_or_hdf(output_path) and not _looks_like_html(output_path)

    html_text = response.text

    token = _extract_confirm_token(response)
    if token:
        response2 = session.get(
            GOOGLE_DRIVE_UC_URL,
            params={"id": file_id, "confirm": token},
            headers=headers,
            stream=True,
            timeout=(30, 240),
        )
        response2.raise_for_status()
        _write_stream_to_file(response2, output_path)
        if _looks_like_netcdf_or_hdf(output_path) and not _looks_like_html(output_path):
            return True

    form = _extract_download_form(html_text)
    if form:
        action, params = form
        response3 = session.get(
            action,
            params=params,
            headers=headers,
            stream=True,
            timeout=(30, 300),
        )
        response3.raise_for_status()
        _write_stream_to_file(response3, output_path)
        if _looks_like_netcdf_or_hdf(output_path) and not _looks_like_html(output_path):
            return True

    output_path.write_text(html_text, encoding="utf-8", errors="ignore")
    return False


def download_google_drive_file(url: str, cache_dir: str = "/tmp/hydroclimate_dashboard_cache", filename: str = "precipitation.nc") -> str:
    file_id = parse_google_drive_file_id(url)
    if not file_id:
        raise ValueError(
            "No pude extraer el FILE_ID de Google Drive. Usa https://drive.google.com/uc?id=FILE_ID, "
            "https://drive.google.com/file/d/FILE_ID/view, o pega solo el FILE_ID."
        )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(file_id.encode("utf-8")).hexdigest()[:16]
    output_path = cache_path / f"{key}_{filename}"

    if output_path.exists() and _looks_like_netcdf_or_hdf(output_path) and not _looks_like_html(output_path):
        return str(output_path)

    if output_path.exists():
        output_path.unlink()

    if _download_with_gdown(file_id, output_path):
        return str(output_path)

    if _download_with_requests(file_id, output_path):
        return str(output_path)

    diagnosis = _diagnose_html(output_path)
    raise RuntimeError(
        "Google Drive devolvió HTML, no NetCDF. "
        "El archivo quizá no está público, Drive bloqueó la descarga por cuota, "
        "o Streamlit Cloud no pudo completar la descarga del archivo grande. "
        f"Diagnóstico: {diagnosis[:1000]}"
    )


def open_netcdf_local(path: str, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
    chunks = chunks or {"Z1": 365, "latitude": 50, "longitude": 50}
    errors = []
    for engine in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(
                path,
                engine=engine,
                chunks=chunks,
                decode_times=False,
                mask_and_scale=True,
            )
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
        variable = "precipitation"
    else:
        candidates = []
        for name, da in ds.data_vars.items():
            if name.lower() in ("crs", "spatial_ref"):
                continue
            if lat_name in da.dims and lon_name in da.dims:
                candidates.append(name)
        if not candidates:
            raise ValueError("No se encontró variable de datos con dimensiones de latitud y longitud.")
        variable = candidates[0]

    return {"variable": variable, "lat": lat_name, "lon": lon_name, "time": time_name}


def clean_precipitation_da(da: xr.DataArray) -> xr.DataArray:
    fill_value = da.attrs.get("_FillValue", None)
    if fill_value is not None:
        da = da.where(da != fill_value)
    da = da.where(da > -1e20)
    da = da.where(da >= 0)
    return da


def ensure_latitude_descending(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    values = da[lat_name].values
    if len(values) >= 2 and values[1] > values[0]:
        da = da.sortby(lat_name, ascending=False)
    return da
