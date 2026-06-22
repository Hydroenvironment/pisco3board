from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import shape
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from core.data_loader import load_dataset_from_google_drive, infer_dataset_roles, clean_precipitation_da, ensure_latitude_descending
from core.time_utils import assign_datetime_coordinate
from core.geometry import geometry_mask_dataarray, load_vector_from_zip, dissolve_geometry
from core.statistics import base_metrics, etccdi_precipitation_indices, annual_statistics, monthly_statistics, mann_kendall_approximation
from core.indices import spi_gamma_monthly, classify_spi
from core.events import detect_precipitation_events
from core.climatology import monthly_climatology, dayofyear_climatology, monthly_anomalies
from core.clustering import spatial_kmeans_from_mean_field

st.set_page_config(page_title="Hydroclimate Scientific Dashboard", page_icon="🌧️", layout="wide", initial_sidebar_state="expanded")
st.title("🌧️ Hydroclimate Scientific Dashboard")
st.caption("Sistema avanzado para análisis de precipitación diaria gridded en NetCDF: punto, polígono, shapefile, climatología, anomalías, SPI, extremos, eventos y mapas.")

with st.sidebar:
    st.header("1. Fuente de datos")
    url = st.text_input("NetCDF URL de Google Drive", value="", placeholder="https://drive.google.com/uc?id=FILE_ID")
    start_date = st.text_input("Fecha inicial asociada a Z1", value="1981-01-01")
    st.divider()
    st.header("2. Opciones de análisis")
    analysis_mode = st.selectbox("Modo principal", ["Punto por coordenadas o clic en mapa", "Polígono dibujado en mapa", "Shapefile / GeoPackage / GeoJSON", "Mapa espacial de un día", "Clustering espacial"])
    wet_threshold = st.number_input("Umbral día húmedo (mm/día)", value=1.0, min_value=0.0, step=0.5)
    heavy_threshold = st.number_input("Umbral lluvia intensa (mm/día)", value=10.0, min_value=0.0, step=1.0)
    very_heavy_threshold = st.number_input("Umbral lluvia muy intensa (mm/día)", value=20.0, min_value=0.0, step=1.0)
    extreme_threshold = st.number_input("Umbral lluvia extrema (mm/día)", value=50.0, min_value=0.0, step=5.0)
    st.divider()
    st.header("3. Rendimiento")
    full_period = st.checkbox("Usar periodo completo", value=False, help="Para Streamlit Cloud se recomienda iniciar con una ventana temporal reducida.")
    if not full_period:
        analysis_start = st.date_input("Inicio análisis", value=pd.to_datetime("2015-01-01"))
        analysis_end = st.date_input("Fin análisis", value=pd.to_datetime("2020-12-31"))
    else:
        analysis_start = None
        analysis_end = None
    area_weighted_polygon = st.checkbox("Promedio de polígono ponderado por cos(lat)", value=True)

if not url:
    st.info("Pega el enlace público de Google Drive del NetCDF en la barra lateral para iniciar.")
    st.stop()

@st.cache_resource(show_spinner=False)
def cached_load_dataset(url_value: str):
    return load_dataset_from_google_drive(url_value)

try:
    with st.spinner("Descargando o leyendo NetCDF desde cache local temporal. El primer arranque puede tardar varios minutos para archivos grandes."):
        ds, local_path = cached_load_dataset(url)
except Exception as exc:
    st.error("No se pudo descargar o abrir el NetCDF desde Google Drive.")
    st.exception(exc)
    st.markdown("""
**Acciones correctivas:**

1. Verifica que el archivo de Google Drive esté público: `Anyone with the link`.
2. Usa un enlace con formato `https://drive.google.com/uc?id=FILE_ID`.
3. Confirma que Google Drive no haya bloqueado temporalmente la descarga por exceso de tráfico.
4. Reinicia la app después de cambiar permisos del archivo.
""")
    st.stop()

roles = infer_dataset_roles(ds)
var_name = roles["variable"]
lat_name = roles["lat"]
lon_name = roles["lon"]
time_name = roles["time"]

da = ds[var_name]
da = clean_precipitation_da(da)
da = ensure_latitude_descending(da, lat_name=lat_name)
da = assign_datetime_coordinate(da, time_dim=time_name, start_date=start_date)
time_dim = "time" if "time" in da.dims else time_name
lat_values = da[lat_name].values
lon_values = da[lon_name].values
time_values = pd.to_datetime(da[time_dim].values) if time_dim in da.dims else None

with st.expander("Diagnóstico del NetCDF cargado", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variable", var_name)
    c2.metric("Dimensión temporal", time_dim)
    c3.metric("Latitudes", len(lat_values))
    c4.metric("Longitudes", len(lon_values))
    st.write("Ruta local temporal:", local_path)
    st.write("Dimensiones:", dict(da.sizes))
    st.write("Rango latitud:", float(np.nanmin(lat_values)), "a", float(np.nanmax(lat_values)))
    st.write("Rango longitud:", float(np.nanmin(lon_values)), "a", float(np.nanmax(lon_values)))
    if time_values is not None:
        st.write("Rango temporal:", str(time_values.min())[:10], "a", str(time_values.max())[:10])


def restrict_da_time(input_da):
    if full_period or time_dim not in input_da.dims:
        return input_da
    return input_da.sel({time_dim: slice(pd.to_datetime(analysis_start), pd.to_datetime(analysis_end))})


def extract_point_series(latitude: float, longitude: float) -> pd.DataFrame:
    sub = restrict_da_time(da)
    point_da = sub.sel({lat_name: latitude, lon_name: longitude}, method="nearest")
    values = point_da.compute().values.astype(float)
    return pd.DataFrame({"time": pd.to_datetime(point_da[time_dim].values), "precipitation": values, "nearest_latitude": float(point_da[lat_name].values), "nearest_longitude": float(point_da[lon_name].values)})


def extract_polygon_series(geometry) -> pd.DataFrame:
    sub = restrict_da_time(da)
    mask = geometry_mask_dataarray(geometry, lat_values=sub[lat_name].values, lon_values=sub[lon_name].values, lat_name=lat_name, lon_name=lon_name)
    masked = sub.where(mask)
    if area_weighted_polygon:
        weights_1d = np.cos(np.deg2rad(masked[lat_name]))
        ts = masked.weighted(weights_1d).mean(dim=(lat_name, lon_name), skipna=True)
    else:
        ts = masked.mean(dim=(lat_name, lon_name), skipna=True)
    values = ts.compute().values.astype(float)
    df = pd.DataFrame({"time": pd.to_datetime(ts[time_dim].values), "precipitation": values})
    df["polygon_cells"] = int(mask.sum().values)
    return df


def make_timeseries_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["precipitation"], mode="lines", name="Precipitación diaria", hovertemplate="%{x|%Y-%m-%d}<br>Precipitación: %{y:.2f} mm<extra></extra>"))
    fig.update_layout(title="Serie temporal diaria de precipitación", xaxis_title="Fecha", yaxis_title="Precipitación diaria (mm)", height=520, hovermode="x unified", legend_title_text="Serie", margin=dict(l=40, r=20, t=70, b=40))
    return fig


def make_cumulative_figure(df: pd.DataFrame) -> go.Figure:
    out = df.copy()
    out["cumulative_precipitation"] = out["precipitation"].fillna(0).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=out["time"], y=out["cumulative_precipitation"], mode="lines", name="Acumulado"))
    fig.update_layout(title="Precipitación acumulada", xaxis_title="Fecha", yaxis_title="Precipitación acumulada (mm)", height=480, hovermode="x unified")
    return fig


def render_full_analysis(df: pd.DataFrame, label: str):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    st.success(f"Análisis ejecutado: {label}")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Serie temporal", "Métricas", "Extremos", "Climatología", "Anomalías", "SPI", "Eventos", "Tablas y descarga"])

    with tab1:
        st.plotly_chart(make_timeseries_figure(df), use_container_width=True)
        st.plotly_chart(make_cumulative_figure(df), use_container_width=True)
        rolling_window = st.slider("Ventana media móvil (días)", min_value=3, max_value=365, value=30, step=1)
        rolling = df[["time", "precipitation"]].copy()
        rolling["rolling_mean"] = rolling["precipitation"].rolling(rolling_window, min_periods=max(1, rolling_window // 3)).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling["time"], y=rolling["precipitation"], mode="lines", name="Diaria", opacity=0.35))
        fig.add_trace(go.Scatter(x=rolling["time"], y=rolling["rolling_mean"], mode="lines", name=f"Media móvil {rolling_window} días"))
        fig.update_layout(title="Serie diaria con suavizado", xaxis_title="Fecha", yaxis_title="Precipitación (mm)", height=520, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        metrics_df = base_metrics(df, wet_threshold=wet_threshold)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        annual = annual_statistics(df)
        trend = mann_kendall_approximation(annual["total"].to_numpy())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total analizado (mm)", f"{float(df['precipitation'].sum(skipna=True)):,.2f}")
        c2.metric("Media diaria (mm/día)", f"{float(df['precipitation'].mean(skipna=True)):,.2f}")
        c3.metric("Máximo diario (mm)", f"{float(df['precipitation'].max(skipna=True)):,.2f}")
        c4.metric("Observaciones válidas", f"{int(df['precipitation'].notna().sum()):,}")
        st.subheader("Tendencia aproximada sobre totales anuales")
        st.json(trend)
        fig = px.bar(annual, x="year", y="total", title="Precipitación total anual", labels={"year": "Año", "total": "Precipitación total anual (mm)"})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(annual, use_container_width=True, hide_index=True)

    with tab3:
        ext_df = etccdi_precipitation_indices(df, wet_threshold=wet_threshold, heavy_threshold=heavy_threshold, very_heavy_threshold=very_heavy_threshold, extreme_threshold=extreme_threshold)
        st.dataframe(ext_df, use_container_width=True, hide_index=True)
        fig = px.bar(ext_df, x="index", y="value", hover_data=["unit", "description"], title="Índices extremos de precipitación", labels={"index": "Índice", "value": "Valor"})
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        monthly = monthly_statistics(df)
        heat = monthly.pivot(index="year", columns="month", values="total")
        fig = px.imshow(heat, aspect="auto", title="Mapa calendario de precipitación mensual acumulada", labels=dict(x="Mes", y="Año", color="mm"))
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        base_start = st.date_input("Inicio periodo base climatológico", value=pd.to_datetime("1991-01-01"), key=f"{label}_clim_start")
        base_end = st.date_input("Fin periodo base climatológico", value=pd.to_datetime("2020-12-31"), key=f"{label}_clim_end")
        monthly_clim = monthly_climatology(df, base_start=base_start, base_end=base_end)
        doy_clim = dayofyear_climatology(df, base_start=base_start, base_end=base_end)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_clim["month_name"], y=monthly_clim["total_mean"], name="Total mensual medio"))
        fig.update_layout(title="Climatología mensual de precipitación acumulada", xaxis_title="Mes", yaxis_title="Precipitación mensual media (mm)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=doy_clim["dayofyear"], y=doy_clim["mean"], mode="lines", name="Media diaria"))
        fig.add_trace(go.Scatter(x=doy_clim["dayofyear"], y=doy_clim["p90"], mode="lines", name="P90 diario"))
        fig.add_trace(go.Scatter(x=doy_clim["dayofyear"], y=doy_clim["p10"], mode="lines", name="P10 diario"))
        fig.update_layout(title="Ciclo anual climatológico diario", xaxis_title="Día del año", yaxis_title="Precipitación diaria (mm)", height=520, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(monthly_clim, use_container_width=True, hide_index=True)
        st.dataframe(doy_clim, use_container_width=True, hide_index=True)

    with tab5:
        base_start_anom = st.date_input("Inicio periodo base para anomalías", value=pd.to_datetime("1991-01-01"), key=f"{label}_anom_start")
        base_end_anom = st.date_input("Fin periodo base para anomalías", value=pd.to_datetime("2020-12-31"), key=f"{label}_anom_end")
        anom = monthly_anomalies(df, base_start=base_start_anom, base_end=base_end_anom)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=anom["date"], y=anom["anomaly_mm"], name="Anomalía mensual"))
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(title="Anomalías mensuales de precipitación respecto al periodo base", xaxis_title="Fecha", yaxis_title="Anomalía mensual (mm)", height=520, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anom["date"], y=anom["standardized_anomaly"], mode="lines", name="Anomalía estandarizada"))
        fig.add_hline(y=0, line_width=1)
        fig.add_hline(y=1, line_dash="dash")
        fig.add_hline(y=-1, line_dash="dash")
        fig.update_layout(title="Anomalías mensuales estandarizadas", xaxis_title="Fecha", yaxis_title="Z-score", height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(anom, use_container_width=True, hide_index=True)

    with tab6:
        spi_scale = st.selectbox("Escala SPI mensual", [1, 3, 6, 12, 24], index=1)
        spi_df = spi_gamma_monthly(df, scale_months=spi_scale)
        spi_col = f"SPI_{spi_scale}"
        spi_df["category"] = spi_df[spi_col].apply(classify_spi)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=spi_df["month"], y=spi_df[spi_col], name=spi_col))
        fig.add_hline(y=0, line_width=1)
        for y in [-2, -1.5, -1, 1, 1.5, 2]:
            fig.add_hline(y=y, line_dash="dash")
        fig.update_layout(title=f"Índice Estandarizado de Precipitación SPI-{spi_scale}", xaxis_title="Mes", yaxis_title=f"SPI-{spi_scale}", height=540)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(spi_df, use_container_width=True, hide_index=True)
        st.info("El SPI se calcula con agregación mensual y ajuste gamma por mes calendario. Para uso operacional debe validarse contra un paquete hidrológico especializado y un periodo base suficientemente largo.")

    with tab7:
        event_threshold = st.number_input("Umbral para segmentar eventos (mm/día)", value=heavy_threshold, min_value=0.0, step=1.0)
        min_duration = st.number_input("Duración mínima del evento (días)", value=1, min_value=1, step=1)
        max_gap = st.number_input("Brecha máxima permitida dentro del evento (días)", value=0, min_value=0, step=1)
        events = detect_precipitation_events(df, threshold=event_threshold, min_duration_days=int(min_duration), max_gap_days=int(max_gap))
        st.dataframe(events, use_container_width=True, hide_index=True)
        if not events.empty:
            fig = px.bar(events, x="event_id", y="event_total_mm", hover_data=["start", "end", "duration_days", "event_max_1day_mm"], title="Totales por evento de precipitación", labels={"event_id": "ID de evento", "event_total_mm": "Total del evento (mm)"})
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

    with tab8:
        st.subheader("Serie diaria extraída")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Descargar serie diaria CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="precipitation_timeseries.csv", mime="text/csv")
        st.download_button("Descargar estadísticas anuales CSV", data=annual_statistics(df).to_csv(index=False).encode("utf-8"), file_name="annual_precipitation_statistics.csv", mime="text/csv")

st.subheader("Mapa interactivo de selección")
map_center = [float(np.nanmean(lat_values)), float(np.nanmean(lon_values))]
m = folium.Map(location=map_center, zoom_start=5, tiles="OpenStreetMap")
bounds = [[float(np.nanmin(lat_values)), float(np.nanmin(lon_values))], [float(np.nanmax(lat_values)), float(np.nanmax(lon_values))]]
folium.Rectangle(bounds=bounds, tooltip="Extensión del NetCDF", fill=False).add_to(m)
Draw(export=True, draw_options={"polyline": False, "circle": False, "circlemarker": False, "marker": True, "polygon": True, "rectangle": True}).add_to(m)
map_data = st_folium(m, height=560, use_container_width=True)

if analysis_mode == "Punto por coordenadas o clic en mapa":
    st.header("Análisis por punto")
    default_lat = float(np.nanmean(lat_values))
    default_lon = float(np.nanmean(lon_values))
    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        default_lat = float(clicked["lat"])
        default_lon = float(clicked["lng"])
        st.info(f"Coordenada seleccionada desde el mapa: lat={default_lat:.5f}, lon={default_lon:.5f}")
    c1, c2 = st.columns(2)
    selected_lat = c1.number_input("Latitud", value=default_lat, format="%.6f")
    selected_lon = c2.number_input("Longitud", value=default_lon, format="%.6f")
    if st.button("Ejecutar análisis por punto", type="primary"):
        try:
            render_full_analysis(extract_point_series(selected_lat, selected_lon), label="point")
        except Exception as exc:
            st.error("Falló la extracción por punto.")
            st.exception(exc)

elif analysis_mode == "Polígono dibujado en mapa":
    st.header("Análisis por polígono dibujado")
    drawing = map_data.get("last_active_drawing") if map_data else None
    if drawing is None:
        st.info("Dibuja un polígono o rectángulo sobre el mapa y luego ejecuta el análisis.")
    else:
        geom = shape(drawing["geometry"])
        st.write("Tipo de geometría:", geom.geom_type)
        st.write("Área aproximada en grados cuadrados:", geom.area)
        if st.button("Ejecutar análisis por polígono dibujado", type="primary"):
            try:
                render_full_analysis(extract_polygon_series(geom), label="drawn_polygon")
            except Exception as exc:
                st.error("Falló la extracción por polígono.")
                st.exception(exc)

elif analysis_mode == "Shapefile / GeoPackage / GeoJSON":
    st.header("Análisis por archivo vectorial")
    uploaded = st.file_uploader("Sube un ZIP con shapefile, GeoPackage o GeoJSON", type=["zip"])
    if uploaded is not None:
        try:
            gdf = load_vector_from_zip(uploaded)
            st.success(f"Archivo vectorial cargado: {len(gdf)} geometría(s).")
            st.dataframe(gdf.drop(columns="geometry", errors="ignore"), use_container_width=True)
            selected_index = st.selectbox("Geometría a analizar", options=["Todas disueltas"] + list(range(len(gdf))))
            geom = dissolve_geometry(gdf) if selected_index == "Todas disueltas" else gdf.geometry.iloc[int(selected_index)]
            if st.button("Ejecutar análisis por shapefile", type="primary"):
                render_full_analysis(extract_polygon_series(geom), label="uploaded_vector")
        except Exception as exc:
            st.error("No se pudo leer o analizar el archivo vectorial.")
            st.exception(exc)

elif analysis_mode == "Mapa espacial de un día":
    st.header("Mapa espacial de precipitación diaria")
    selected_date = st.date_input("Fecha del mapa", value=pd.to_datetime(time_values[0]), min_value=pd.to_datetime(time_values.min()), max_value=pd.to_datetime(time_values.max()))
    if st.button("Generar mapa espacial", type="primary"):
        try:
            day = da.sel({time_dim: pd.to_datetime(selected_date)}, method="nearest").compute()
            fig = px.imshow(day.values.astype(float), x=lon_values, y=lat_values, origin="upper", aspect="auto", labels=dict(x="Longitud", y="Latitud", color="mm/día"), title=f"Precipitación diaria espacial: {pd.to_datetime(day[time_dim].values).date()}")
            fig.update_layout(height=720)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error("Falló la generación del mapa espacial.")
            st.exception(exc)

elif analysis_mode == "Clustering espacial":
    st.header("Clustering espacial exploratorio")
    st.warning("Este módulo calcula la precipitación media temporal por celda y aplica KMeans. Puede tardar porque usa todo el dominio espacial y la ventana temporal seleccionada.")
    n_clusters = st.slider("Número de clusters", min_value=2, max_value=12, value=5, step=1)
    if st.button("Ejecutar clustering espacial", type="primary"):
        try:
            sub = restrict_da_time(da)
            cluster_da = spatial_kmeans_from_mean_field(sub, time_name=time_dim, lat_name=lat_name, lon_name=lon_name, n_clusters=n_clusters)
            fig = px.imshow(cluster_da.values, x=cluster_da[lon_name].values, y=cluster_da[lat_name].values, origin="upper", aspect="auto", labels=dict(x="Longitud", y="Latitud", color="Cluster"), title="Clusters espaciales basados en precipitación media")
            fig.update_layout(height=720)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error("Falló el clustering espacial.")
            st.exception(exc)
