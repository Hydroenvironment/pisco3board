import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import shape
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

from core.indices import hydro_metrics, extreme_indices, spi_like
from core.climatology import climatology
from core.geometry import raster_mask
from core.data_loader import load_dataset

st.set_page_config(layout='wide')
st.title('Hydroclimate Scientific Dashboard')

url = st.sidebar.text_input('NetCDF URL (Google Drive direct link)')

if not url:
    st.stop()

ds = load_dataset(url)

p = ds['precipitation']
lat = ds['latitude'].values
lon = ds['longitude'].values
time = np.arange(ds.dims['Z1'])

mode = st.sidebar.selectbox(
    'Mode',
    ['Point', 'Polygon', 'Climatology', 'Extremes', 'SPI']
)

m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=5)
Draw(export=True).add_to(m)
map_data = st_folium(m, height=500)

df = None

if mode == 'Point':
    la = st.sidebar.number_input('lat', float(lat.mean()))
    lo = st.sidebar.number_input('lon', float(lon.mean()))

    if st.button('Run'):
        ts = p.sel(latitude=la, longitude=lo, method='nearest').values
        df = pd.DataFrame({'t': time, 'p': ts})

elif mode == 'Polygon':
    if map_data and map_data.get('last_active_drawing'):
        geom = shape(map_data['last_active_drawing']['geometry'])

        mask = raster_mask(geom, (len(lat), len(lon)))
        ts = p.where(mask).mean(('latitude', 'longitude')).values
        df = pd.DataFrame({'t': time, 'p': ts})

elif mode == 'Climatology':
    ts = p[:,0,0].values
    df = pd.DataFrame({'clim': climatology(ts)})

elif mode == 'Extremes':
    ts = p[:,0,0].values
    df = pd.DataFrame([extreme_indices(ts)])

elif mode == 'SPI':
    ts = p[:,0,0].values
    df = pd.DataFrame({'spi': spi_like(ts)})

if df is not None and 'p' in df.columns:
    st.line_chart(df.set_index('t'))

    st.subheader('Metrics')
    st.json(hydro_metrics(df['p'].values))

    st.subheader('Extremes')
    st.json(extreme_indices(df['p'].values))

    st.subheader('SPI')
    st.line_chart(spi_like(df['p'].values))
