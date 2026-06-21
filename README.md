# Hydroclimate Scientific Dashboard v2

Sistema avanzado en Streamlit para análisis hidroclimático de precipitación diaria gridded en NetCDF.

## Dataset esperado

El sistema está preparado para un NetCDF con estructura equivalente a:

- variable principal: `precipitation`
- dimensiones: `(Z1, latitude, longitude)`
- coordenadas espaciales: `latitude`, `longitude`
- sistema de referencia: EPSG:4326
- precipitación diaria en milímetros

También permite detectar nombres alternativos de variables de tiempo, latitud y longitud.

## Corrección importante para Google Drive

`xarray.open_dataset()` no puede abrir de forma robusta un enlace de Google Drive como si fuera un archivo NetCDF remoto.
Por eso esta versión descarga primero el archivo desde Google Drive hacia el almacenamiento temporal de la instancia Streamlit y luego lo abre localmente.

Formato recomendado de enlace:

```text
https://drive.google.com/uc?id=FILE_ID
```

También acepta enlaces del tipo:

```text
https://drive.google.com/file/d/FILE_ID/view
```

El archivo de Google Drive debe estar público:

- General access: Anyone with the link
- Role: Viewer

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicación gratuita en Streamlit Community Cloud

1. Crea un repositorio en GitHub.
2. Sube todos los archivos de esta carpeta.
3. Entra a Streamlit Community Cloud.
4. Crea una nueva app.
5. Selecciona tu repositorio.
6. Archivo principal: `app.py`.
7. Haz deploy.
8. Dentro del dashboard pega el enlace público de Google Drive.

## Limitaciones operativas

Un NetCDF de aproximadamente 1.42 GB puede funcionar en Streamlit Cloud, pero el primer arranque puede tardar porque el archivo debe descargarse. La instancia gratuita tiene límites de memoria, CPU, almacenamiento temporal y tiempo de inactividad. Para uso público de alta concurrencia se requiere infraestructura persistente o preprocesamiento a productos derivados.

## Recomendación práctica

Para uso público sin costo:

- mantener el NetCDF como Google Drive público;
- usar esta versión con descarga cacheada;
- limitar consultas espaciales muy grandes;
- usar ventanas temporales en vez de todo el periodo para análisis pesados;
- precomputar productos climatológicos si se espera mucho tráfico.
