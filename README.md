# Hydroclimate Scientific Dashboard v3

Versión corregida para NetCDF grandes alojados en Google Drive.

## Corrección principal

Google Drive devuelve una página HTML de advertencia de análisis de virus para archivos grandes. Esta versión no intenta abrir directamente el URL con `xarray.open_dataset(url)`. Primero descarga el archivo a `/tmp/hydroclimate_dashboard_cache`, resuelve el aviso de archivo grande usando `gdown`, token de confirmación o formulario `download-form`, y recién después abre el NetCDF localmente con `xarray`.

## Enlace recomendado

```text
https://drive.google.com/uc?id=FILE_ID
```

También acepta:

```text
https://drive.google.com/file/d/FILE_ID/view
FILE_ID
```

Para tu archivo:

```text
https://drive.google.com/uc?id=18ImwlJFRHvdwnK4f45lPIi5cOi4iISup
```

## Permisos en Google Drive

- General access: Anyone with the link
- Role: Viewer

## Publicación en Streamlit Community Cloud

1. Descomprime esta carpeta.
2. Reemplaza todos los archivos del repositorio anterior.
3. Haz `git add .`, `git commit`, `git push`.
4. En Streamlit Cloud usa `Manage app` -> `Reboot app`.
5. Pega el enlace de Google Drive en la barra lateral.

## Limitaciones

Si Google Drive responde cuota excedida o si Streamlit Cloud no tiene suficiente almacenamiento temporal para descargar el NetCDF, no existe una corrección de código que lo evite en una cuenta gratuita. En ese caso se debe duplicar el archivo en otro Drive, reducir el NetCDF, o preprocesar productos derivados.
