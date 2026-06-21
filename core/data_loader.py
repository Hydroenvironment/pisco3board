import xarray as xr

def load_dataset(url):
    ds = xr.open_dataset(url, chunks={'Z1': 1000})
    return ds
