import rasterio.features as rfeatures
import numpy as np

def raster_mask(geom, shape):
    mask = rfeatures.geometry_mask(
        [geom],
        out_shape=shape,
        invert=True
    )
    return mask
