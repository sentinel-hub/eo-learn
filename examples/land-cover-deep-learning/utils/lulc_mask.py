"""
Read in land use and land cover (LULC) mask, and rasterized the shapefile

lulc_mask.py

Script adopted from eo-learn land-cover-example

autohr: @developmentseed
"""
import numpy as np
import geopandas as gpd
from eolearn.core import FeatureType

from eolearn.geometry import VectorToRaster

def lulc_arr(lulc_mask, lulc_codes):
    """read in dataset LULC mask and the associated codes and rasterized the mask into raster"""
    land_cover = gpd.read_file(lulc_mask)
    land_cover_array = []
    for val in lulc_codes:
        temp = land_cover[land_cover.lulcid == val]
        temp.reset_index(drop=True, inplace=True)
        land_cover_array.append(temp)
        del temp

    rshape = (FeatureType.MASK, 'IS_VALID')

    land_cover_task_array = []
    for el, val in zip(land_cover_array, lulc_codes):
        land_cover_task_array.append(VectorToRaster(
            feature=(FeatureType.MASK_TIMELESS, 'LULC'),
            vector_data=el,
            raster_value=val,
            raster_shape=rshape,
            raster_dtype=np.uint8))

    return land_cover_task_array
