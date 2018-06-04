"""
Module containing various geometrical tasks
"""

import numpy as np
from rasterio import features, transform
from shapely.geometry import Polygon

from eolearn.core import EOTask, FeatureType


class VectorToRaster(EOTask):
    """
    Task burns into one of the EOPatch's features geo-referenced shapes given in provided Geopandas DataFrame.

    :param feature_type: Type of the vector feature which will be added to EOPatch
    :type feature_type: eolearn.core.FeatureType
    :param feature_name: Name of the vector feature which will be added to EOPatch
    :type feature_name: str
    :param vector_data: Vector data
    :type vector_data: geopandas.GeoDataFrame
    :param raster_value: Value of raster pixels which are contained inside of vector polygons
    :type raster_value: int or float
    :param raster_dtype: `numpy` data type of the obtained raster array
    :type raster_dtype: numpy.dtype
    :param no_data_value: Value of raster pixels which are outside of vector polygons
    :type no_data_value: int or float
    """
    def __init__(self, feature_type, feature_name, vector_data, raster_value, raster_dtype=np.uint8, no_data_value=0):
        self.feature_type = feature_type
        self.feature_name = feature_name
        self.vector_data = vector_data
        self.raster_value = raster_value
        self.raster_dtype = raster_dtype
        self.no_data_value = no_data_value

    def _get_submap(self, eopatch):
        """
        Returns a new geopandas dataframe with same structure as original one (columns) except that
        it contains only polygons that are contained within the given bbox.

        :param eopatch: input EOPatch
        :type eopatch: eolearn.core.EOPatch
        :return: New EOPatch
        :rtype: eolearn.core.EOPatch
        """
        bbox = Polygon(eopatch.bbox.get_polygon())
        orig_idxs = self.vector_data.index[self.vector_data.geometry.intersects(bbox)]
        copy = self.vector_data.iloc[orig_idxs].copy(deep=True)
        copy.geometry = copy.loc[orig_idxs].intersection(bbox)

        return copy

    def execute(self, eopatch):
        """ Execute function which adds new vector layer to the EOPatch

        :param eopatch: input EOPatch
        :type eopatch: eolearn.core.EOPatch
        :return: New EOPatch with added vector layer
        :rtype: eolearn.core.EOPatch
        """
        bbox_map = self._get_submap(eopatch)

        data_arr = eopatch.get_feature(FeatureType.MASK, 'IS_DATA')

        dst_shape = data_arr.shape
        dst_transform = transform.from_bounds(*eopatch.bbox, width=dst_shape[2], height=dst_shape[1])

        if eopatch.feature_exists(self.feature_type, self.feature_name):
            raster = eopatch.get_feature(self.feature_type, self.feature_name)
        else:
            raster = np.ones(dst_shape[1:3], dtype=self.raster_dtype) * self.no_data_value

        if bbox_map:
            features.rasterize([(bbox_map.cascaded_union.buffer(0), self.raster_value)], out=raster,
                               transform=dst_transform, dtype=self.raster_dtype)

        eopatch.add_feature(self.feature_type, self.feature_name, raster[..., np.newaxis])

        return eopatch
