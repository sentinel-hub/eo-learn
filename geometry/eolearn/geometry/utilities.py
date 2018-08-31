"""
Module containing various geometrical tasks
"""

import numpy as np
from rasterio import features, transform
from shapely.geometry import Polygon

from eolearn.core import EOTask


class VectorToRaster(EOTask):
    """
    Task burns into one of the EOPatch's features geo-referenced shapes given in provided Geopandas DataFrame.

    :param feature: A tuple of feature type and feature name, e.g. (FeatureType.MASK, 'cloud_mask')
    :type feature: (FeatureType, str)
    :param vector_data: Vector data
    :type vector_data: geopandas.GeoDataFrame
    :param raster_value: Value of raster pixels which are contained inside of vector polygons
    :type raster_value: int or float
    :param raster_shape: Can be a tuple in form of (height, width) of an existing feature from which the shape will be
    taken e.g. (FeatureType.MASK, 'IS_DATA')
    :type raster_shape: (int, int) or (FeatureType, str)
    :param raster_dtype: `numpy` data type of the obtained raster array
    :type raster_dtype: numpy.dtype
    :param no_data_value: Value of raster pixels which are outside of vector polygons
    :type no_data_value: int or float
    """
    def __init__(self, feature, vector_data, raster_value, raster_shape, raster_dtype=np.uint8, no_data_value=0):
        self.feature_type, self.feature_name = next(iter(self._parse_features(feature)))
        self.vector_data = vector_data
        self.raster_value = raster_value
        self.raster_shape = raster_shape
        self.raster_dtype = raster_dtype
        self.no_data_value = no_data_value

    def _get_submap(self, eopatch):
        """
        Returns a new geopandas dataframe with same structure as original one (columns) except that
        it contains only polygons that are contained within the given bbox.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: New EOPatch
        :rtype: EOPatch
        """
        bbox = Polygon(eopatch.bbox.get_polygon())

        filtered_data = self.vector_data[self.vector_data.geometry.intersects(bbox)].copy(deep=True)
        filtered_data.geometry = filtered_data.geometry.intersection(bbox)

        return filtered_data

    def _get_shape(self, eopatch):
        if isinstance(self.raster_shape, (tuple, list)) and len(self.raster_shape) == 2:
            if isinstance(self.raster_shape[0], int) and isinstance(self.raster_shape[1], int):
                return self.raster_shape

            feature_type, feature_name = next(self._parse_features(self.raster_shape)(eopatch))
            return eopatch.get_spatial_dimension(feature_type, feature_name)

        raise ValueError('Could not determine shape of the raster image')

    def execute(self, eopatch):
        """ Execute function which adds new vector layer to the EOPatch

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: New EOPatch with added vector layer
        :rtype: EOPatch
        """
        bbox_map = self._get_submap(eopatch)
        height, width = self._get_shape(eopatch)
        dst_transform = transform.from_bounds(*eopatch.bbox, width=width, height=height)

        if self.feature_name in eopatch[self.feature_type]:
            raster = eopatch[self.feature_type][self.feature_name].squeeze()
        else:
            raster = np.ones((height, width), dtype=self.raster_dtype) * self.no_data_value

        if not bbox_map.empty:
            features.rasterize([(bbox_map.cascaded_union.buffer(0), self.raster_value)], out=raster,
                               transform=dst_transform, dtype=self.raster_dtype)

        eopatch[self.feature_type][self.feature_name] = raster[..., np.newaxis]

        return eopatch
