"""
Transformations between vector and raster formats of data

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
import shapely.geometry
import shapely.ops
import shapely.wkt
from geopandas import GeoSeries, GeoDataFrame

from sentinelhub import CRS, bbox_to_dimensions
from eolearn.core import EOTask, FeatureType, FeatureTypeSet

LOGGER = logging.getLogger(__name__)


class VectorToRaster(EOTask):
    """ Task that transforms vector data into a raster feature.

    Vector data can be given as an EOPatch feature or as an independent geopandas `GeoDataFrame`.

    In the background it uses `rasterio.features.rasterize`, documented at
    https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.rasterize

    Note: Rasterization of time-dependent vector features is not yet supported.
    """
    def __init__(self, vector_input, raster_feature, *, values=None, values_column=None, raster_shape=None,
                 raster_resolution=None, raster_dtype=np.uint8, no_data_value=0, write_to_existing=False,
                 overlap_value=None, buffer=0, **rasterio_params):
        """
        :param vector_input: Vector data to be used for rasterization. It can be given as a feature in `EOPatch` or
            as an independent geopandas `GeoDataFrame`.
        :type vector_input: (FeatureType, str) or GeoDataFrame
        :param raster_feature: New or existing raster feature into which data will be written. If existing raster
            raster feature is given it will by default take existing values and write over them.
        :type raster_feature: (FeatureType, str)
        :param values: If `values_column` parameter is specified then only polygons which have one of these specified
            values in `values_column` will be rasterized. It can be also left to `None`. If `values_column` parameter
            is not specified `values` parameter has to be a single number into which everything will be rasterized.
        :type values: list(int or float) or int or float or None
        :param values_column: A column in gived dataframe where values, into which polygons will be rasterized,
            are stored. If it is left to `None` then `values` parameter should be a single number into which
            everything will be rasterized.
        :type values_column: str or None
        :param raster_shape: Can be a tuple in form of (height, width) or an existing feature from which the spatial
            dimensions will be taken. Parameter `raster_resolution` can be specified instead of `raster_shape`.
        :type raster_shape: (int, int) or (FeatureType, str) or None
        :param raster_resolution: Resolution of raster into which data will be rasterized. Has to be given as a
            number or a tuple of numbers in meters. Parameter `raster_shape` can be specified instead or
            `raster_resolution`.
        :type raster_resolution: float or (float, float) or None
        :param raster_dtype: Data type of the obtained raster array, default is `numpy.uint8`.
        :type raster_dtype: numpy.dtype
        :param no_data_value: Default value of all other raster pixels into which no value will be rasterized.
        :type no_data_value: int or float
        :param write_to_existing: If `True` it will write to existing raster array and overwrite parts of its values.
            If `False` it will create a new raster array and remove the old one. Default is `False`.
        :param overlap_value: A value to override parts of raster where polygons of different classes overlap. If None,
            rasterization overlays polygon as it is the default behavior of `rasterio.features.rasterize`.
        :type overlap_value: raster's dtype
        :type write_to_existing: bool
        :param buffer: Buffer value passed to vector_data.buffer() before rasterization. If 0, no buffering is done.
        :type buffer: float
        :param: rasterio_params: Additional parameters to be passed to `rasterio.features.rasterize`. Currently
            available parameters are `all_touched` and `merge_alg`
        """

        self.vector_input, self.raster_feature = self._parse_main_params(vector_input, raster_feature)

        self.values = values
        self.values_column = values_column
        if values_column is None and (values is None or not isinstance(values, (int, float))):
            raise ValueError("One of parameters 'values' and 'values_column' is missing")

        self.raster_shape = raster_shape
        self.raster_resolution = raster_resolution
        if (raster_shape is None) == (raster_resolution is None):
            raise ValueError("Exactly one of parameters 'raster_shape' and 'raster_resolution' has to be specified")

        self.raster_dtype = raster_dtype
        self.no_data_value = no_data_value
        self.write_to_existing = write_to_existing
        self.rasterio_params = rasterio_params
        self.overlap_value = overlap_value
        self.buffer = buffer

    @staticmethod
    def _parse_main_params(vector_input, raster_feature):
        """ Parsing first 2 task parameters - what vector data will be used and in which raster feature it will be saved
        """
        if VectorToRaster._is_geopandas_object(raster_feature):
            # pylint: disable=W1114
            warnings.warn('In the new version of VectorToRaster task order of parameters changed. Parameter for '
                          'specifying vector data or feature has to be before parameter specifying new raster feature',
                          DeprecationWarning, stacklevel=3)
            return VectorToRaster._parse_main_params(raster_feature, vector_input)

        if not VectorToRaster._is_geopandas_object(vector_input):
            vector_input = VectorToRaster._parse_features(vector_input,
                                                          allowed_feature_types={FeatureType.VECTOR_TIMELESS})
        raster_feature = next(iter(
            VectorToRaster._parse_features(raster_feature, allowed_feature_types=FeatureTypeSet.RASTER_TYPES_3D)
        ))

        return vector_input, raster_feature

    @staticmethod
    def _is_geopandas_object(data):
        """ A frequently used check if object is geopandas `GeoDataFrame` or `GeoSeries`
        """
        return isinstance(data, (GeoDataFrame, GeoSeries))

    def _get_vector_data(self, eopatch):
        """ Provides vector data which will be rasterized
        """
        if self._is_geopandas_object(self.vector_input):
            return self.vector_input

        feature_type, feature_name = next(self.vector_input(eopatch))
        return eopatch[feature_type][feature_name]

    def _get_rasterization_shapes(self, eopatch, group_classes=False):
        """ Returns a generator of pairs of geometrical shapes and values. In rasterization process each shape will be
        rasterized to it's corresponding value.
        If there are no such geometries it will return `None`
        :param group_classes: If true, function returns a zip that iterates through cascaded unions of polygons of the
        same class, otherwise zip iterates through all polygons regardless of their class.
        :type group_classes: boolean
        """
        vector_data = self._get_vector_data(eopatch)

        if 'init' not in vector_data.crs:
            raise ValueError('Cannot recognize CRS of vector data')
        vector_data_crs = CRS(vector_data.crs['init'])
        if eopatch.bbox.crs is not vector_data_crs:
            warnings.warn('Vector data is not in the same CRS as EOPatch, this task will re-project vector data for '
                          'each execution', RuntimeWarning)
            vector_data = vector_data.to_crs(epsg=eopatch.bbox.crs.epsg)

        bbox_poly = eopatch.bbox.geometry
        vector_data = vector_data[vector_data.geometry.intersects(bbox_poly)].copy(deep=True)

        if vector_data.empty:
            return None

        if self.buffer:
            vector_data.geometry = vector_data.geometry.buffer(self.buffer)
            vector_data = vector_data[~vector_data.is_empty]

            # vector_data could be empty as a result of (negative) buffer
            if vector_data.empty:
                return None

        if not vector_data.geometry.is_valid.all():
            warnings.warn('Given vector polygons contain some invalid geometries, they will be fixed', RuntimeWarning)
            vector_data.geometry = vector_data.geometry.buffer(0)

        if vector_data.geometry.has_z.any():
            warnings.warn('Given vector polygons contain some 3D geometries, they will be projected to 2D',
                          RuntimeWarning)
            vector_data.geometry = vector_data.geometry.apply(lambda geo: shapely.wkt.loads(geo.to_wkt()))

        if self.values_column is None:
            return zip(vector_data.geometry, [self.values] * len(vector_data.index))

        if self.values is not None:
            values = [self.values] if isinstance(self.values, (int, float)) else self.values
            vector_data = vector_data[vector_data[self.values_column].isin(values)]

        if group_classes:
            classes = np.unique(vector_data[self.values_column])
            grouped = (vector_data.geometry[vector_data[self.values_column] == cl] for cl in classes)
            grouped = (shapely.ops.cascaded_union(group) for group in grouped)
            return zip(grouped, classes)

        return zip(vector_data.geometry, vector_data[self.values_column])

    def _get_raster_shape(self, eopatch):
        """ Determines the shape of new raster feature, returns a pair (height, width)
        """
        if isinstance(self.raster_shape, (tuple, list)) and len(self.raster_shape) == 2:
            if isinstance(self.raster_shape[0], int) and isinstance(self.raster_shape[1], int):
                return self.raster_shape

            feature_type, feature_name = next(self._parse_features(self.raster_shape)(eopatch))
            return eopatch.get_spatial_dimension(feature_type, feature_name)

        if self.raster_resolution:
            resolution = float(self.raster_resolution.strip('m')) if isinstance(self.raster_resolution, str) else \
                self.raster_resolution
            width, height = bbox_to_dimensions(eopatch.bbox, resolution)
            return height, width

        raise ValueError('Could not determine shape of the raster image')

    def _get_raster(self, eopatch, height, width):
        """ Provides raster in which data will be written
        """
        feature_type, feature_name = self.raster_feature

        if self.write_to_existing and feature_name in eopatch[feature_type]:
            raster = eopatch[self.raster_feature].squeeze(axis=-1)

            if (height, width) != raster.shape[:2]:
                warnings.warn('Writing to existing raster with spatial dimensions {}, but dimensions {} were expected'
                              ''.format(raster.shape[:2], (height, width)), RuntimeWarning)

            return raster

        return np.ones((height, width), dtype=self.raster_dtype) * self.no_data_value

    def rasterize_overlapped(self, shapes, out, **rasterize_args):
        """ Rasterize overlapped classes.

        :param shapes: Shapes to be rasterized.
        :type shapes: an iterable of pairs (rasterio.polygon, int)
        :param out: A numpy array to which to rasterize polygon classes.
        :type out: numpy.ndarray
        :param rasterize_args: Keyword arguments to be passed to `rasterio.features.rasterize`.
        :type rasterize_args: dict
        """
        rasters = [rasterio.features.rasterize([shape], out=np.copy(out), **rasterize_args) for shape in shapes]

        overlap_mask = np.zeros(out.shape, dtype=np.bool)
        no_data = self.no_data_value

        out[:] = rasters[0][:]
        for raster in rasters[1:]:
            overlap_mask[(out != no_data) & (raster != no_data) & (raster != out)] = True
            out[raster != no_data] = raster[raster != no_data]

        out[overlap_mask] = self.overlap_value

    def execute(self, eopatch):
        """ Execute method

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: New EOPatch with vector data transformed into a raster feature
        :rtype: EOPatch
        """
        if eopatch.bbox is None:
            raise ValueError('EOPatch has to have a bounding box')

        height, width = self._get_raster_shape(eopatch)

        group_classes = self.overlap_value is not None
        rasterization_shapes = self._get_rasterization_shapes(eopatch, group_classes=group_classes)

        if not rasterization_shapes:
            eopatch[self.raster_feature] = np.full((height, width, 1), self.no_data_value, dtype=self.raster_dtype)
            return eopatch

        affine_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)
        rasterize_args = dict(self.rasterio_params, transform=affine_transform, dtype=self.raster_dtype)

        raster = self._get_raster(eopatch, height, width)
        rasterize_func = rasterio.features.rasterize if self.overlap_value is None else self.rasterize_overlapped
        rasterize_func(rasterization_shapes, out=raster, **rasterize_args)

        eopatch[self.raster_feature] = raster[..., np.newaxis]
        return eopatch


class RasterToVector(EOTask):
    """ Task for transforming raster mask feature into vector feature.

    Each connected component with the same value on the raster mask is turned into a shapely polygon. Polygon are
    returned as a geometry column in a ``geopandas.GeoDataFrame``structure together with a column `VALUE` with
    values of each polygon.

    If raster mask feature has time component, vector feature will also have a column `TIMESTAMP` with timestamps to
    which raster image each polygon belongs to.
    If raster mask has multiple channels each of them will be vectorized separately but polygons will be in the
    same vector feature
    """
    def __init__(self, features, *, values=None, values_column='VALUE', raster_dtype=None, **rasterio_params):
        """
        :param features: One or more raster mask features which will be vectorized together with an optional new name
        of vector feature. If no new name is given the same name will be used.

        Examples:
            features=(FeatureType.MASK, 'CLOUD_MASK', 'VECTOR_CLOUD_MASK')

            features=[(FeatureType.MASK_TIMELESS, 'CLASSIFICATION'), (FeatureType.MASK, 'MONOTEMPORAL_CLASSIFICATION')]

        :type features: object supported by eolearn.core.utilities.FeatureParser class
        :param values: List of values which will be vectorized. By default is set to ``None`` and all values will be
            vectorized
        :type values: list(int) or None
        :param values_column: Name of the column in vector feature where raster values will be written
        :type values_column: str
        :param raster_dtype: If raster feature mask is of type which is not supported by ``rasterio.features.shapes``
            (e.g. ``numpy.int64``) this parameter is used to cast the mask into a different type
            (``numpy.int16``, ``numpy.int32``, ``numpy.uint8``, ``numpy.uint16`` or ``numpy.float32``). By default
            value of the parameter is ``None`` and no casting is done.
        :type raster_dtype: numpy.dtype or None
        :param: rasterio_params: Additional parameters to be passed to `rasterio.features.shapes`. Currently
            available is parameter `connectivity`.
        """
        self.feature_gen = self._parse_features(features, new_names=True)
        self.values = values
        self.values_column = values_column
        self.raster_dtype = raster_dtype
        self.rasterio_params = rasterio_params

        for feature_type, _, _ in self.feature_gen:
            if not (feature_type.is_spatial() and feature_type.is_discrete()):
                raise ValueError('Input features should be a spatial mask, but {} found'.format(feature_type))

    def _vectorize_single_raster(self, raster, affine_transform, crs, timestamp=None):
        """ Vectorizes a data slice of a single time component

        :param raster: Numpy array or shape (height, width, channels)
        :type raster: numpy.ndarray
        :param affine_transform: Object holding a transform vector (i.e. geographical location vector) of the raster
        :type affine_transform: affine.Affine
        :param crs: Coordinate reference system
        :type crs: sentinelhub.CRS
        :param timestamp: Time of the data slice
        :type timestamp: datetime.datetime
        :return: Vectorized data
        :rtype: geopandas.GeoDataFrame
        """
        mask = None
        if self.values:
            mask = np.zeros(raster.shape, dtype=np.bool)
            for value in self.values:
                mask[raster == value] = True

        geo_list = []
        value_list = []
        for idx in range(raster.shape[-1]):
            for geojson, value in rasterio.features.shapes(raster[..., idx],
                                                           mask=None if mask is None else mask[..., idx],
                                                           transform=affine_transform, **self.rasterio_params):
                geo_list.append(shapely.geometry.shape(geojson))
                value_list.append(value)

        series_dict = {
            self.values_column: GeoSeries(value_list),
            'geometry': GeoSeries(geo_list)
        }
        if timestamp is not None:
            series_dict['TIMESTAMP'] = GeoSeries([timestamp] * len(geo_list))

        vector_data = GeoDataFrame(series_dict, crs={'init': 'epsg:{}'.format(crs.value)})

        if not vector_data.geometry.is_valid.all():
            vector_data.geometry = vector_data.geometry.buffer(0)

        return vector_data

    def execute(self, eopatch):
        """ Execute function which adds new vector layer to the EOPatch

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: New EOPatch with added vector layer
        :rtype: EOPatch
        """
        for raster_ft, raster_fn, vector_fn in self.feature_gen(eopatch):
            vector_ft = FeatureType.VECTOR_TIMELESS if raster_ft.is_timeless() else FeatureType.VECTOR

            raster = eopatch[raster_ft][raster_fn]
            height, width = raster.shape[:2] if raster_ft.is_timeless() else raster.shape[1: 3]

            if self.raster_dtype:
                raster = raster.astype(self.raster_dtype)

            affine_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)

            crs = eopatch.bbox.get_crs()

            if raster_ft.is_timeless():
                eopatch[vector_ft][vector_fn] = self._vectorize_single_raster(raster, affine_transform, crs)
            else:
                gpd_list = [self._vectorize_single_raster(raster[time_idx, ...], affine_transform, crs,
                                                          timestamp=eopatch.timestamp[time_idx])
                            for time_idx in range(raster.shape[0])]

                eopatch[vector_ft][vector_fn] = GeoDataFrame(pd.concat(gpd_list, ignore_index=True),
                                                             crs=gpd_list[0].crs)

        return eopatch
