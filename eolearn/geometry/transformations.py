"""
Transformations between vector and raster formats of data

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime as dt
import itertools as it
import logging
import warnings
from functools import partial
from typing import Any, Callable, ClassVar, Iterator, Tuple, cast

import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
import shapely.geometry
import shapely.ops
import shapely.wkt
from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry.base import BaseGeometry

from sentinelhub import CRS, BBox, bbox_to_dimensions, parse_time

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.constants import TIMESTAMP_COLUMN
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.types import Feature, FeaturesSpecification, SingleFeatureSpec

LOGGER = logging.getLogger(__name__)

ShapeIterator = Iterator[Tuple[BaseGeometry, float]]


class VectorToRasterTask(EOTask):
    """A task for transforming a vector feature into a raster feature

    Vector data can be given as an EOPatch feature or as an independent geopandas `GeoDataFrame`.

    In the background it uses `rasterio.features.rasterize`, documented in `rasterio documentation
    <https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.rasterize>`__.
    """

    # A mapping between types that are not supported by rasterio into types that are. After rasterization the task
    # will cast results back into the original dtype.
    _RASTERIO_DTYPES_MAP: ClassVar[dict[type | np.dtype, type]] = {
        bool: np.uint8,
        np.dtype(bool): np.uint8,
        np.int8: np.int16,
        np.dtype(np.int8): np.int16,
        float: np.float64,
        np.dtype(float): np.float64,
    }

    def __init__(
        self,
        vector_input: GeoDataFrame | Feature,
        raster_feature: Feature,
        *,
        values: None | float | list[float] = None,
        values_column: str | None = None,
        raster_shape: None | tuple[int, int] | Feature = None,
        raster_resolution: None | float | tuple[float, float] = None,
        raster_dtype: np.dtype | type = np.uint8,
        no_data_value: float = 0,
        write_to_existing: bool = False,
        overlap_value: float | None = None,
        buffer: float = 0,
        **rasterio_params: Any,
    ):
        """
        :param vector_input: Vector data to be used for rasterization. It can be given as a feature in `EOPatch` or
            as a geopandas `GeoDataFrame`.
        :param raster_feature: New or existing raster feature into which data will be written. If existing raster
            feature is given it will by default take existing values and write over them.
        :param values: If `values_column` parameter is specified then only polygons which have one of these specified
            values in `values_column` will be rasterized. It can be also left to `None`. If `values_column` parameter
            is not specified `values` parameter has to be a single number into which everything will be rasterized.
        :param values_column: A column in given dataframe where values, into which polygons will be rasterized,
            are stored. If it is left to `None` then `values` parameter should be a single number into which
            everything will be rasterized.
        :param raster_shape: Can be a tuple in form of (height, width) or an existing feature from which the spatial
            dimensions will be taken. Parameter `raster_resolution` can be specified instead of `raster_shape`.
        :param raster_resolution: Resolution of raster into which data will be rasterized. Has to be given as a
            number or a tuple of numbers in meters. Parameter `raster_shape` can be specified instead or
            `raster_resolution`.
        :param raster_dtype: Data type of the obtained raster array, default is `numpy.uint8`.
        :param no_data_value: Default value of all other raster pixels into which no value will be rasterized.
        :param write_to_existing: If `True` it will write to existing raster array and overwrite parts of its values.
            If `False` it will create a new raster array and remove the old one. Default is `False`.
        :param overlap_value: A value to override parts of raster where polygons of different classes overlap. If None,
            rasterization overlays polygon as it is the default behavior of `rasterio.features.rasterize`.
        :param buffer: Buffer value passed to vector_data.buffer() before rasterization. If 0, no buffering is done.
        :param: rasterio_params: Additional parameters to be passed to `rasterio.features.rasterize`. Currently,
            available parameters are `all_touched` and `merge_alg`
        """
        self.vector_input, self.raster_feature = self._parse_main_params(vector_input, raster_feature)
        self._rasterize_per_timestamp = self.raster_feature[0].is_temporal()

        if _vector_is_timeless(self.vector_input) and not self.raster_feature[0].is_timeless():
            raise ValueError("Vector input has no time-dependence but a time-dependent output feature was selected")

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

    def _parse_main_params(
        self, vector_input: GeoDataFrame | SingleFeatureSpec, raster_feature: SingleFeatureSpec
    ) -> tuple[GeoDataFrame | Feature, Feature]:
        """Parsing first 2 parameters - what vector data will be used and in which raster feature it will be saved"""
        if not _is_geopandas_object(vector_input):
            vector_input = self.parse_feature(vector_input, allowed_feature_types=lambda fty: fty.is_vector())

        parsed_raster_feature = self.parse_feature(raster_feature, allowed_feature_types=lambda fty: fty.is_image())
        return vector_input, parsed_raster_feature

    def _get_vector_data_iterator(
        self, eopatch: EOPatch, join_per_value: bool
    ) -> Iterator[tuple[dt.datetime | None, ShapeIterator]]:
        """Collects and prepares vector shapes for rasterization. It works as an iterator that returns pairs of
        `(timestamp or None, <iterator over shapes and values>)`

        :param eopatch: An EOPatch from where geometries will be obtained
        :param join_per_value: If `True` it will join geometries with the same value using a cascaded union
        """
        vector_data = self.vector_input if _is_geopandas_object(self.vector_input) else eopatch[self.vector_input]
        # EOPatch has a bbox, verified in execute
        vector_data = self._preprocess_vector_data(vector_data, cast(BBox, eopatch.bbox), eopatch.timestamps)

        if self._rasterize_per_timestamp:
            for timestamp, data_for_time in vector_data.groupby(TIMESTAMP_COLUMN):
                if not data_for_time.empty:
                    yield timestamp.to_pydatetime(), self._vector_data_to_shape_iterator(data_for_time, join_per_value)
        elif not vector_data.empty:
            yield None, self._vector_data_to_shape_iterator(vector_data, join_per_value)

    def _preprocess_vector_data(
        self, vector_data: GeoDataFrame, bbox: BBox, timestamps: list[dt.datetime] | None
    ) -> GeoDataFrame:
        """Applies preprocessing steps on a dataframe with geometries and potential values and timestamps"""
        vector_data = self._reduce_vector_data(vector_data, timestamps)

        if bbox.crs is not CRS(vector_data.crs.to_epsg()):
            warnings.warn(
                "Vector data is not in the same CRS as EOPatch, the task will re-project vectors for each execution",
                EORuntimeWarning,
            )
            vector_data = vector_data.to_crs(bbox.crs.pyproj_crs())

        bbox_poly = bbox.geometry
        vector_data = vector_data[vector_data.geometry.intersects(bbox_poly)].copy(deep=True)

        if self.buffer:
            vector_data.geometry = vector_data.geometry.buffer(self.buffer)
            vector_data = vector_data[~vector_data.is_empty]

        if not vector_data.geometry.is_valid.all():
            warnings.warn("Given vector polygons contain some invalid geometries, attempting to fix", EORuntimeWarning)
            vector_data.geometry = vector_data.geometry.buffer(0)

        if vector_data.geometry.has_z.any():
            warnings.warn("Polygons contain 3D geometries, they will be projected to 2D", EORuntimeWarning)
            vector_data.geometry = vector_data.geometry.map(partial(shapely.ops.transform, lambda *args: args[:2]))

        return vector_data

    def _reduce_vector_data(self, vector_data: GeoDataFrame, timestamps: list[dt.datetime] | None) -> GeoDataFrame:
        """Removes all redundant columns and rows."""
        columns_to_keep = ["geometry"]
        if self._rasterize_per_timestamp:
            columns_to_keep.append(TIMESTAMP_COLUMN)
        if self.values_column is not None:
            columns_to_keep.append(self.values_column)
        vector_data = vector_data[columns_to_keep]

        if self._rasterize_per_timestamp:
            vector_data[TIMESTAMP_COLUMN] = vector_data[TIMESTAMP_COLUMN].apply(parse_time)
            vector_data = vector_data[vector_data[TIMESTAMP_COLUMN].isin(timestamps)]

        if self.values_column is not None and self.values is not None:
            values = [self.values] if isinstance(self.values, (int, float)) else self.values
            vector_data = vector_data[vector_data[self.values_column].isin(values)]
        return vector_data

    def _vector_data_to_shape_iterator(self, vector_data: GeoDataFrame, join_per_value: bool) -> ShapeIterator:
        if self.values_column is None:
            value = cast(float, self.values)  # cast is checked at init
            return zip(vector_data.geometry, it.repeat(value))

        values = vector_data[self.values_column]
        if join_per_value:
            groups = {val: vector_data.geometry[values == val] for val in np.unique(values)}
            join_function = shapely.ops.unary_union if shapely.__version__ >= "1.8.0" else shapely.ops.cascaded_union
            return ((join_function(group), val) for val, group in groups.items())

        return zip(vector_data.geometry, values)

    def _get_raster_shape(self, eopatch: EOPatch) -> tuple[int, int]:
        """Determines the shape of new raster feature, returns a pair (height, width)"""
        if isinstance(self.raster_shape, (tuple, list)) and len(self.raster_shape) == 2:
            if isinstance(self.raster_shape[0], int) and isinstance(self.raster_shape[1], int):
                return self.raster_shape

            ftype, fname = self.parse_feature(self.raster_shape, allowed_feature_types=lambda fty: fty.is_array())
            return eopatch.get_spatial_dimension(ftype, fname)

        if self.raster_resolution:
            # parsing from strings is not denoted in types, so an explicit upcast is required
            raw_resolution: str | float | tuple[float, float] = self.raster_resolution
            resolution = float(raw_resolution.strip("m")) if isinstance(raw_resolution, str) else raw_resolution

            width, height = bbox_to_dimensions(cast(BBox, eopatch.bbox), resolution)  # cast verified in execute
            return height, width

        raise ValueError("Could not determine shape of the raster image")

    def _get_raster(self, eopatch: EOPatch, height: int, width: int) -> np.ndarray:
        """Provides raster into which data will be written"""
        raster_shape = (
            (len(eopatch.get_timestamps()), height, width) if self._rasterize_per_timestamp else (height, width)
        )

        if self.write_to_existing and self.raster_feature in eopatch:
            raster = eopatch[self.raster_feature]

            expected_full_shape = (*raster_shape, 1)
            if raster.shape != expected_full_shape:
                msg = (
                    f"The existing raster feature {self.raster_feature} has a shape {raster.shape} but the expected"
                    f" shape is {expected_full_shape}. This might cause errors or unexpected results."
                )
                warnings.warn(msg, EORuntimeWarning)

            return raster.squeeze(axis=-1)

        return np.full(raster_shape, self.no_data_value, dtype=self.raster_dtype)

    def _get_rasterization_function(self, bbox: BBox, height: int, width: int) -> Callable:
        """Provides a function that rasterizes shapes into output raster and already contains all optional parameters"""
        affine_transform = rasterio.transform.from_bounds(*bbox, width=width, height=height)
        rasterize_params = dict(self.rasterio_params, transform=affine_transform)

        base_rasterize_func = rasterio.features.rasterize if self.overlap_value is None else self.rasterize_overlapped

        return partial(base_rasterize_func, **rasterize_params)

    def rasterize_overlapped(self, shapes: ShapeIterator, out: np.ndarray, **rasterize_args: Any) -> None:
        """Rasterize overlapped classes.

        :param shapes: Shapes to be rasterized.
        :param out: A numpy array to which to rasterize polygon classes.
        :param rasterize_args: Keyword arguments to be passed to `rasterio.features.rasterize`.
        """
        rasters = [rasterio.features.rasterize([shape], out=np.copy(out), **rasterize_args) for shape in shapes]

        overlap_mask = np.zeros(out.shape, dtype=bool)
        no_data = self.no_data_value

        out[:] = rasters[0][:]
        for raster in rasters[1:]:
            overlap_mask[(out != no_data) & (raster != no_data) & (raster != out)] = True
            out[raster != no_data] = raster[raster != no_data]

        out[overlap_mask] = self.overlap_value

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute method

        :param eopatch: input EOPatch
        :return: New EOPatch with vector data transformed into a raster feature
        """
        if eopatch.bbox is None:
            raise ValueError("EOPatch has to have a bounding box")

        height, width = self._get_raster_shape(eopatch)

        rasterize_func = self._get_rasterization_function(eopatch.bbox, height=height, width=width)
        vector_data_iterator = self._get_vector_data_iterator(eopatch, join_per_value=self.overlap_value is not None)

        raster = self._get_raster(eopatch, height, width)
        original_dtype = raster.dtype
        if original_dtype in self._RASTERIO_DTYPES_MAP:
            rasterio_dtype = self._RASTERIO_DTYPES_MAP[original_dtype]
            raster = raster.astype(rasterio_dtype)

        timestamps = eopatch.timestamps or []
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}

        for timestamp, shape_iterator in vector_data_iterator:
            if timestamp is None:
                rasterize_func(shape_iterator, out=raster)
            else:
                time_index = timestamp_to_index[timestamp]
                rasterize_func(shape_iterator, out=raster[time_index, ...])

        if original_dtype is not raster.dtype:
            raster = raster.astype(original_dtype)

        eopatch[self.raster_feature] = raster[..., np.newaxis]
        return eopatch


class RasterToVectorTask(EOTask):
    """Task for transforming raster mask feature into vector feature.

    Each connected component with the same value on the raster mask is turned into a shapely polygon. Polygon are
    returned as a geometry column in a ``geopandas.GeoDataFrame`` structure together with a column `VALUE` with
    values of each polygon.

    If raster mask feature has time component, vector feature will also have a column `TIMESTAMP` with timestamps to
    which raster image each polygon belongs to.
    If raster mask has multiple channels each of them will be vectorized separately but polygons will be in the
    same vector feature
    """

    def __init__(
        self,
        features: FeaturesSpecification,
        *,
        values: list[int] | None = None,
        values_column: str = "VALUE",
        raster_dtype: None | np.dtype | type = None,
        **rasterio_params: Any,
    ):
        """
        :param features: One or more raster mask features which will be vectorized together with an optional new name
            of vector feature. If no new name is given the same name will be used.

            Examples:

            - `features=(FeatureType.MASK, 'CLOUD_MASK', 'VECTOR_CLOUD_MASK')`
            - `features=[(FeatureType.MASK_TIMELESS, 'CLASSIFICATION'), (FeatureType.MASK, 'TEMPORAL_CLASSIFICATION')]`
        :param values: List of values which will be vectorized. By default, is set to ``None`` and all values will be
            vectorized
        :param values_column: Name of the column in vector feature where raster values will be written
        :param raster_dtype: If raster feature mask is of type which is not supported by ``rasterio.features.shapes``
            (e.g. ``numpy.int64``) this parameter is used to cast the mask into a different type
            (``numpy.int16``, ``numpy.int32``, ``numpy.uint8``, ``numpy.uint16`` or ``numpy.float32``). By default,
            value of the parameter is ``None`` and no casting is done.
        :param: rasterio_params: Additional parameters to be passed to `rasterio.features.shapes`. Currently,
            available is parameter `connectivity`.
        """
        self.feature_parser = self.get_feature_parser(features, allowed_feature_types=lambda fty: fty.is_discrete())
        self.values = values
        self.values_column = values_column
        self.raster_dtype = raster_dtype
        self.rasterio_params = rasterio_params

    def _vectorize_single_raster(
        self, raster: np.ndarray, affine_transform: Affine, crs: CRS, timestamps: dt.datetime | None = None
    ) -> GeoDataFrame:
        """Vectorizes a data slice of a single time component

        :param raster: Numpy array or shape (height, width, channels)
        :param affine_transform: Object holding a transform vector (i.e. geographical location vector) of the raster
        :param crs: Coordinate reference system
        :param timestamp: Time of the data slice
        :return: Vectorized data
        """
        mask = np.isin(raster, self.values) if self.values is not None else None

        geo_list = []
        value_list = []
        for idx in range(raster.shape[-1]):
            idx_mask = None if mask is None else mask[..., idx]
            for geojson, value in rasterio.features.shapes(
                raster[..., idx], mask=idx_mask, transform=affine_transform, **self.rasterio_params
            ):
                geo_list.append(shapely.geometry.shape(geojson))
                value_list.append(value)

        series_dict = {self.values_column: pd.Series(value_list, dtype=self.raster_dtype)}
        if timestamps is not None:
            series_dict[TIMESTAMP_COLUMN] = pd.to_datetime([timestamps] * len(geo_list))

        vector_data = GeoDataFrame(series_dict, geometry=geo_list, crs=crs.pyproj_crs())

        if not vector_data.geometry.is_valid.all():
            vector_data.geometry = vector_data.geometry.buffer(0)

        return vector_data

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute function which adds new vector layer to the EOPatch

        :param eopatch: input EOPatch
        :return: New EOPatch with added vector layer
        """
        if eopatch.bbox is None:
            raise ValueError("EOPatch has to have a bounding box")
        crs = eopatch.bbox.crs

        for raster_type, raster_name, vector_name in self.feature_parser.get_renamed_features(eopatch):
            vector_type = FeatureType.VECTOR_TIMELESS if raster_type.is_timeless() else FeatureType.VECTOR
            raster = eopatch[raster_type, raster_name]
            height, width = raster.shape[:2] if raster_type.is_timeless() else raster.shape[1:3]

            if self.raster_dtype:
                raster = raster.astype(self.raster_dtype)

            affine_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)

            if raster_type.is_timeless():
                eopatch[vector_type, vector_name] = self._vectorize_single_raster(raster, affine_transform, crs)
            else:
                timestamps = eopatch.get_timestamps()
                gpd_list = [
                    self._vectorize_single_raster(raster[idx, ...], affine_transform, crs, timestamps[idx])
                    for idx in range(raster.shape[0])
                ]

                eopatch[vector_type, vector_name] = GeoDataFrame(
                    pd.concat(gpd_list, ignore_index=True), crs=gpd_list[0].crs
                )

        return eopatch


def _is_geopandas_object(data: object) -> bool:
    """A frequently used check if object is geopandas `GeoDataFrame` or `GeoSeries`"""
    return isinstance(data, (GeoDataFrame, GeoSeries))


def _vector_is_timeless(vector_input: GeoDataFrame | tuple[FeatureType, Any]) -> bool:
    """Used to check if the vector input (either geopandas object EOPatch Feature) is time independent"""
    if _is_geopandas_object(vector_input):
        return TIMESTAMP_COLUMN not in vector_input

    vector_type, _ = vector_input
    return vector_type.is_timeless()
