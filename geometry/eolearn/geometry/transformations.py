"""
Transformations between vector and raster formats of data

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import datetime as dt
import functools
import logging
import warnings
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pyproj
import rasterio.features
import rasterio.transform
import shapely.geometry
import shapely.ops
import shapely.wkt
from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry.base import BaseGeometry

from sentinelhub import CRS, BBox, bbox_to_dimensions, parse_time

from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.types import FeaturesSpecification, SingleFeatureSpec

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
    _RASTERIO_BASIC_DTYPES_MAP = {
        bool: np.uint8,
        np.int8: np.int16,
        float: np.float64,
    }
    _RASTERIO_DTYPES_MAP = {
        dtype: rasterio_type
        for basic_type, rasterio_type in _RASTERIO_BASIC_DTYPES_MAP.items()
        for dtype in [basic_type, np.dtype(basic_type)]
    }

    def __init__(
        self,
        vector_input: Union[GeoDataFrame, SingleFeatureSpec],
        raster_feature: SingleFeatureSpec,
        *,
        values: Union[None, float, List[float]] = None,
        values_column: Optional[str] = None,
        raster_shape: Union[None, Tuple[int, int], SingleFeatureSpec] = None,
        raster_resolution: Union[None, float, Tuple[float, float]] = None,
        raster_dtype: Union[np.dtype, type] = np.uint8,
        no_data_value: float = 0,
        write_to_existing: bool = False,
        overlap_value: Optional[float] = None,
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

        if _vector_is_timeless(self.vector_input) and not self.raster_feature[0].is_timeless():
            raise ValueError("Vector input has no time-dependence but a time-dependent raster feature was selected")

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

        self._rasterize_per_timestamp = self.raster_feature[0].is_temporal()

    def _parse_main_params(
        self, vector_input: Union[GeoDataFrame, SingleFeatureSpec], raster_feature: SingleFeatureSpec
    ) -> Tuple[Union[GeoDataFrame, Tuple[FeatureType, str]], Tuple[FeatureType, str]]:
        """Parsing first 2 parameters - what vector data will be used and in which raster feature it will be saved"""
        if not _is_geopandas_object(vector_input):
            vector_input = self.parse_feature(vector_input, allowed_feature_types=FeatureTypeSet.VECTOR_TYPES)

        parsed_raster_feature = self.parse_feature(
            raster_feature, allowed_feature_types=FeatureTypeSet.RASTER_TYPES_3D.union(FeatureTypeSet.RASTER_TYPES_4D)
        )
        return vector_input, parsed_raster_feature  # type: ignore[return-value]

    def _get_vector_data_iterator(
        self, eopatch: EOPatch, join_per_value: bool
    ) -> Iterator[Tuple[Optional[dt.datetime], Optional[ShapeIterator]]]:
        """Collects and prepares vector shapes for rasterization. It works as an iterator that returns pairs of
        `(timestamp or None, <iterator over shapes and values>)`

        :param eopatch: An EOPatch from where geometries will be obtained
        :param join_per_value: If `True` it will join geometries with the same value using a cascaded union
        """
        vector_data = self._get_vector_data_from_eopatch(eopatch)
        # EOPatch has a bbox, verified in execute
        vector_data = self._preprocess_vector_data(vector_data, cast(BBox, eopatch.bbox), eopatch.timestamp)

        if self._rasterize_per_timestamp:
            for timestamp, vector_data_per_timestamp in vector_data.groupby("TIMESTAMP"):
                yield timestamp.to_pydatetime(), self._vector_data_to_shape_iterator(
                    vector_data_per_timestamp, join_per_value
                )
        else:
            yield None, self._vector_data_to_shape_iterator(vector_data, join_per_value)

    def _get_vector_data_from_eopatch(self, eopatch: EOPatch) -> GeoDataFrame:
        """Provides a vector dataframe either from the attribute or from given EOPatch feature"""
        if _is_geopandas_object(self.vector_input):
            return self.vector_input

        return eopatch[self.vector_input]

    def _preprocess_vector_data(
        self, vector_data: GeoDataFrame, bbox: BBox, timestamps: List[dt.datetime]
    ) -> GeoDataFrame:
        """Applies preprocessing steps on a dataframe with geometries and potential values and timestamps"""
        columns_to_keep = ["geometry"]
        if self._rasterize_per_timestamp:
            columns_to_keep.append("TIMESTAMP")
        if self.values_column is not None:
            columns_to_keep.append(self.values_column)
        vector_data = vector_data[columns_to_keep]

        if self._rasterize_per_timestamp:
            vector_data["TIMESTAMP"] = vector_data.TIMESTAMP.apply(parse_time)
            vector_data = vector_data[vector_data.TIMESTAMP.isin(timestamps)]

        if self.values_column is not None and self.values is not None:
            values = [self.values] if isinstance(self.values, (int, float)) else self.values
            vector_data = vector_data[vector_data[self.values_column].isin(values)]

        gpd_crs = vector_data.crs
        # This special case has to be handled because of WGS84 and lat-lon order:
        if isinstance(gpd_crs, pyproj.CRS):
            gpd_crs = gpd_crs.to_epsg()
        vector_data_crs = CRS(gpd_crs)

        if bbox.crs is not vector_data_crs:
            warnings.warn(
                (
                    "Vector data is not in the same CRS as EOPatch, this task will re-project vector data for "
                    "each execution"
                ),
                EORuntimeWarning,
            )
            vector_data = vector_data.to_crs(bbox.crs.pyproj_crs())

        bbox_poly = bbox.geometry
        vector_data = vector_data[vector_data.geometry.intersects(bbox_poly)].copy(deep=True)

        if self.buffer:
            vector_data.geometry = vector_data.geometry.buffer(self.buffer)
            vector_data = vector_data[~vector_data.is_empty]

        if not vector_data.geometry.is_valid.all():
            warnings.warn("Given vector polygons contain some invalid geometries, they will be fixed", EORuntimeWarning)
            vector_data.geometry = vector_data.geometry.buffer(0)

        if vector_data.geometry.has_z.any():
            warnings.warn(
                "Given vector polygons contain some 3D geometries, they will be projected to 2D", EORuntimeWarning
            )
            vector_data.geometry = vector_data.geometry.map(
                functools.partial(shapely.ops.transform, lambda *args: args[:2])
            )

        return vector_data

    def _vector_data_to_shape_iterator(
        self, vector_data: GeoDataFrame, join_per_value: bool
    ) -> Optional[ShapeIterator]:
        """Returns an iterator of pairs `(shape, value)` or `None` if given dataframe is empty"""
        if vector_data.empty:
            return None

        if self.values_column is None:
            value = cast(float, self.values)  # cast is checked at init
            return zip(vector_data.geometry, [value] * len(vector_data.index))

        if join_per_value:
            classes = np.unique(vector_data[self.values_column])
            grouped = (vector_data.geometry[vector_data[self.values_column] == cl] for cl in classes)
            join_function = shapely.ops.unary_union if shapely.__version__ >= "1.8.0" else shapely.ops.cascaded_union
            grouped = (join_function(group) for group in grouped)
            return zip(grouped, classes)

        return zip(vector_data.geometry, vector_data[self.values_column])

    def _get_raster_shape(self, eopatch: EOPatch) -> Tuple[int, int]:
        """Determines the shape of new raster feature, returns a pair (height, width)"""
        if isinstance(self.raster_shape, (tuple, list)) and len(self.raster_shape) == 2:
            if isinstance(self.raster_shape[0], int) and isinstance(self.raster_shape[1], int):
                return self.raster_shape

            feature_type, feature_name = self.parse_feature(
                self.raster_shape, allowed_feature_types=FeatureTypeSet.RASTER_TYPES
            )
            return eopatch.get_spatial_dimension(feature_type, cast(str, feature_name))  # cast verified in parser

        if self.raster_resolution:
            # parsing from strings is not denoted in types, so an explicit upcast is required
            raw_resolution: Union[str, float, Tuple[float, float]] = self.raster_resolution
            resolution = float(raw_resolution.strip("m")) if isinstance(raw_resolution, str) else raw_resolution

            width, height = bbox_to_dimensions(cast(BBox, eopatch.bbox), resolution)  # cast verified in execute
            return height, width

        raise ValueError("Could not determine shape of the raster image")

    def _get_raster(self, eopatch: EOPatch, height: int, width: int) -> np.ndarray:
        """Provides raster into which data will be written"""
        feature_type, feature_name = self.raster_feature
        raster_shape = (len(eopatch.timestamp), height, width) if self._rasterize_per_timestamp else (height, width)

        if self.write_to_existing and feature_name in eopatch[feature_type]:
            raster = eopatch[self.raster_feature]

            expected_full_shape = raster_shape + (1,)
            if raster.shape != expected_full_shape:
                warnings.warn(
                    (
                        f"The existing raster feature {self.raster_feature} has a shape {raster.shape} but "
                        f"the expected shape is {expected_full_shape}. This might cause errors or unexpected "
                        "results."
                    ),
                    EORuntimeWarning,
                )

            return raster.squeeze(axis=-1)

        return np.full(raster_shape, self.no_data_value, dtype=self.raster_dtype)

    def _get_rasterization_function(self, bbox: BBox, height: int, width: int) -> Callable:
        """Provides a function that rasterizes shapes into output raster and already contains all optional parameters"""
        affine_transform = rasterio.transform.from_bounds(*bbox, width=width, height=height)
        rasterize_params = dict(self.rasterio_params, transform=affine_transform)

        base_rasterize_func = rasterio.features.rasterize if self.overlap_value is None else self.rasterize_overlapped

        return functools.partial(base_rasterize_func, **rasterize_params)

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

        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(eopatch.timestamp)}

        for timestamp, shape_iterator in vector_data_iterator:
            if shape_iterator is None:
                continue

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
        values: Optional[List[int]] = None,
        values_column: str = "VALUE",
        raster_dtype: Union[None, np.dtype, type] = None,
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
        self.feature_parser = self.get_feature_parser(features, allowed_feature_types=FeatureTypeSet.DISCRETE_TYPES)
        self.values = values
        self.values_column = values_column
        self.raster_dtype = raster_dtype
        self.rasterio_params = rasterio_params

    def _vectorize_single_raster(
        self, raster: np.ndarray, affine_transform: Affine, crs: CRS, timestamp: Optional[dt.datetime] = None
    ) -> GeoDataFrame:
        """Vectorizes a data slice of a single time component

        :param raster: Numpy array or shape (height, width, channels)
        :param affine_transform: Object holding a transform vector (i.e. geographical location vector) of the raster
        :param crs: Coordinate reference system
        :param timestamp: Time of the data slice
        :return: Vectorized data
        """
        mask = None
        if self.values:
            mask = np.zeros(raster.shape, dtype=bool)
            for value in self.values:
                mask[raster == value] = True

        geo_list = []
        value_list = []
        for idx in range(raster.shape[-1]):
            for geojson, value in rasterio.features.shapes(
                raster[..., idx],
                mask=None if mask is None else mask[..., idx],
                transform=affine_transform,
                **self.rasterio_params,
            ):
                geo_list.append(shapely.geometry.shape(geojson))
                value_list.append(value)

        series_dict = {self.values_column: pd.Series(value_list, dtype=self.raster_dtype)}
        if timestamp is not None:
            series_dict["TIMESTAMP"] = pd.to_datetime([timestamp] * len(geo_list))

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

        for raster_ft, raster_fn, vector_fn in self.feature_parser.get_renamed_features(eopatch):
            vector_ft = FeatureType.VECTOR_TIMELESS if raster_ft.is_timeless() else FeatureType.VECTOR

            raster = eopatch[raster_ft][raster_fn]
            height, width = raster.shape[:2] if raster_ft.is_timeless() else raster.shape[1:3]

            if self.raster_dtype:
                raster = raster.astype(self.raster_dtype)

            affine_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)

            crs = eopatch.bbox.crs

            if raster_ft.is_timeless():
                eopatch[vector_ft][vector_fn] = self._vectorize_single_raster(raster, affine_transform, crs)
            else:
                gpd_list = [
                    self._vectorize_single_raster(
                        raster[time_idx, ...], affine_transform, crs, timestamp=eopatch.timestamp[time_idx]
                    )
                    for time_idx in range(raster.shape[0])
                ]

                eopatch[vector_ft][vector_fn] = GeoDataFrame(
                    pd.concat(gpd_list, ignore_index=True), crs=gpd_list[0].crs
                )

        return eopatch


def _is_geopandas_object(data: object) -> bool:
    """A frequently used check if object is geopandas `GeoDataFrame` or `GeoSeries`"""
    return isinstance(data, (GeoDataFrame, GeoSeries))


def _vector_is_timeless(vector_input: Union[GeoDataFrame, Tuple[FeatureType, Any]]) -> bool:
    """Used to check if the vector input (either geopandas object EOPatch Feature) is time independent"""
    if _is_geopandas_object(vector_input):
        return "TIMESTAMP" not in vector_input

    vector_type, _ = vector_input
    return vector_type.is_timeless()
