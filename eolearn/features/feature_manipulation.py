"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import partial
from typing import Any, Callable, Iterable, Literal

import numpy as np
from geopandas import GeoDataFrame

from sentinelhub import bbox_to_dimensions

from eolearn.core import EOPatch, EOTask
from eolearn.core.constants import TIMESTAMP_COLUMN
from eolearn.core.types import Feature, FeaturesSpecification
from eolearn.core.utils.parsing import parse_renamed_features

from .utils import ResizeLib, ResizeMethod, ResizeParam, spatially_resize_image

LOGGER = logging.getLogger(__name__)


class SimpleFilterTask(EOTask):
    """
    Transforms an eopatch of shape [n, w, h, d] into [m, w, h, d] for m <= n. It removes all slices which don't
    conform to the filter_func.

    A filter_func is a callable which takes a numpy array or list of datetimes and returns a bool.
    """

    def __init__(
        self,
        feature: Feature | Literal["timestamps"],
        filter_func: Callable[[np.ndarray], bool] | Callable[[dt.datetime], bool],
        filter_features: FeaturesSpecification = ...,
    ):
        """
        :param feature: Feature in the EOPatch , e.g. feature=(FeatureType.DATA, 'bands')
        :param filter_func: A callable that takes a numpy evaluates to bool.
        :param filter_features: A collection of features which will be filtered into a new EOPatch
        """
        if feature == "timestamps":
            self.feature: Feature | Literal["timestamps"] = "timestamps"
        else:
            self.feature = self.parse_feature(
                feature, allowed_feature_types=lambda fty: fty.is_temporal() and fty.is_array()
            )
        self.filter_func = filter_func
        self.filter_features_parser = self.get_feature_parser(filter_features)

    def _get_filtered_indices(self, feature_data: Iterable) -> list[int]:
        """Get valid time indices from either a numpy array or a list of timestamps."""
        return [idx for idx, img in enumerate(feature_data) if self.filter_func(img)]

    @staticmethod
    def _filter_vector_feature(gdf: GeoDataFrame, good_idxs: list[int], timestamps: list[dt.datetime]) -> GeoDataFrame:
        """Filters rows that don't match with the timestamps that will be kept."""
        timestamps_to_keep = {timestamps[idx] for idx in good_idxs}
        return gdf[gdf[TIMESTAMP_COLUMN].isin(timestamps_to_keep)]

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: An input EOPatch.
        :return: A new EOPatch with filtered features.
        """
        selector_data = eopatch.timestamps if self.feature == "timestamps" else eopatch[self.feature]
        good_idxs = self._get_filtered_indices(selector_data)
        timestamps = None if eopatch.timestamps is None else [eopatch.timestamps[idx] for idx in good_idxs]
        filtered_eopatch = EOPatch(bbox=eopatch.bbox, timestamps=timestamps)

        for feature in self.filter_features_parser.get_features(eopatch):
            feature_type, _ = feature
            data = eopatch[feature]

            if feature_type.is_temporal():
                if feature_type.is_array():
                    data = data[good_idxs]
                else:
                    data = self._filter_vector_feature(data, good_idxs, eopatch.get_timestamps())

            filtered_eopatch[feature] = data

        return filtered_eopatch


class FilterTimeSeriesTask(SimpleFilterTask):
    """
    Removes all frames in the time-series with dates outside the user specified time interval.
    """

    def _filter_func(self, date: dt.datetime) -> bool:
        return self.start_date <= date <= self.end_date

    def __init__(self, start_date: dt.datetime, end_date: dt.datetime, filter_features: FeaturesSpecification = ...):
        """
        :param start_date: Start date. All frames within the time-series taken after this date will be kept.
        :param end_date: End date. All frames within the time-series taken before this date will be kept.
        :param filter_features: A collection of features which will be filtered
        """
        self.start_date = start_date
        self.end_date = end_date

        if not isinstance(start_date, dt.datetime) or not isinstance(end_date, dt.datetime):
            raise ValueError("Both start_date and end_date must be datetime.datetime objects.")

        super().__init__("timestamps", self._filter_func, filter_features)


class SpatialResizeTask(EOTask):
    """Resizes the specified spatial features of EOPatch."""

    def __init__(
        self,
        *,
        resize_type: ResizeParam,
        height_param: float,
        width_param: float,
        features: FeaturesSpecification = ...,
        resize_method: ResizeMethod = ResizeMethod.LINEAR,
        resize_library: ResizeLib = ResizeLib.CV2,
    ):
        """
        :param features: Which features to resize. Supports new names for features.
        :param resize_type: Determines type of resizing process and how `width_param` and `height_param` are used.
            Options:
                * `new_size`: Resizes data to size (width_param, height_param)
                * | `resolution`: Resizes the data to have width_param, height_param resolution over width/height axis.
                  | Uses EOPatch bbox to compute.
                * | `scale_factor` Resizes the data by scaling the width and height by a factor set by
                  | width_param and height_param respectively.
        :param height_param: Parameter to be applied to the height in combination with the resize_type
        :param width_param: Parameter to be applied to the width in combination with the resize_type
        :param resize_method: Interpolation method used for resizing.
        :param resize_library: Which Python library to use for resizing. Default is CV2 because it is faster, but one
            can use PIL, which features anti-aliasing.
        """
        self.features = features
        self.resize_type = ResizeParam(resize_type)
        self.height_param = height_param
        self.width_param = width_param

        self.resize_function = partial(
            spatially_resize_image, resize_method=resize_method, resize_library=resize_library
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        resize_fun_kwargs: dict[str, Any]
        if self.resize_type == ResizeParam.RESOLUTION:
            if not eopatch.bbox:
                raise ValueError("Resolution-specified resizing can only be done on EOPatches with a defined BBox.")
            new_width, new_height = bbox_to_dimensions(eopatch.bbox, (self.width_param, self.height_param))
            resize_fun_kwargs = {ResizeParam.NEW_SIZE.value: (new_height, new_width)}
        else:
            resize_fun_kwargs = {self.resize_type.value: (self.height_param, self.width_param)}

        for ftype, fname, new_name in parse_renamed_features(self.features, eopatch=eopatch):
            if ftype.is_spatial() and ftype.is_array():
                eopatch[ftype, new_name] = self.resize_function(eopatch[ftype, fname], **resize_fun_kwargs)
        return eopatch
