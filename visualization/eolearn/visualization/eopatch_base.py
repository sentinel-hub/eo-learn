"""
This module implements base objects for `EOPatch` visualizations.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from geopandas import GeoDataFrame

from eolearn.core import EOPatch
from eolearn.core.types import SingleFeatureSpec
from eolearn.core.utils.common import is_discrete_type
from eolearn.core.utils.parsing import parse_feature


@dataclass
class BasePlotConfig:
    """A base class for advanced plotting configuration parameters.

    :param rgb_factor: A factor by which to scale RGB images to make them look better.
    :param timestamp_column: A name of a column containing timestamps in a `GeoDataFrame` feature. If set to `None` it
        will plot temporal vector features as if they were timeless.
    :param geometry_column: A name of a column containing geometries in a `GeoDataFrame` feature.
    """

    rgb_factor: Optional[float] = 3.5
    timestamp_column: Optional[str] = "TIMESTAMP"
    geometry_column: str = "geometry"


class BaseEOPatchVisualization(metaclass=abc.ABCMeta):
    """A base class for EOPatch visualization"""

    def __init__(
        self,
        eopatch: EOPatch,
        feature: SingleFeatureSpec,
        *,
        config: BasePlotConfig,
        times: Union[List[int], slice, None] = None,
        channels: Union[List[int], slice, None] = None,
        channel_names: Optional[List[str]] = None,
        rgb: Optional[Tuple[int, int, int]] = None,
    ):
        """
        :param eopatch: An EOPatch with a feature to plot.
        :param feature: A feature from the given EOPatch to plot.
        :param config: A configuration object with advanced plotting parameters.
        :param times: A list or a slice of indices on temporal axis to be used for plotting. If not provided all
            indices will be used.
        :param channels: A list or a slice of indices on channels axis to be used for plotting. If not provided all
            indices will be used.
        :param channel_names: Names of channels of the last dimension in the given raster feature.
        :param rgb: If provided, it should be a list of 3 indices of RGB channels to be plotted. It will plot only RGB
            images with these channels. This only works for raster features with spatial dimension.
        """
        self.eopatch = eopatch
        self.feature = parse_feature(feature)
        feature_type, _ = self.feature
        self.config = config

        if times is not None and not feature_type.is_temporal():
            raise ValueError("Parameter times can only be provided for temporal features.")
        self.times = times

        self.channels = channels
        self.channel_names = None if channel_names is None else [str(name) for name in channel_names]

        if rgb and len(rgb) != 3:
            raise ValueError(f"Parameter rgb should be a list of 3 indices but got {rgb}")
        if rgb and not (feature_type.is_spatial() and feature_type.is_raster()):
            raise ValueError("Parameter rgb can only be provided for plotting spatial raster features.")
        self.rgb = rgb

        if self.channels and self.rgb:
            raise ValueError("Only one of parameters channels and rgb can be provided.")

    @abc.abstractmethod
    def plot(self) -> object:
        """Plots the given feature"""

    def collect_and_prepare_feature(self) -> Tuple[Any, List[dt.datetime]]:
        """Collects a feature from EOPatch and modifies it according to plotting parameters"""
        feature_type, _ = self.feature
        data = self.eopatch[self.feature]
        timestamps = self.eopatch.timestamp

        if feature_type.is_raster():
            if self.times is not None:
                data = data[self.times, ...]
                if timestamps:
                    timestamps = list(np.array(timestamps)[self.times])

            if self.channels is not None:
                data = data[..., self.channels]

            if feature_type.is_spatial() and self.rgb:
                data = self._prepare_rgb_data(data)

            number_of_plot_columns = 1 if self.rgb else data.shape[-1]
            if self.channel_names and len(self.channel_names) != number_of_plot_columns:
                raise ValueError(
                    f"Provided {len(self.channel_names)} channel names but attempting to make plots with "
                    f"{number_of_plot_columns} columns for the given feature channels."
                )

        if feature_type.is_vector() and self.times is not None:
            data = self._filter_temporal_dataframe(data)

        return data, timestamps

    def _prepare_rgb_data(self, data: np.ndarray) -> np.ndarray:
        """Prepares data array for RGB plotting"""
        data = data[..., self.rgb]

        if self.config.rgb_factor is not None:
            data = data * self.config.rgb_factor

        if is_discrete_type(data.dtype):
            data = np.clip(data, 0, 255)
        else:
            data = np.clip(data, 0.0, 1.0)

        return data

    def _filter_temporal_dataframe(self, dataframe: GeoDataFrame) -> GeoDataFrame:
        """Prepares a list of unique timestamps from the dataframe, applies filter on them and returns a new
        dataframe with rows that only contain filtered timestamps."""
        unique_timestamps = dataframe[self.config.timestamp_column].unique()
        filtered_timestamps = np.sort(unique_timestamps)[self.times]
        filtered_rows = dataframe[self.config.timestamp_column].isin(filtered_timestamps)
        return dataframe[filtered_rows]
