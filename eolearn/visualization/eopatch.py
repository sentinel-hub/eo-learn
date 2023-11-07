"""
This module implements visualizations for `EOPatch`

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime as dt
import itertools as it
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS

from eolearn.core import EOPatch
from eolearn.core.constants import TIMESTAMP_COLUMN
from eolearn.core.types import Feature
from eolearn.core.utils.common import is_discrete_type
from eolearn.core.utils.parsing import parse_feature


class PlotBackend(Enum):
    """Types of backend for plotting"""

    MATPLOTLIB = "matplotlib"


def plot_eopatch(*args: Any, backend: PlotBackend | str = PlotBackend.MATPLOTLIB, **kwargs: Any) -> object:
    """The main `EOPatch` plotting function. It pr

    :param args: Positional arguments to be propagated to a plotting backend.
    :param backend: Which plotting backend to use.
    :param kwargs: Keyword arguments to be propagated to a plotting backend.
    :return: A plot object that depends on the backend used.
    """
    backend = PlotBackend(backend)

    if backend is PlotBackend.MATPLOTLIB:
        return MatplotlibVisualization(*args, **kwargs).plot()

    raise ValueError(f"EOPatch plotting backend {backend} is not supported")


@dataclass
class PlotConfig:
    """Advanced plotting configurations

    :param rgb_factor: A factor by which to scale RGB images to make them look better.
    :param timestamp_column: A name of a column containing timestamps in a `GeoDataFrame` feature. If set to `None` it
        will plot temporal vector features as if they were timeless.
    :param geometry_column: A name of a column containing geometries in a `GeoDataFrame` feature.
    :param subplot_width: A width of each subplot in a grid
    :param subplot_height: A height of each subplot in a grid
    :param subplot_kwargs: A dictionary of parameters that will be passed to `matplotlib.pyplot.subplots` function.
    :param show_title: A flag to specify if plot title should be shown.
    :param title_kwargs: A dictionary of parameters that will be passed to `matplotlib.figure.Figure.suptitle`.
    :param label_kwargs: A dictionary of parameters that will be passed to `matplotlib` methods for setting axes labels.
    :param bbox_kwargs: A dictionary of parameters that will be passed to `GeoDataFrame.plot` when plotting a bounding
        box.
    """

    rgb_factor: float | None = 3.5
    timestamp_column: str | None = TIMESTAMP_COLUMN
    geometry_column: str = "geometry"
    subplot_width: float | int = 8
    subplot_height: float | int = 8
    interpolation: str = "none"
    subplot_kwargs: dict[str, object] = field(default_factory=dict)
    show_title: bool = True
    title_kwargs: dict[str, object] = field(default_factory=dict)
    label_kwargs: dict[str, object] = field(default_factory=dict)
    bbox_kwargs: dict[str, object] = field(default_factory=dict)


class MatplotlibVisualization:
    """EOPatch visualization using `matplotlib` framework."""

    def __init__(
        self,
        eopatch: EOPatch,
        feature: Feature,
        *,
        axes: np.ndarray | None = None,
        config: PlotConfig | None = None,
        times: list[int] | slice | None = None,
        channels: list[int] | slice | None = None,
        channel_names: list[str] | None = None,
        rgb: tuple[int, int, int] | None = None,
    ):
        """
        :param eopatch: An EOPatch with a feature to plot.
        :param feature: A feature from the given EOPatch to plot.
        :param axes: A grid of axes on which to write plots. If not provided it will create a new grid.
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
        self.config = config or PlotConfig()

        if times is not None and not feature_type.is_temporal():
            raise ValueError("Parameter times can only be provided for temporal features.")
        self.times = times

        self.channels = channels
        self.channel_names = None if channel_names is None else [str(name) for name in channel_names]

        if rgb and not (feature_type.is_spatial() and feature_type.is_array()):
            raise ValueError("Parameter rgb can only be provided for plotting spatial raster features.")
        self.rgb = rgb

        if self.channels and self.rgb:
            raise ValueError("Only one of parameters channels and rgb can be provided.")

        if axes is not None and not isinstance(axes, np.ndarray):
            axes = np.array([np.array([axes])])  # type: ignore[unreachable]
        self.axes = axes

    def plot(self) -> np.ndarray:
        """Plots the given feature"""
        feature_type, feature_name = self.feature
        data, timestamps = self.collect_and_prepare_feature(self.eopatch)

        if feature_type.is_vector():
            return self._plot_vector_feature(
                data,
                timestamp_column=self.config.timestamp_column if feature_type.is_temporal() else None,
                title=feature_name,
            )

        if not feature_type.is_array():
            raise ValueError(f"Plotting of {feature_type} is not supported")

        if feature_type.is_spatial():
            if feature_type.is_timeless():
                return self._plot_raster_grid(data[np.newaxis, ...], title=feature_name)
            return self._plot_raster_grid(data, timestamps=timestamps, title=feature_name)

        if feature_type.is_timeless():
            return self._plot_bar(data, title=feature_name)
        return self._plot_time_series(data, timestamps=timestamps, title=feature_name)

    def collect_and_prepare_feature(self, eopatch: EOPatch) -> tuple[Any, list[dt.datetime] | None]:
        """Collects a feature from EOPatch and modifies it according to plotting parameters"""
        feature_type, _ = self.feature
        data = eopatch[self.feature]
        timestamps = eopatch.timestamps

        if feature_type.is_array():
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

    def _plot_raster_grid(
        self, raster: np.ndarray, timestamps: list[dt.datetime] | None = None, title: str | None = None
    ) -> np.ndarray:
        """Plots a grid of raster images"""
        rows, _, _, columns = raster.shape
        if self.rgb:
            columns = 1

        axes = self._provide_axes(
            nrows=rows,
            ncols=columns,
            title=title,
            sharey=True,
            subplot_kw={"xticks": [], "yticks": [], "frame_on": False},
        )

        label_kwargs = self._get_label_kwargs()
        for (row_idx, column_idx), axis in zip(it.product(range(rows), range(columns)), axes.flatten()):
            raster_slice = raster[row_idx, ...] if self.rgb else raster[row_idx, ..., column_idx]
            axis.imshow(raster_slice, interpolation=self.config.interpolation)

            if timestamps and column_idx == 0:
                axis.set_ylabel(timestamps[row_idx].isoformat(), **label_kwargs)
            if self.channel_names:
                axis.set_xlabel(self.channel_names[column_idx], **label_kwargs)

        return axes

    def _plot_time_series(
        self, series: np.ndarray, timestamps: list[dt.datetime] | None = None, title: str | None = None
    ) -> np.ndarray:
        """Plots time series feature."""
        axes = self._provide_axes(nrows=1, ncols=1, title=title)
        axis = axes.flatten()[0]

        xlabels = np.array(timestamps) if timestamps else np.arange(series.shape[0])
        channel_num = series.shape[-1]
        for idx in range(channel_num):
            channel_label = self.channel_names[idx] if self.channel_names else None
            axis.plot(xlabels, series[..., idx], label=channel_label)

        if self.channel_names:
            axis.legend()
        return axes

    def _plot_bar(self, values: np.ndarray, title: str | None = None) -> np.ndarray:
        """Make a bar plot from values."""
        axes = self._provide_axes(nrows=1, ncols=1, title=title)
        axis = axes.flatten()[0]

        xlabels = np.array(self.channel_names) if self.channel_names else np.arange(values.size)
        axis.bar(xlabels, values)

        return axes

    def _plot_vector_feature(
        self, dataframe: GeoDataFrame, timestamp_column: str | None = None, title: str | None = None
    ) -> np.ndarray:
        """Plots a GeoDataFrame vector feature"""
        rows = len(dataframe[timestamp_column].unique()) if timestamp_column else 1
        axes = self._provide_axes(nrows=rows, ncols=1, title=title)

        if self.eopatch.bbox:
            self._plot_bbox(axes=axes, target_crs=dataframe.crs)

        if timestamp_column is None:
            dataframe.plot(ax=axes.flatten()[0])
            return axes

        timestamp_groups = dataframe.groupby(timestamp_column)
        timestamps = sorted(timestamp_groups.groups)

        label_kwargs = self._get_label_kwargs()
        for timestamp, axis in zip(timestamps, axes.flatten()):
            timestamp_groups.get_group(timestamp).plot(ax=axis)
            axis.set_ylabel(timestamp.isoformat(), **label_kwargs)

        return axes

    def _plot_bbox(self, axes: np.ndarray | None = None, target_crs: CRS | None = None) -> np.ndarray:
        """Plot a bounding box"""
        bbox = self.eopatch.bbox
        if bbox is None:
            raise ValueError("EOPatch doesn't have a bounding box")

        if axes is None:
            axes = self._provide_axes(nrows=1, ncols=1, title="Bounding box")

        bbox_gdf = GeoDataFrame(geometry=[bbox.geometry], crs=bbox.crs.pyproj_crs())
        if target_crs is not None:
            bbox_gdf = bbox_gdf.to_crs(target_crs)

        bbox_kwargs = {
            "color": "#00000000",
            "edgecolor": "red",
            "linestyle": "--",
            "zorder": 10**6,
            **self.config.bbox_kwargs,
        }

        for axis in axes.flatten():
            bbox_gdf.plot(ax=axis, **bbox_kwargs)

        return axes

    def _provide_axes(self, *, nrows: int, ncols: int, title: str | None = None, **subplot_kwargs: Any) -> np.ndarray:
        """Either provides an existing grid of axes or creates new one"""
        if self.axes is not None:
            return self.axes

        subplot_kwargs = {
            "squeeze": False,
            "tight_layout": True,
            **subplot_kwargs,
            **self.config.subplot_kwargs,  # Config kwargs override the ones above
        }
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(self.config.subplot_width * ncols, self.config.subplot_height * nrows),
            **subplot_kwargs,
        )
        if title and self.config.show_title:
            title_kwargs = {"t": title, "fontsize": 16, "y": 1.0, **self.config.title_kwargs}
            fig.suptitle(**title_kwargs)

        fig.subplots_adjust(wspace=0.06, hspace=0.06)

        return axes

    def _get_label_kwargs(self) -> dict[str, object]:
        """Provides `matplotlib` arguments for writing labels in plots."""
        return {"fontsize": 12, **self.config.label_kwargs}
