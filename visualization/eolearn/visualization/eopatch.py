"""
This module implements visualizations for `EOPatch`

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import datetime as dt
import itertools as it
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS

from eolearn.core import EOPatch, FeatureType
from eolearn.core.types import SingleFeatureSpec

from .eopatch_base import BaseEOPatchVisualization, BasePlotConfig


class PlotBackend(Enum):
    """Types of backend for plotting"""

    MATPLOTLIB = "matplotlib"


def plot_eopatch(*args: Any, backend: Union[PlotBackend, str] = PlotBackend.MATPLOTLIB, **kwargs: Any) -> object:
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
class PlotConfig(BasePlotConfig):
    """Advanced plotting configurations

    :param subplot_width: A width of each subplot in a grid
    :param subplot_height: A height of each subplot in a grid
    :param subplot_kwargs: A dictionary of parameters that will be passed to `matplotlib.pyplot.subplots` function.
    :param show_title: A flag to specify if plot title should be shown.
    :param title_kwargs: A dictionary of parameters that will be passed to `matplotlib.figure.Figure.suptitle`.
    :param label_kwargs: A dictionary of parameters that will be passed to `matplotlib` methods for setting axes labels.
    :param bbox_kwargs: A dictionary of parameters that will be passed to `GeoDataFrame.plot` when plotting a bounding
        box.
    """

    subplot_width: Union[float, int] = 8
    subplot_height: Union[float, int] = 8
    interpolation: str = "none"
    subplot_kwargs: Dict[str, object] = field(default_factory=dict)
    show_title: bool = True
    title_kwargs: Dict[str, object] = field(default_factory=dict)
    label_kwargs: Dict[str, object] = field(default_factory=dict)
    bbox_kwargs: Dict[str, object] = field(default_factory=dict)


class MatplotlibVisualization(BaseEOPatchVisualization):
    """EOPatch visualization using `matplotlib` framework."""

    config: PlotConfig

    def __init__(
        self,
        eopatch: EOPatch,
        feature: SingleFeatureSpec,
        *,
        axes: Optional[np.ndarray] = None,
        config: Optional[PlotConfig] = None,
        **kwargs: Any,
    ):
        """
        :param eopatch: An EOPatch with a feature to plot.
        :param feature: A feature from the given EOPatch to plot.
        :param axes: A grid of axes on which to write plots. If not provided it will create a new grid.
        :param config: A configuration object with advanced plotting parameters.
        :param kwargs: Parameters to be passed to the base class.
        """
        config = config or PlotConfig()
        super().__init__(eopatch, feature, config=config, **kwargs)

        if axes is not None and not isinstance(axes, np.ndarray):
            axes = np.array([np.array([axes])])  # type: ignore[unreachable]
        self.axes = axes

    def plot(self) -> np.ndarray:
        """Plots the given feature"""
        feature_type, feature_name = self.feature
        data, timestamps = self.collect_and_prepare_feature()

        if feature_type is FeatureType.BBOX:
            return self._plot_bbox()

        if feature_type.is_vector():
            return self._plot_vector_feature(
                data,
                timestamp_column=self.config.timestamp_column if feature_type.is_temporal() else None,
                title=feature_name,
            )

        if not feature_type.is_raster():
            raise ValueError(f"Plotting of {feature_type} is not supported")

        if feature_type.is_spatial():
            if feature_type.is_timeless():
                return self._plot_raster_grid(data[np.newaxis, ...], title=feature_name)
            return self._plot_raster_grid(data, timestamps=timestamps, title=feature_name)

        if feature_type.is_timeless():
            return self._plot_bar(data, title=feature_name)
        return self._plot_time_series(data, timestamps=timestamps, title=feature_name)

    def _plot_raster_grid(
        self, raster: np.ndarray, timestamps: Optional[List[dt.datetime]] = None, title: Optional[str] = None
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
        self, series: np.ndarray, timestamps: Optional[List[dt.datetime]] = None, title: Optional[str] = None
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

    def _plot_bar(self, values: np.ndarray, title: Optional[str] = None) -> np.ndarray:
        """Make a bar plot from values."""
        axes = self._provide_axes(nrows=1, ncols=1, title=title)
        axis = axes.flatten()[0]

        xlabels = np.array(self.channel_names) if self.channel_names else np.arange(values.size)
        axis.bar(xlabels, values)

        return axes

    def _plot_vector_feature(
        self, dataframe: GeoDataFrame, timestamp_column: Optional[str] = None, title: Optional[str] = None
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

    def _plot_bbox(self, axes: Optional[np.ndarray] = None, target_crs: Optional[CRS] = None) -> np.ndarray:
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

    def _provide_axes(
        self, *, nrows: int, ncols: int, title: Optional[str] = None, **subplot_kwargs: Any
    ) -> np.ndarray:
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

    def _get_label_kwargs(self) -> Dict[str, object]:
        """Provides `matplotlib` arguments for writing labels in plots."""
        return {"fontsize": 12, **self.config.label_kwargs}
