"""
This module implements visualizations for `EOPatch`

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Dict

import numpy as np
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt

from eolearn.core import FeatureType, EOPatch

from .eopatch_base import _BasePlotConfig, _BaseEOPatchVisualization


class PlotBackend(Enum):
    """Types of backend for plotting"""

    MATPLOTLIB = "matplotlib"
    HVPLOT = "hvplot"


def plot_eopatch(*args, backend: Union[PlotBackend, str] = PlotBackend.MATPLOTLIB, **kwargs):
    """The main `EOPatch` plotting function. It pr

    :param args: Positional arguments to be propagated to a plotting backend.
    :param backend: Which plotting backend to use.
    :param kwargs: Keyword arguments to be propagated to a plotting backend.
    :return: A grid of axes
    """
    backend = PlotBackend(backend)

    if backend is PlotBackend.MATPLOTLIB:
        return MatplotlibVisualization(*args, **kwargs).plot()

    if backend is PlotBackend.HVPLOT:
        # pylint: disable=import-outside-toplevel
        from .extra.hvplot import HvPlotVisualization

        return HvPlotVisualization(*args, **kwargs).plot()

    raise ValueError(f"EOPatch plotting backend {backend} is not supported")


@dataclass
class PlotConfig(_BasePlotConfig):
    """Advanced plotting configurations

    :param subplot_width: A width of each subplot in a grid
    :param subplot_height: A height of each subplot in a grid
    :param subplot_kwargs: A dictionary of parameters that will be passed to `matplotlib.pyplot.subplots` function.
    """

    subplot_width: Union[float, int] = 10
    subplot_height: Union[float, int] = 10
    subplot_kwargs: Dict[str, object] = field(default_factory=dict)


class MatplotlibVisualization(_BaseEOPatchVisualization):
    """EOPatch visualization using `matplotlib` framework."""

    def __init__(self, eopatch: EOPatch, feature, *, axes=None, config: Optional[PlotConfig] = None, **kwargs):
        """
        :param eopatch: An EOPatch with a feature to plot.
        :param feature: A feature from the given EOPatch to plot.
        :param axes: A grid of axes on which to write plots. If not provided it will create a new grid.
        :param config: A configuration object with advanced plotting parameters.
        :param kwargs: Parameters to be passed to the base class.
        """
        config = config or PlotConfig()
        super().__init__(eopatch, feature, config=config, **kwargs)

        self.axes = axes

    def plot(self):
        """Plots the given feature"""
        feature_type, feature_name = self.feature
        data = self.collect_and_prepare_feature()

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
            return self._plot_raster_grid(data, timestamps=self.eopatch.timestamp, title=feature_name)

        if feature_type.is_temporal():
            return self._plot_time_series(data, self.eopatch.timestamp, title=feature_name)
        return self._plot_series(data, title=feature_name)

    def _plot_raster_grid(self, raster, timestamps=None, title=None):
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

        for row_idx in range(rows):
            for column_idx in range(columns):
                axis = axes[row_idx][column_idx]
                raster_slice = raster[row_idx, ...] if self.rgb else raster[row_idx, ..., column_idx]
                axis.imshow(raster_slice)

                if timestamps and column_idx == 0:
                    axis.set_ylabel(timestamps[row_idx].isoformat(), fontsize=12)
                if self.channel_names:
                    axis.set_xlabel(self.channel_names[column_idx], fontsize=12)

        return axes

    def _plot_time_series(self, series, timestamps, title=None):
        """Plots time series feature."""
        axes = self._provide_axes(nrows=1, ncols=1, title=title)
        axis = axes[0][0]

        timestamp_array = np.array(timestamps)
        channel_num = series.shape[-1]
        for idx in range(channel_num):
            channel_label = self.channel_names[idx] if self.channel_names else None
            axis.plot(timestamp_array, series[..., idx], label=channel_label)

        if self.channel_names:
            axis.legend()
        return axes

    def _plot_series(self, series, title=None):
        """Plot a series of values."""
        axes = self._provide_axes(nrows=1, ncols=1, title=title)
        axis = axes[0][0]

        axis.plot(np.arange(series.size), series)

        return axes

    def _plot_vector_feature(self, dataframe, timestamp_column=None, title=None):
        """Plots a GeoDataFrame vector feature"""
        rows = len(dataframe[timestamp_column].unique()) if timestamp_column else 1
        axes = self._provide_axes(nrows=rows, ncols=1, title=title)

        self._plot_bbox(axes=axes, target_crs=dataframe.crs)

        if timestamp_column is None:
            dataframe.plot(ax=axes.flatten()[0])
            return axes

        timestamp_groups = dataframe.groupby(timestamp_column)
        timestamps = sorted(timestamp_groups.groups)

        for timestamp, axis in zip(timestamps, axes.flatten()):
            timestamp_groups.get_group(timestamp).plot(ax=axis)
            axis.set_ylabel(timestamp.isoformat(), fontsize=12)

        return axes

    def _plot_bbox(self, axes=None, target_crs=None):
        """Plot a bounding box"""
        bbox = self.eopatch.bbox
        if bbox is None:
            raise ValueError("EOPatch doesn't have a bounding box")

        if axes is None:
            axes = self._provide_axes(nrows=1, ncols=1, title="Bounding box")

        bbox_gdf = GeoDataFrame(geometry=[bbox.geometry], crs=bbox.crs.pyproj_crs())
        if target_crs is not None:
            bbox_gdf = bbox_gdf.to_crs(target_crs)

        for axis in axes.flatten():
            bbox_gdf.plot(ax=axis, color="#00000000", edgecolor="red", linestyle="--", zorder=10**6)

        return axes

    def _provide_axes(self, *, nrows: int, ncols: int, title: Optional[str] = None, **subplot_kwargs):
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
        if title:
            fig.suptitle(title, fontsize=16, y=1.0)

        fig.subplots_adjust(wspace=0.06, hspace=0.06)

        return axes
