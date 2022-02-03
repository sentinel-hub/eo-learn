"""
This module implements dynamic visualizations for EOPatch

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import dataclasses
from typing import Optional, cast

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

try:
    import xarray as xr
    import holoviews as hv
    import geoviews as gv
    import hvplot  # pylint: disable=unused-import
    import hvplot.xarray  # pylint: disable=unused-import
    import hvplot.pandas  # pylint: disable=unused-import
    from cartopy import crs as ccrs
except ImportError as exception:
    raise ImportError(
        "This module requires an installation of dynamic plotting package extension. It can be installed with:\n"
        "pip install eo-learn-visualization[HVPLOT]"
    ) from exception

from sentinelhub import CRS
from eolearn.core import EOPatch, FeatureType, FeatureTypeSet
from eolearn.core.utils.parsing import parse_feature

from .xarray import array_to_dataframe, get_new_coordinates, string_to_variable
from ..eopatch_base import _BasePlotConfig, _BaseEOPatchVisualization


@dataclasses.dataclass
class HvPlotConfig(_BasePlotConfig):
    """Additional advanced configurations for `hvplot` visualization.

    :param plot_width: A width of the plot.
    :param plot_height: A height of the plot.
    :param plot_per_pixel: Whether plot data for each pixel (line), for `FeatureType.DATA` and `FeatureType.MASK`.
    :param vdims: Value dimensions for plotting a `GeoDataFrame`.
    """

    plot_width: int = 800
    plot_height: int = 500
    plot_per_pixel: bool = False
    vdims: Optional[str] = None


class HvPlotVisualization(_BaseEOPatchVisualization):
    """EOPatch visualization using `HvPlot` framework."""

    def __init__(
        self, eopatch: EOPatch, feature, *, mask_feature=None, config: Optional[HvPlotConfig] = None, **kwargs
    ):
        """
        :param eopatch: An EOPatch with a feature to plot.
        :param feature: A feature from the given EOPatch to plot.
        :param mask_feature: A mask feature to be applied as a mask to the feature that is being plotted
        """
        config = config or HvPlotConfig()
        super().__init__(eopatch, feature, config=config, **kwargs)
        self.config = cast(HvPlotConfig, self.config)

        self.mask_feature = parse_feature(mask_feature)

    def plot(self):
        """Creates a `hvplot` of the feature from the given `EOPatch`."""
        feature_type, feature_name = self.feature
        if self.config.plot_per_pixel and feature_type in FeatureTypeSet.RASTER_TYPES_4D:
            vis = self._plot_pixel(feature_type, feature_name)
        elif feature_type in (FeatureType.MASK, *FeatureTypeSet.RASTER_TYPES_3D):
            vis = self._plot_raster(feature_type, feature_name)
        elif feature_type is FeatureType.DATA:
            vis = self._plot_data(feature_name)
        elif feature_type is FeatureType.VECTOR:
            vis = self._plot_vector(feature_name)
        elif feature_type is FeatureType.VECTOR_TIMELESS:
            vis = self._plot_vector_timeless(feature_name)
        else:
            vis = self._plot_scalar_label(feature_type, feature_name)

        return vis.opts(plot=dict(width=self.config.plot_width, height=self.config.plot_height))

    def _plot_data(self, feature_name):
        """Plots the FeatureType.DATA of eopatch."""
        crs = self.eopatch.bbox.crs
        crs = CRS.POP_WEB if crs is CRS.WGS84 else crs
        data_da = array_to_dataframe(self.eopatch, (FeatureType.DATA, feature_name), crs=crs)
        if self.mask_feature:
            data_da = self._mask_data(data_da)
        timestamps = self.eopatch.timestamp
        crs = self.eopatch.bbox.crs
        if not self.rgb:
            return data_da.hvplot(x="x", y="y", crs=ccrs.epsg(crs.epsg))
        data_rgb = self._eopatch_da_to_rgb(data_da, feature_name, crs)
        rgb_dict = {timestamp_: self._plot_rgb_one(data_rgb, timestamp_) for timestamp_ in timestamps}

        return hv.HoloMap(rgb_dict, kdims=["time"])

    @staticmethod
    def _plot_rgb_one(eopatch_da, timestamp):
        """Returns visualization for one timestamp for FeatureType.DATA

        :param eopatch_da: eopatch converted to xarray DataArray
        :type eopatch_da: xarray DataArray
        :param timestamp: timestamp to make plot for
        :type timestamp: datetime
        """
        return eopatch_da.sel(time=timestamp).drop("time").hvplot(x="x", y="y")

    def _plot_raster(self, feature_type, feature_name):
        """Makes visualization for raster data (except for FeatureType.DATA)

        :param feature_type: type of eopatch feature
        :type feature_type: FeatureType
        :param feature_name: name of eopatch feature
        :type feature_name: str
        """
        crs = self.eopatch.bbox.crs
        crs = CRS.POP_WEB if crs is CRS.WGS84 else crs
        data_da = array_to_dataframe(self.eopatch, (feature_type, feature_name), crs=crs)
        data_min = data_da.values.min()
        data_max = data_da.values.max()
        data_levels = len(np.unique(data_da))
        data_levels = 11 if data_levels > 11 else data_levels
        data_da = data_da.where(data_da > 0).fillna(-1)
        vis = data_da.hvplot(x="x", y="y", crs=ccrs.epsg(crs.epsg)).opts(
            clim=(data_min, data_max), clipping_colors={"min": "transparent"}, color_levels=data_levels
        )
        return vis

    def _plot_vector(self, feature_name):
        """A visualization for vector feature

        :param feature_name: name of eopatch feature
        :type feature_name: str
        """
        crs = self.eopatch.bbox.crs
        timestamps = self.eopatch.timestamp
        data_gpd = self._fill_vector(FeatureType.VECTOR, feature_name)
        if crs is CRS.WGS84:
            crs = CRS.POP_WEB
            data_gpd = data_gpd.to_crs(crs.pyproj_crs())
        shapes_dict = {timestamp_: self._plot_shapes_one(data_gpd, timestamp_, crs) for timestamp_ in timestamps}
        return hv.HoloMap(shapes_dict, kdims=["time"])

    def _fill_vector(self, feature_type, feature_name):
        """Adds timestamps from eopatch to GeoDataFrame.

        :param feature_type: type of eopatch feature
        :type feature_type: FeatureType
        :param feature_name: name of eopatch feature
        :type feature_name: str
        :return: GeoDataFrame with added data
        :rtype: geopandas.GeoDataFrame
        """
        vector = self.eopatch[feature_type][feature_name].copy()
        vector["valid"] = True
        eopatch_timestamps = self.eopatch.timestamp
        vector_timestamps = set(vector[self.config.timestamp_column])
        blank_timestamps = [timestamp for timestamp in eopatch_timestamps if timestamp not in vector_timestamps]
        dummy_geometry = self._create_dummy_polygon(0.0000001)

        temp_df = self._create_dummy_dataframe(vector, blank_timestamps=blank_timestamps, dummy_geometry=dummy_geometry)

        final_vector = gpd.GeoDataFrame(pd.concat((vector, temp_df), ignore_index=True), crs=vector.crs)
        return final_vector

    def _create_dummy_dataframe(self, geodataframe, blank_timestamps, dummy_geometry, fill_str="", fill_numeric=1):
        """Creates a `GeoDataFrame` to fill with dummy data (for visualization)

        :param geodataframe: dataframe to append rows to
        :type geodataframe: geopandas.GeoDataFrame
        :param blank_timestamps: timestamps for constructing dataframe
        :type blank_timestamps: list of timestamps
        :param dummy_geometry: geometry to plot when there is no data
        :type dummy_geometry: shapely.geometry.Polygon
        :param fill_str: insert when there is no value in str column
        :type fill_str: str
        :param fill_numeric: insert when
        :type fill_numeric: float
        :return: dataframe with dummy data
        :rtype: geopandas.GeoDataFrame
        """
        dataframe = pd.DataFrame(data=blank_timestamps, columns=[self.config.timestamp_column])

        for column in geodataframe.columns:
            if column == self.config.timestamp_column:
                continue
            if column == self.config.geometry_column:
                dataframe[column] = dummy_geometry
            elif column == "valid":
                dataframe[column] = False
            elif geodataframe[column].dtype in (int, float):
                dataframe[column] = fill_numeric
            else:
                dataframe[column] = fill_str

        return dataframe

    def _create_dummy_polygon(self, addition_factor):
        """Creates geometry/polygon to plot if there is no data (at timestamp)

        :param addition_factor: size of the 'blank polygon'
        :type addition_factor: float
        :return: polygon
        :rtype: shapely.geometry.Polygon
        """
        x_blank, y_blank = self.eopatch.bbox.lower_left
        dummy_geometry = Polygon(
            [
                [x_blank, y_blank],
                [x_blank + addition_factor, y_blank],
                [x_blank + addition_factor, y_blank + addition_factor],
                [x_blank, y_blank + addition_factor],
            ]
        )

        return dummy_geometry

    def _plot_scalar_label(self, feature_type, feature_name):
        """Line plot for FeatureType.SCALAR, FeatureType.LABEL

        :param feature_type: type of eopatch feature
        :type feature_type: FeatureType
        :param feature_name: name of eopatch feature
        :type feature_name: str
        """
        data_da = array_to_dataframe(self.eopatch, (feature_type, feature_name))
        return data_da.hvplot()

    def _plot_shapes_one(self, data_gpd, timestamp, crs):
        """Plots shapes for one timestamp from geopandas GeoDataFrame

        :param data_gpd: data to plot
        :type data_gpd: geopandas.GeoDataFrame
        :param timestamp: timestamp to plot data for
        :type timestamp: datetime
        :param crs: in which crs is the data to plot
        :type crs: sentinelhub.crs
        """
        out = data_gpd.loc[data_gpd[self.config.timestamp_column] == timestamp]
        return gv.Polygons(out, crs=ccrs.epsg(int(crs.value)))

    def _plot_vector_timeless(self, feature_name):
        """Plot FeatureType.VECTOR_TIMELESS data

        :param feature_name: name of the eopatch feature
        :type feature_name: str
        """
        crs = self.eopatch.bbox.crs
        data_gpd = self.eopatch[FeatureType.VECTOR_TIMELESS][feature_name]
        if crs is CRS.WGS84:
            crs = CRS.POP_WEB
            data_gpd = data_gpd.to_crs(crs.pyproj_crs())

        return gv.Polygons(data_gpd, crs=ccrs.epsg(crs.epsg), vdims=self.config.vdims)

    def _eopatch_da_to_rgb(self, eopatch_da, feature_name, crs):
        """Creates new xarray DataArray (from old one) to plot rgb image with hv.Holomap

        :param eopatch_da: eopatch DataArray
        :type eopatch_da: DataArray
        :param feature_name: name of the feature to plot
        :type feature_name:  str
        :param crs: in which crs are the data
        :type crs: sentinelhub.constants.crs
        :return: eopatch DataArray with proper coordinates, dimensions, crs
        :rtype: xarray.DataArray
        """
        timestamps = eopatch_da.coords["time"].values
        bands = eopatch_da[..., self.rgb] * self.config.rgb_factor
        bands = bands.rename({string_to_variable(feature_name, "_dim"): "band"}).transpose("time", "band", "y", "x")
        x_values, y_values = get_new_coordinates(eopatch_da, crs, CRS.POP_WEB)
        eopatch_rgb = xr.DataArray(
            data=np.clip(bands.data, 0, 1),
            coords={"time": timestamps, "band": self.rgb, "y": np.flip(y_values), "x": x_values},
            dims=("time", "band", "y", "x"),
        )
        return eopatch_rgb

    def _plot_pixel(self, feature_type, feature_name):
        """
        Plots one pixel through time
        :return: visualization
        :rtype: holoviews
        """
        data_da = array_to_dataframe(self.eopatch, (feature_type, feature_name))
        if self.mask_feature:
            data_da = self._mask_data(data_da)
        return data_da.hvplot(x="time")

    def _mask_data(self, data_da):
        """
        Creates a copy of array and insert 0 where data is masked.
        :param data_da: DataArray
        :type data_da: xarray.DataArray
        :return: dataaray
        :rtype: xarray.DataArray
        """
        mask = self.eopatch[self.mask_feature]
        if len(data_da.values.shape) == 4:
            mask = np.repeat(mask, data_da.values.shape[-1], -1)
        else:
            mask = np.squeeze(mask, axis=-1)
        data_da = data_da.copy()
        # pylint: disable=invalid-unary-operand-type
        data_da.values[~mask] = 0

        return data_da
