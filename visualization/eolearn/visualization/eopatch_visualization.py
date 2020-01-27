"""
This module implements visualizations for EOPatch

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import holoviews as hv
import geoviews as gv

import hvplot         # pylint: disable=unused-import
import hvplot.xarray  # pylint: disable=unused-import
import hvplot.pandas  # pylint: disable=unused-import

from cartopy import crs as ccrs
from shapely.geometry import Polygon

from sentinelhub import CRS

from eolearn.core import FeatureType, FeatureTypeSet, FeatureParser

from .xarray_utils import array_to_dataframe, new_coordinates, string_to_variable

PLOT_WIDTH = 800
PLOT_HEIGHT = 500


class EOPatchVisualization:
    """
    Plot class for making visulizations.

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param rgb: bands for creating RGB image
    :type rgb: [int, int, int]
    :param rgb_factor: multiplication factor for constructing rgb image
    :type rgb_factor: float
    :param vdims: value dimensions for plotting geopandas.GeoDataFrame
    :type vdims: str
    :param timestamp_column: geopandas.GeoDataFrame columns with timestamps
    :type timestamp_column: str
    :param geometry_column: geopandas.GeoDataFrame columns with geometry
    :type geometry_column: geometry
    :param pixel: wheather plot data for each pixel (line), for FeatureType.DATA and FeatureType.MASK
    :type pixel: bool
    :param mask: name of the FeatureType.MASK to apply to data
    :type mask: str

    """
    def __init__(self, eopatch, feature, rgb=None, rgb_factor=3.5, vdims=None,
                 timestamp_column='TIMESTAMP', geometry_column='geometry', pixel=False, mask=None):
        self.eopatch = eopatch
        self.feature = feature
        self.rgb = rgb
        self.rgb_factor = rgb_factor
        self.vdims = vdims
        self.timestamp_column = timestamp_column
        self.geometry_column = geometry_column
        self.pixel = pixel
        self.mask = mask

    def plot(self):
        """ Plots eopatch

        :return: plot
        :rtype: holovies/bokeh
        """

        features = list(FeatureParser(self.feature))
        feature_type, feature_name = features[0]
        if self.pixel and feature_type in FeatureTypeSet.RASTER_TYPES_4D:
            vis = self.plot_pixel(feature_type, feature_name)
        elif feature_type in (FeatureType.MASK, *FeatureTypeSet.RASTER_TYPES_3D):
            vis = self.plot_raster(feature_type, feature_name)
        elif feature_type is FeatureType.DATA:
            vis = self.plot_data(feature_name)
        elif feature_type is FeatureType.VECTOR:
            vis = self.plot_vector(feature_name)
        elif feature_type is FeatureType.VECTOR_TIMELESS:
            vis = self.plot_vector_timeless(feature_name)
        else:      # elif feature_type in (FeatureType.SCALAR, FeatureType.LABEL):
            vis = self.plot_scalar_label(feature_type, feature_name)

        return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))

    def plot_data(self, feature_name):
        """ Plots the FeatureType.DATA of eopatch.

        :param feature_name: name of the eopatch feature
        :type feature_name: str
        :return: visualization
        :rtype: holoview/geoviews/bokeh
        """
        crs = self.eopatch.bbox.crs
        crs = CRS.POP_WEB if crs is CRS.WGS84 else crs
        data_da = array_to_dataframe(self.eopatch, (FeatureType.DATA, feature_name), crs=crs)
        if self.mask:
            data_da = self.mask_data(data_da)
        timestamps = self.eopatch.timestamp
        crs = self.eopatch.bbox.crs
        if not self.rgb:
            return data_da.hvplot(x='x', y='y', crs=ccrs.epsg(int(crs.value)))
        data_rgb = self.eopatch_da_to_rgb(data_da, feature_name, crs)
        rgb_dict = {timestamp_: self.plot_rgb_one(data_rgb, timestamp_) for timestamp_ in timestamps}

        return hv.HoloMap(rgb_dict, kdims=['time'])

    @staticmethod
    def plot_rgb_one(eopatch_da, timestamp):  # OK
        """ Returns visualization for one timestamp for FeatureType.DATA
        :param eopatch_da: eopatch converted to xarray DataArray
        :type eopatch_da: xarray DataArray
        :param timestamp: timestamp to make plot for
        :type timestamp: datetime
        :return: visualization
        :rtype:  holoviews/geoviews/bokeh
        """
        return eopatch_da.sel(time=timestamp).drop('time').hvplot(x='x', y='y')

    def plot_raster(self, feature_type, feature_name):
        """ Makes visualization for raster data (except for FeatureType.DATA)

        :param feature_type: type of eopatch feature
        :type feature_type: FeatureType
        :param feature_name: name of eopatch feature
        :type feature_name: str
        :return: visualization
        :rtype: holoviews/geoviews/bokeh
        """
        crs = self.eopatch.bbox.crs
        crs = CRS.POP_WEB if crs is CRS.WGS84 else crs
        data_da = array_to_dataframe(self.eopatch, (feature_type, feature_name), crs=crs)
        data_min = data_da.values.min()
        data_max = data_da.values.max()
        data_levels = len(np.unique(data_da))
        data_levels = 11 if data_levels > 11 else data_levels
        data_da = data_da.where(data_da > 0).fillna(-1)
        vis = data_da.hvplot(x='x', y='y',
                             crs=ccrs.epsg(int(crs.value))).opts(clim=(data_min, data_max),
                                                                 clipping_colors={'min': 'transparent'},
                                                                 color_levels=data_levels)
        return vis

    def plot_vector(self, feature_name):
        """ Visualizaton for vector (FeatureType.VECTOR) data

        :param feature_name: name of eopatch feature
        :type feature_name: str
        :return: visualization
        :rtype: holoviews/geoviews/bokeh

        """
        crs = self.eopatch.bbox.crs
        timestamps = self.eopatch.timestamp
        data_gpd = self.fill_vector(FeatureType.VECTOR, feature_name)
        if crs is CRS.WGS84:
            crs = CRS.POP_WEB
            data_gpd = data_gpd.to_crs({'init': 'epsg:{}'.format(crs.value)})
        shapes_dict = {timestamp_: self.plot_shapes_one(data_gpd, timestamp_, crs)
                       for timestamp_ in timestamps}
        return hv.HoloMap(shapes_dict, kdims=['time'])

    def fill_vector(self, feature_type, feature_name):
        """ Adds timestamps from eopatch to GeoDataFrame.

        :param feature_type: type of eopatch feature
        :type feature_type: FeatureType
        :param feature_name: name of eopatch feature
        :type feature_name: str
        :return: GeoDataFrame with added data
        :rtype: geopandas.GeoDataFrame
        """
        vector = self.eopatch[feature_type][feature_name].copy()
        vector['valid'] = True
        eopatch_timestamps = self.eopatch.timestamp
        vector_timestamps = set(vector[self.timestamp_column])
        blank_timestamps = [timestamp for timestamp in eopatch_timestamps if timestamp not in vector_timestamps]
        dummy_geometry = self.create_dummy_polygon(0.0000001)

        temp_df = self.create_dummy_dataframe(vector,
                                              blank_timestamps=blank_timestamps,
                                              dummy_geometry=dummy_geometry)

        final_vector = gpd.GeoDataFrame(pd.concat((vector, temp_df), ignore_index=True),
                                        crs=vector.crs)
        return final_vector

    def create_dummy_dataframe(self, geodataframe, blank_timestamps, dummy_geometry,
                               fill_str='', fill_numeric=1):
        """ Creates geopadnas GeoDataFrame to fill with dummy data (for visualization)

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
        dataframe = pd.DataFrame(data=blank_timestamps, columns=[self.timestamp_column])

        for column in geodataframe.columns:
            if column == self.timestamp_column:
                continue

            if column == self.geometry_column:
                dataframe[column] = dummy_geometry
            elif column == 'valid':
                dataframe[column] = False
            elif geodataframe[column].dtype in (int, float):
                dataframe[column] = fill_numeric
            else:
                dataframe[column] = fill_str

        return dataframe

    def create_dummy_polygon(self, addition_factor):
        """ Creates geometry/polygon to plot if there is no data (at timestamp)

        :param addition_factor: size of the 'blank polygon'
        :type addition_factor: float
        :return: polygon
        :rtype: shapely.geometry.Polygon
        """
        x_blank, y_blank = self.eopatch.bbox.lower_left
        dummy_geometry = Polygon([[x_blank, y_blank],
                                  [x_blank + addition_factor, y_blank],
                                  [x_blank + addition_factor, y_blank + addition_factor],
                                  [x_blank, y_blank + addition_factor]])

        return dummy_geometry

    def plot_scalar_label(self, feature_type, feature_name):
        """ Line plot for FeatureType.SCALAR, FeatureType.LABEL

        :param feature_type: type of eopatch feature
        :type feature_type: FeatureType
        :param feature_name: name of eopatch feature
        :type feature_name: str
        :return: visualization
        :rtype: holoviews/geoviews/bokeh
        """
        data_da = array_to_dataframe(self.eopatch, (feature_type, feature_name))
        if data_da.dtype == np.bool:
            data_da = data_da.astype(np.int8)
        return data_da.hvplot()

    def plot_shapes_one(self, data_gpd, timestamp, crs):
        """ Plots shapes for one timestamp from geopandas GeoDataFrame

        :param data_gpd: data to plot
        :type data_gpd: geopandas.GeoDataFrame
        :param timestamp: timestamp to plot data for
        :type timestamp: datetime
        :param crs: in which crs is the data to plot
        :type crs: sentinelhub.crs
        :return: visualization
        :rtype: geoviews
        """
        out = data_gpd.loc[data_gpd[self.timestamp_column] == timestamp]
        return gv.Polygons(out, crs=ccrs.epsg(int(crs.value)))

    def plot_vector_timeless(self, feature_name):
        """ Plot FeatureType.VECTOR_TIMELESS data

        :param feature_name: name of the eopatch featrue
        :type feature_name: str
        :return: visalization
        :rtype: geoviews
        """
        crs = self.eopatch.bbox.crs
        if crs is CRS.WGS84:
            crs = CRS.POP_WEB
            data_gpd = self.eopatch[FeatureType.VECTOR_TIMELESS][feature_name].to_crs(
                {'init': 'epsg:{}'.format(crs.value)})
        else:
            data_gpd = self.eopatch[FeatureType.VECTOR_TIMELESS][feature_name]
        return gv.Polygons(data_gpd, crs=ccrs.epsg(int(crs.value)), vdims=self.vdims)

    def eopatch_da_to_rgb(self, eopatch_da, feature_name, crs):
        """ Creates new xarray DataArray (from old one) to plot rgb image with hv.Holomap

        :param eopatch_da: eopatch DataArray
        :type eopatch_da: DataArray
        :param feature_name: name of the feature to plot
        :type feature_name:  str
        :param crs: in which crs are the data
        :type crs: sentinelhub.constants.crs
        :return: eopatch DataArray with proper coordinates, dimensions, crs
        :rtype: xarray.DataArray
        """
        timestamps = eopatch_da.coords['time'].values
        bands = eopatch_da[..., self.rgb] * self.rgb_factor
        bands = bands.rename({string_to_variable(feature_name, '_dim'): 'band'}).transpose('time', 'band', 'y', 'x')
        x_values, y_values = new_coordinates(eopatch_da, crs, CRS.POP_WEB)
        eopatch_rgb = xr.DataArray(data=np.clip(bands.data, 0, 1),
                                   coords={'time': timestamps,
                                           'band': self.rgb,
                                           'y': np.flip(y_values),
                                           'x': x_values},
                                   dims=('time', 'band', 'y', 'x'))
        return eopatch_rgb

    def plot_pixel(self, feature_type, feature_name):
        """
        Plots one pixel through time
        :return: visualization
        :rtype: holoviews
        """
        data_da = array_to_dataframe(self.eopatch, (feature_type, feature_name))
        if self.mask:
            data_da = self.mask_data(data_da)
        if data_da.dtype == np.bool:
            data_da = data_da.astype(np.int8)
        return data_da.hvplot(x='time')

    def mask_data(self, data_da):
        """
        Creates a copy of array and insert 0 where data is masked.
        :param data_da: dataarray
        :type data_da: xarray.DataArray
        :return: dataaray
        :rtype: xarray.DataArray
        """
        mask = self.eopatch[FeatureType.MASK][self.mask]
        if len(data_da.values.shape) == 4:
            mask = np.repeat(mask, data_da.values.shape[-1], -1)
        else:
            mask = np.squeeze(mask, axis=-1)
        data_da = data_da.copy()
        data_da.values[~mask] = 0

        return data_da
