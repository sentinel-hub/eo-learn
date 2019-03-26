"""
This module implements visualizations for EOPatch (with conversion to xarray DataArray/Dataset)
"""

import numpy as np
import geoviews as gv
import xarray as xr
import holoviews as hv
import pandas as pd
import geopandas as gpd

import hvplot         # pylint: disable=unused-import
import hvplot.xarray  # pylint: disable=unused-import
import hvplot.pandas  # pylint: disable=unused-import

from cartopy import crs as ccrs
from shapely import geometry

from sentinelhub import BBox, CRS

from .constants import FeatureType
from .utilities import FeatureParser

PLOT_WIDTH = 800
PLOT_HEIGHT = 500


def get_spatial_coordinates(bbox, data, feature_type):
    """ Returns spatial coordinates (dictionary) for creating xarray DataArray/Dataset

    :param bbox: eopatch bbox
    :type bbox: EOPatch BBox
    :param data: values for calculating number of coordinates
    :type data: numpy array
    :param feature_type: type of the feature
    :type feature_type: FeatureType

    :return: spatial coordinates
    :rtype: dict {'x':, 'y':}
    """
    index_x = 2
    index_y = 1
    if feature_type in (FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS):
        index_x = 1
        index_y = 0
    pixel_width = (bbox.max_x - bbox.min_x)/data.shape[index_x]
    pixel_height = (bbox.max_y - bbox.min_y)/data.shape[index_y]
    coordinates = {
        'x': np.linspace(bbox.min_x+pixel_width/2, bbox.max_x-pixel_width/2, data.shape[index_x]),
        'y': np.linspace(bbox.max_y-pixel_height/2, bbox.min_y+pixel_height/2, data.shape[index_y])
    }

    return coordinates


def get_temporal_coordinates(timestamps):
    """ Returns temporal coordinates dictionary for creating xarray DataArray/Dataset

    :param timestamps: timestamps
    :type timestamps: EOpatch.timestamp

    :return: temporal coordinates
    :rtype: dict {'time': }
    """
    coordinates = {
        'time': timestamps
    }

    return coordinates


def get_depth_coordinates(feature_name, data, names_of_channels=None):
    """ Returns band/channel/dept coordinates for xarray DataArray/Dataset

    :param feature_name: name of feature of EOPatch
    :type feature_name: FeatureType
    :param data: data of EOPatch
    :type data: numpy.array
    :param names_of_channels: coordinates for the last (band/dept/chanel) dimension
    :type names_of_channels: list
    :return: depth/band coordinates
    :rtype: dict
    """
    coordinates = {}
    depth = feature_name.replace('-', '_')+'_dim'
    if names_of_channels:
        coordinates[depth] = names_of_channels
    elif isinstance(data, np.ndarray):
        coordinates[depth] = np.arange(data.shape[-1])

    return coordinates


def get_coordinates(eopatch, feature, epsg_number):
    """ Creates coordinates for xarray DataArray

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param epsg_number: convert spatial coordinates to crs epsg:epsg_number
    :type epsg_number: int
    :return: coordinates for xarry DataArray/Dataset
    :rtype: dict
    """

    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0][0], features[0][1]
    original_epsg_number = eopatch.bbox.crs.ogc_string().split(':')[1]
    if epsg_number and original_epsg_number != epsg_number:
        bbox = eopatch.bbox.transform(CRS(epsg_number))
    else:
        bbox = eopatch.bbox
    data = eopatch[feature_type][feature_name]
    timestamps = eopatch.timestamp

    if feature_type in (FeatureType.DATA, FeatureType.MASK):
        coordinates = {**get_temporal_coordinates(timestamps),
                       **get_spatial_coordinates(bbox, data, feature_type),
                       **get_depth_coordinates(data=data, feature_name=feature_name)}
    elif feature_type in (FeatureType.SCALAR, FeatureType.LABEL):
        coordinates = {**get_temporal_coordinates(timestamps),
                       **get_depth_coordinates(data=data, feature_name=feature_name)}
    elif feature_type in (FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS):
        coordinates = {**get_spatial_coordinates(bbox, data, feature_type),
                       **get_depth_coordinates(data=data, feature_name=feature_name)}
    else:      # elif feature_type in (FeatureType.SCALAR_TIMELESS, FeatureType.LABEL_TIMELESS):
        coordinates = get_depth_coordinates(data=data, feature_name=feature_name)

    return coordinates


def get_dimensions(feature):
    """ Returns list of dimensions for xarray DataArray/Dataset

    :param feature: eopatch feature
    :type feature: (FeatureType, str)
    :return: dimensions for xarray DataArray/Dataset
    :rtype: list(str)
    """
    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0][0], features[0][1]
    depth = feature_name.replace('-', '_') + "_dim"
    if feature_type in (FeatureType.DATA, FeatureType.MASK):
        dimensions = ['time', 'y', 'x', depth]
    elif feature_type in (FeatureType.SCALAR, FeatureType.LABEL):
        dimensions = ['time', depth]
    elif feature_type in (FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS):
        dimensions = ['y', 'x', depth]
    else:      # elif feature_type in (FeatureType.SCALAR_TIMELESS, FeatureType.LABEL_TIMELESS):
        dimensions = [depth]

    return dimensions


def array_to_dataframe(eopatch, feature, remove_depth=True, epsg_number=None):
    """ Converts one feature of eopathc to xarray DataArray

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param remove_depth: removes last dimension if it is one
    :type remove_depth: bool
    :param epsg_number: converts dimensions to epsg:epsg_number crs
    :type epsg_number: int
    :return: dataarray
    :rtype: xarray DataArray
    """
    features = list(FeatureParser(feature))
    feature_type = features[0][0]
    feature_name = features[0][1]
    bbox = eopatch.bbox
    data = eopatch[feature_type][feature_name]
    if isinstance(data, xr.DataArray):
        data = data.values
    dimensions = get_dimensions(feature)
    coordinates = get_coordinates(eopatch, feature, epsg_number=epsg_number)
    dataframe = xr.DataArray(data=data,
                             coords=coordinates,
                             dims=dimensions,
                             attrs={'crs': str(bbox.crs),
                                    'feature_type': feature_type,
                                    'feature_name': feature_name},
                             name=feature_name.replace('-', '_'))
    if remove_depth and dataframe.values.shape[-1] == 1:
        dataframe = dataframe.squeeze()
        dataframe = dataframe.drop(feature_name + '_dim')

    return dataframe


def eopatch_to_dataset(eopatch, remove_depth=True):
    """
    Converts eopatch to xarray Dataset

    :param eopatch: eopathc
    :type eopatch: EOPatch
    :param remove_depth: removes last dimension if it is one
    :type remove_depth: bool
    :return: dataset
    :rtype: xarray Dataset
    """
    dataset = xr.Dataset()
    for feature in eopatch.get_feature_list():
        if not isinstance(feature, tuple):
            continue
        feature_type = feature[0]
        feature_name = feature[1]
        if feature_type not in (FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS, FeatureType.META_INFO):
            dataframe = array_to_dataframe(eopatch, (feature_type, feature_name), remove_depth)
            dataset[feature_name] = dataframe

    return dataset


def new_coordinates(data, crs, new_crs):
    """ Returns coordinates for data in new crs.

    :param data: data for converting coordinates for
    :type data: xarray.DataArray or xarray.Dataset
    :param crs: old crs
    :type crs: BBox.crs
    :param new_crs: new crs
    :type new_crs: BBox.crs
    :return: new x and y coordinates
    :rtype: (float, float)
    """
    x_values = data.coords['x'].values
    y_values = data.coords['y'].values
    bbox = (x_values[0], y_values[0], x_values[-1], y_values[-1])
    bbox = BBox(bbox, crs=crs)
    bbox = bbox.transform(new_crs)
    xmin, ymin = bbox.get_lower_left()
    xmax, ymax = bbox.get_upper_right()
    new_xs = np.linspace(xmin, xmax, len(x_values))
    new_ys = np.linspace(ymin, ymax, len(y_values))

    return new_xs, new_ys


def plot(eopatch, feature, alpha=1, rgb=None, rgb_factor=3.5, vdims=None,
         timestamp_column='TIMESTAMP', geometry_column='geometry'):
    """ Plots eopatch

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param alpha: opacity parameter
    :type alpha: float
    :param rgb: indexes of bands to create rgb image from
    :type rgb: [int, int, int]
    :param rgb_factor: factor for rgb bands multiplication
    :type rgb_factor: float
    :param vdims: value dimension for vector data
    :type vdims: str
    :param timestamp_column: name of the timestamp column, valid for vector data
    :type timestamp_column: str
    :param geometry_column: name of the geometry column, valid for vector data
    :type geometry_column: str
    :return: plot
    :rtype: holovies/bokeh
    """

    features = list(FeatureParser(feature))
    feature_type = features[0][0]
    feature_name = features[0][1]
    if feature_type in (FeatureType.MASK, FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS):
        vis = plot_raster(eopatch, feature_type, feature_name, alpha=alpha)
    elif feature_type == FeatureType.DATA:
        vis = plot_data(eopatch, feature_name, rgb=rgb, rgb_factor=rgb_factor)
    elif feature_type == FeatureType.VECTOR:
        vis = plot_vector(eopatch, feature_name, alpha=alpha,
                          timestamp_column=timestamp_column,
                          geometry_column=geometry_column)
    elif feature_type == FeatureType.VECTOR_TIMELESS:
        vis = plot_vector_timeless(eopatch, feature_name, alpha=alpha, vdims=vdims)
    else:      # elif feature_type in (FeatureType.SCALAR, FeatureType.LABEL):
        vis = plot_scalar_label(eopatch, feature_type, feature_name)

    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))


def plot_data(eopatch, feature_name, rgb, rgb_factor):
    """ Plots the FeatureType.DATA of eopatch.

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature_name: name of the feature
    :param feature_name: str
    :param rgb: wheather to output rgb image, list of indices to create rgb image from
    :type rgb: [int, int, int]
    :param rgb_factor: factor for rgb bands multiplication
    :type rgb_factor: float
    :return: visualization
    :rtype: holoview/geoviews/bokeh
    """
    epsg_number = eopatch.bbox.crs.ogc_string().split(':')[1]
    epsg_number = 3857 if epsg_number == 4326 else epsg_number
    data_da = array_to_dataframe(eopatch, (FeatureType.DATA, feature_name), epsg_number=epsg_number)
    timestamps = eopatch.timestamp
    crs = eopatch.bbox.crs
    if not rgb:
        return data_da.hvplot(x='x', y='y', crs=ccrs.epsg(epsg_number))
    data_rgb = eopatch_da_to_rgb(data_da, feature_name, crs, rgb, rgb_factor)
    rgb_dict = {timestamp_: plot_rgb_one(data_rgb, timestamp_) for timestamp_ in timestamps}

    return hv.HoloMap(rgb_dict, kdims=['time'])


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


def plot_raster(eopatch, feature_type, feature_name, alpha):
    """ Makes visualization for raster data (except for FeatureType.DATA)

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature_type: type of eopatch feature
    :type feature_type: FeatureType
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :param alpha: transparency of the visualization
    :type alpha: float
    :return: visualization
    :rtype: holoviews/geoviews/bokeh
    """
    epsg_number = eopatch.bbox.crs.ogc_string().split(':')[1]
    epsg_number = 3857 if epsg_number == 4326 else epsg_number
    data_da = array_to_dataframe(eopatch, (feature_type, feature_name), epsg_number=epsg_number)
    data_min = data_da.values.min()
    data_max = data_da.values.max()
    data_levels = len(np.unique(data_da))
    data_levels = 11 if data_levels > 11 else data_levels
    data_da = data_da.where(data_da > 0).fillna(-1)
    vis = data_da.hvplot(x='x', y='y',
                         crs=ccrs.epsg(epsg_number)).opts(clim=(data_min, data_max),
                                                          clipping_colors={'min': 'transparent'},
                                                          color_levels=data_levels,
                                                          alpha=alpha)
    return vis


def plot_vector(eopatch, feature_name, alpha, timestamp_column, geometry_column):
    """ Visualizaton for vector (FeatureType.VECTOR) data

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :param alpha: transparency of visualization
    :type alpha: float
    :param timestamp_column: name of the timestamp column, valid for vector data
    :type timestamp_column: str
    :param geometry_column: name of the geometry column, valid for vector data
    :type geometry_column: str
    :return: visualization
    :rtype: holoviews/geoviews/bokeh

    """
    epsg_number = eopatch.bbox.crs.ogc_string().split(':')[1]
    timestamps = eopatch.timestamp
    data_gpd = fill_vector(eopatch, FeatureType.VECTOR, feature_name,
                           timestamp_column=timestamp_column,
                           geometry_column=geometry_column)
    if epsg_number == 4326:
        epsg_number = 3857
        data_gpd = data_gpd.to_crs({'init': 'epsg:3857'})
    shapes_dict = {timestamp_: plot_shapes_one(data_gpd, timestamp_,
                                               epsg_number, alpha, timestamp_column)
                   for timestamp_ in timestamps}
    return hv.HoloMap(shapes_dict, kdims=['time'])


def fill_vector(eopatch, feature_type, feature_name, timestamp_column, geometry_column):
    """ Adds timestamps from eopatch to GeoDataFrame.

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature_type: type of eopatch feature
    :type feature_type: FeatureType
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :param timestamp_column: name of the timestamp column, valid for vector data
    :type timestamp_column: str
    :param geometry_column: name of the geometry column, valid for vector data
    :type geometry_column: str
    :return: GeoDataFrame with added data
    :rtype: geopandas.GeoDataFrame
    """
    vector = eopatch[feature_type][feature_name].copy()
    vector['valid'] = True
    eopatch_timestamps = eopatch.timestamp
    vector_timestamps = list(vector[timestamp_column])
    blank_timestamps = [timestamp for timestamp in eopatch_timestamps if timestamp not in vector_timestamps]
    dummy_geometry = create_dummy_polygon(eopatch.bbox, 0.0000001)

    temp_df = create_dummy_dataframe(vector,
                                     blank_timestamps=blank_timestamps,
                                     dummy_geometry=dummy_geometry,
                                     timestamp_column=timestamp_column,
                                     geometry_column=geometry_column)

    final_vector = gpd.GeoDataFrame(pd.concat((vector, temp_df), ignore_index=True))
    return final_vector


def create_dummy_dataframe(geodataframe, blank_timestamps, dummy_geometry,
                           timestamp_column, geometry_column,
                           fill_str='', fill_numeric=1):
    """ Creates geopadnas GeoDataFrame to fill with dummy data (for visualization)

    :param geodataframe: dataframe to append rows to
    :type geodataframe: geopandas.GeoDataFrame
    :param blank_timestamps: timestamps for constructing dataframe
    :type blank_timestamps: list of timestamps
    :param dummy_geometry: geometry to plot when there is no data
    :type dummy_geometry: shapely.geometry.Polygon
    :param timestamp_column: name of the timestamp column, valid for vector data
    :type timestamp_column: str
    :param geometry_column: name of the geometry column, valid for vector data
    :type geometry_column: str
    :param fill_str: insert when there is no value in str column
    :type fill_str: str
    :param fill_numeric: insert when
    :type fill_numeric: float
    :return: dataframe with dummy data
    :rtype: geopandas.GeoDataFrame
    """
    dataframe = pd.DataFrame(data=blank_timestamps, columns=[timestamp_column])

    for column in geodataframe.columns:
        if column == timestamp_column:
            continue
        elif column == geometry_column:
            dataframe[column] = dummy_geometry
        elif column == 'valid':
            dataframe[column] = False
        elif geodataframe[column].dtype in (int, float):
            dataframe[column] = fill_numeric
        else:
            dataframe[column] = fill_str

    return dataframe


def create_dummy_polygon(bbox, addition_factor):
    """ Creates geometry/polygon to plot if there is no data (at timestamp)

    :param bbox: eopatch bbox
    :type bbox: EOPatch.bbox
    :param addition_factor: size of the 'blank polygon'
    :type addition_factor: float
    :return: polygon
    :rtype: shapely.geometry.Polygon
    """
    x_blank, y_blank = bbox.lower_left
    dummy_geometry = geometry.Polygon([[x_blank, y_blank],
                                       [x_blank + addition_factor, y_blank],
                                       [x_blank + addition_factor, y_blank + addition_factor],
                                       [x_blank, y_blank + addition_factor]])

    return dummy_geometry


def plot_scalar_label(eopatch, feature_type, feature_name):
    """ Line plot for FeatureType.SCALAR, FeatureType.LABEL

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature_type: type of eopatch feature
    :type feature_type: FeatureType
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :return: visualization
    :rtype: holoviews/geoviews/bokeh
    """
    data_da = array_to_dataframe(eopatch, (feature_type, feature_name))
    if data_da.dtype == np.bool:
        data_da = data_da.astype(np.int8)
    return data_da.hvplot()


def plot_shapes_one(data_gpd, timestamp, epsg_number, alpha, timestamp_column):  # OK
    """ Plots shapes for one timestamp from geopandas GeoDataFRame

    :param data_gpd: data to plot
    :type data_gpd: geopandas.GeoDataFrame
    :param timestamp: timestamp to plot data for
    :type timestamp: datetime
    :param epsg_number: in which crs is the data to plot
    :type epsg_number: int
    :param alpha: transpareny
    :type alpha: float
    :param timestamp_column: name of the timestamp column
    :type timestamp_column: str
    :return: visualization
    :rtype: geoviews
    """
    out = data_gpd.loc[data_gpd[timestamp_column] == timestamp]
    return gv.Polygons(out, crs=ccrs.epsg(epsg_number)).opts(alpha=alpha)


def plot_vector_timeless(eopatch, feature_name, alpha, vdims):
    """ Plot FeatureType.VECTOR_TIMELESS data

    :param eopatch:
    :type eopatch: EOPatch
    :param feature_name: name of the eopatch featrue
    :type feature_name: str
    :param alpha: transparency
    :type alpha: float
    :param vdims: value dimension for plot (for coloring the shapes)
    :type vdims: str
    :return: visalization
    :rtype: geoviews
    """
    epsg_number = eopatch.bbox.crs.ogc_string().split(':')[1]
    if epsg_number == 4326:
        epsg_number = 3857
        data_gpd = eopatch[FeatureType.VECTOR_TIMELESS][feature_name].to_crs({'init': 'epsg:3857'})
    else:
        data_gpd = eopatch[FeatureType.VECTOR_TIMELESS][feature_name]
    return gv.Polygons(data_gpd, crs=ccrs.epsg(epsg_number), vdims=vdims).opts(alpha=alpha)


def eopatch_da_to_rgb(eopatch_da, feature_name, crs, rgb, rgb_factor):
    """ Creates new xarray DataArray (from old one) to plot rgb image with hv.Holomap

    :param eopatch_da: eopatch DataArray
    :type eopatch_da: DataArray
    :param feature_name: name of the feature to plot
    :type feature_name:  str
    :param crs: in which crs are the data
    :type crs: sentinelhub.constants.crs
    :param rgb: list of bands to use as rgb channels
    :type rgb: [int, int, int]
    :param rgb_factor: factor for rgb bands multiplication
    :type rgb_factor: float
    :return: eopatch DataArray with proper coordinates, dimensions, crs
    :rtype: xarray.DataArray
    """
    timestamps = eopatch_da.coords['time'].values
    bands = eopatch_da[..., rgb] * rgb_factor
    bands = bands.rename({feature_name.replace('-', '_') + '_dim': 'band'}).transpose('time', 'band', 'y', 'x')
    x_values, y_values = new_coordinates(eopatch_da, crs, CRS(3857))
    eopatch_rgb = xr.DataArray(data=np.clip(bands.data, 0, 1),
                               coords={'time': timestamps,
                                       'band': rgb,
                                       'y': np.flip(y_values),
                                       'x': x_values},
                               dims=('time', 'band', 'y', 'x'))
    return eopatch_rgb
