"""
This module implements conversion from/to xarray DataArray/Dataset

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import re
import numpy as np
import xarray as xr

from sentinelhub import BBox

from eolearn.core import FeatureTypeSet, FeatureParser


def string_to_variable(string, extension=None):
    """

    :param string: string to be used as python variable name
    :type string: str
    :param extension: string to be appended to string
    :type extension: str
    :return: valid python variable name
    :rtype: str
    """

    string = re.sub('[^0-9a-zA-Z_]', '', string)
    string = re.sub('^[^a-zA-Z_]+', '', string)
    if extension:
        string += extension

    return string


def _get_spatial_coordinates(bbox, data, feature_type):
    """ Returns spatial coordinates (dictionary) for creating xarray DataArray/Dataset
        Makes sense for data

    :param bbox: eopatch bbox
    :type bbox: EOPatch BBox
    :param data: values for calculating number of coordinates
    :type data: numpy array
    :param feature_type: type of the feature
    :type feature_type: FeatureType
    :return: spatial coordinates
    :rtype: dict {'x':, 'y':}
    """
    if not (feature_type.is_spatial() and feature_type.is_raster()):
        raise ValueError('Data should be raster and have spatial dimension')
    index_x, index_y = 2, 1
    if feature_type.is_timeless():
        index_x, index_y = 1, 0
    pixel_width = (bbox.max_x - bbox.min_x)/data.shape[index_x]
    pixel_height = (bbox.max_y - bbox.min_y)/data.shape[index_y]

    return {'x': np.linspace(bbox.min_x+pixel_width/2, bbox.max_x-pixel_width/2, data.shape[index_x]),
            'y': np.linspace(bbox.max_y-pixel_height/2, bbox.min_y+pixel_height/2, data.shape[index_y])}


def _get_temporal_coordinates(timestamps):
    """ Returns temporal coordinates dictionary for creating xarray DataArray/Dataset

    :param timestamps: timestamps
    :type timestamps: EOpatch.timestamp
    :return: temporal coordinates
    :rtype: dict {'time': }
    """
    return {'time': timestamps}


def _get_depth_coordinates(feature_name, data, names_of_channels=None):
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
    depth = string_to_variable(feature_name, '_dim')
    if names_of_channels:
        coordinates[depth] = names_of_channels
    elif isinstance(data, np.ndarray):
        coordinates[depth] = np.arange(data.shape[-1])

    return coordinates


def get_coordinates(eopatch, feature, crs):
    """ Creates coordinates for xarray DataArray

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param crs: convert spatial coordinates to crs
    :type crs: sentinelhub.crs
    :return: coordinates for xarry DataArray/Dataset
    :rtype: dict
    """

    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0]
    original_crs = eopatch.bbox.crs
    if crs and original_crs != crs:
        bbox = eopatch.bbox.transform(crs)
    else:
        bbox = eopatch.bbox
    data = eopatch[feature_type][feature_name]
    timestamps = eopatch.timestamp

    if feature_type in FeatureTypeSet.RASTER_TYPES_4D:
        return {**_get_temporal_coordinates(timestamps),
                **_get_spatial_coordinates(bbox, data, feature_type),
                **_get_depth_coordinates(data=data, feature_name=feature_name)}
    if feature_type in FeatureTypeSet.RASTER_TYPES_2D:
        return {**_get_temporal_coordinates(timestamps),
                **_get_depth_coordinates(data=data, feature_name=feature_name)}
    if feature_type in FeatureTypeSet.RASTER_TYPES_3D:
        return {**_get_spatial_coordinates(bbox, data, feature_type),
                **_get_depth_coordinates(data=data, feature_name=feature_name)}
    return _get_depth_coordinates(data=data, feature_name=feature_name)


def get_dimensions(feature):
    """ Returns list of dimensions for xarray DataArray/Dataset

    :param feature: eopatch feature
    :type feature: (FeatureType, str)
    :return: dimensions for xarray DataArray/Dataset
    :rtype: list(str)
    """
    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0]
    depth = string_to_variable(feature_name, '_dim')
    if feature_type in FeatureTypeSet.RASTER_TYPES_4D:
        return ['time', 'y', 'x', depth]
    if feature_type in FeatureTypeSet.RASTER_TYPES_2D:
        return ['time', depth]
    if feature_type in FeatureTypeSet.RASTER_TYPES_3D:
        return ['y', 'x', depth]
    return [depth]


def array_to_dataframe(eopatch, feature, remove_depth=True, crs=None):
    """ Converts one feature of eopatch to xarray DataArray

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param remove_depth: removes last dimension if it is one
    :type remove_depth: bool
    :param crs: converts dimensions to crs
    :type crs: sentinelhub.crs
    :return: dataarray
    :rtype: xarray DataArray
    """
    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0]
    bbox = eopatch.bbox
    data = eopatch[feature_type][feature_name]
    if isinstance(data, xr.DataArray):
        data = data.values
    dimensions = get_dimensions(feature)
    coordinates = get_coordinates(eopatch, feature, crs=crs)
    dataframe = xr.DataArray(data=data,
                             coords=coordinates,
                             dims=dimensions,
                             attrs={'crs': str(bbox.crs),
                                    'feature_type': feature_type,
                                    'feature_name': feature_name},
                             name=string_to_variable(feature_name))

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
        if feature_type.is_raster():
            dataframe = array_to_dataframe(eopatch, (feature_type, feature_name), remove_depth)
            dataset[feature_name] = dataframe

    return dataset


def new_coordinates(data, crs, new_crs):
    """ Returns coordinates for xarray DataArray/Dataset in new crs.

    :param data: data for converting coordinates for
    :type data: xarray.DataArray or xarray.Dataset
    :param crs: old crs
    :type crs: sentinelhub.CRS
    :param new_crs: new crs
    :type new_crs: sentinelhub.CRS
    :return: new x and y coordinates
    :rtype: (float, float)
    """
    x_values = data.coords['x'].values
    y_values = data.coords['y'].values
    bbox = BBox((x_values[0], y_values[0], x_values[-1], y_values[-1]), crs=crs)
    bbox = bbox.transform(new_crs)
    xmin, ymin = bbox.get_lower_left()
    xmax, ymax = bbox.get_upper_right()
    new_xs = np.linspace(xmin, xmax, len(x_values))
    new_ys = np.linspace(ymin, ymax, len(y_values))

    return new_xs, new_ys
