import numpy as np
import geoviews as gv
import xarray as xr
import holoviews as hv
import pandas as pd
import geopandas as gpd
import hvplot
import hvplot.xarray
import hvplot.pandas

from sentinelhub import BBox, CRS
from cartopy import crs as ccrs
from eolearn.core.eodata import FeatureType


def get_spatial_coordinates(bbox, data, feature_type):
    """ Returns spatial coordinates dictionary for creating xarray DataArray/Dataset

    :param bbox: EOpatch BBox
    :param data: numpy array
    :param feature_type: type of the feature
    :return: spatial coordinates
    """
    index_x = 2
    index_y = 1
    if feature_type == FeatureType.DATA_TIMELESS or feature_type == FeatureType.MASK_TIMELESS:
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

    :param: timestamp: EOpatch.timestap
    :return: temporal coordinates
    """
    coordinates = {
        'time': timestamps
    }

    return coordinates


def get_depth_coordinates(feature_name, data, names_of_channels=None):
    """ Returns band/channel/dept coordinates for

    :param: feature_name: name of feature of EOPatch
    :param: data: data of EOPatch
    :return: depth/band coordinates
    """
    coordinates = {}
    depth = feature_name.replace('-', '_')+'_dim'
    if names_of_channels:
        coordinates[depth] = names_of_channels
    elif isinstance(data, np.ndarray):
        coordinates[depth] = np.arange(data.shape[-1])

    return coordinates


def get_coordinates(feature_type, feature_name, bbox, data, timestamps, names_of_channels=None):
    """ Creates coordinates for xarray DataArray

    :param feature_type: FeatureType of EOPatch
    :param bbox: BBox of EOPatch
    :param data: data of EOPatch
    :param timestamps: timestamps of EOPatch
    :return:
    """

    if feature_type == FeatureType.DATA or feature_type == FeatureType.MASK:
        return {**get_temporal_coordinates(timestamps),
                **get_spatial_coordinates(bbox, data, feature_type),
                **get_depth_coordinates(data=data, feature_name=feature_name)
                }
    elif feature_type == FeatureType.SCALAR or feature_type == FeatureType.LABEL:
        return {**get_temporal_coordinates(timestamps),
                **get_depth_coordinates(data=data, feature_name=feature_name)
                }
    elif feature_type == FeatureType.DATA_TIMELESS or feature_type == FeatureType.MASK_TIMELESS:
        return {**get_spatial_coordinates(bbox, data, feature_type),
                **get_depth_coordinates(data=data, feature_name=feature_name)
                }
    elif feature_type == FeatureType.SCALAR_TIMELESS or feature_type == FeatureType.LABEL_TIMELESS:
        return get_depth_coordinates(data=data, feature_name=feature_name)


def get_dimensions(feature_type, feature_name):
    """ Returns list of dimensions for xarray DataArray

    :return:
    """
    depth = feature_name.replace('-', '_') + "_dim"
    if feature_type == FeatureType.DATA or feature_type == FeatureType.MASK:
        return ['time', 'y', 'x', depth]
    elif feature_type == FeatureType.SCALAR or feature_type == FeatureType.LABEL:
        return['time', depth]
    elif feature_type == FeatureType.DATA_TIMELESS or feature_type == FeatureType.MASK_TIMELESS:
        return['y', 'x', depth]
    elif feature_type == FeatureType.SCALAR_TIMELESS or feature_type == FeatureType.LABEL_TIMELESS:
        return[depth]


def array_to_dataframe(eopatch, feature_type, feature_name, remove_depth=True):
    """
        Convert one numpy ndarray to xarray dataframe
    """

    bbox = eopatch.bbox
    timestamps = eopatch.timestamp
    data = eopatch[feature_type][feature_name]
    if isinstance(data, xr.DataArray):
        data = data.values
    dimensions = get_dimensions(feature_type, feature_name)
    coordinates = get_coordinates(feature_type, feature_name, bbox, data, timestamps)
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
    :param eopatch:
    :return:
    """
    dataset = xr.Dataset()
    for feature in eopatch.get_feature_list():
        if not isinstance(feature, tuple):
            continue
        feature_type = feature[0]
        feature_name = feature[1]
        if feature_type not in (FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS, FeatureType.META_INFO):
            dataframe = array_to_dataframe(eopatch, feature_type, feature_name, remove_depth)
            dataset[feature_name] = dataframe

    return dataset


def new_coordinates(data, crs, new_crs):
    xs = data.coords['x'].values
    ys = data.coords['y'].values
    bbox = (xs[0], ys[0], xs[-1], ys[-1])
    bbox = BBox(bbox, crs=crs)
    bbox = bbox.transform(new_crs)
    xmin, ymin = bbox.get_lower_left()
    xmax, ymax = bbox.get_upper_right()
    new_xs = np.linspace(xmin, xmax, len(xs))
    new_ys = np.linspace(ymin, ymax, len(ys))

    return new_xs, new_ys


def plot_bands(eopatch_xr, timestamp, band):
    return (eopatch_xr.sel(time=timestamp)
            .isel(BANDS_S2_L1C_dim=band)
            .hvplot(x='x', y='y', crs=ccrs.UTM(33),
                    width=600, height=600))


TYPE_NO_TIME = (FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS, FeatureType.VECTOR_TIMELESS)
TYPE_TIME = (FeatureType.DATA, FeatureType.MASK, FeatureType.VECTOR)
TYPE_VECTOR = (FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS)
TYPE_TRANSPARENT = (FeatureType.DATA, FeatureType.MASK, FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS)


def plot(eopatch, feature_type, feature_name, time=None, alpha=1, rgb=None):
    vis = None
    if feature_type in (FeatureType.MASK, FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS):
        vis = plot_raster(eopatch, feature_type, feature_name, alpha=alpha)
    elif feature_type == FeatureType.DATA:
        vis = plot_data(eopatch, feature_name, rgb)
    elif feature_type == FeatureType.VECTOR:
        vis = plot_vector(eopatch, feature_name, alpha)
    elif feature_type == FeatureType.VECTOR_TIMELESS:
        vis = plot_vector_timeless(eopatch, feature_name, alpha)
    elif feature_type in (FeatureType.SCALAR, FeatureType.LABEL):
        vis = plot_scalar_label(eopatch, feature_type, feature_name)

    return vis


def plot_one():
    pass


def plot_data(eopatch, feature_name, rgb):
    data_da = array_to_dataframe(eopatch, FeatureType.DATA, feature_name)
    timestamps = eopatch.timestamp
    if not rgb:
        return data_da.hvplot(x='x', y='y', crs=ccrs.UTM(33))
    else:
        data_rgb = eopatch_da_to_rgb(data_da, feature_name, rgb)
        rgb_dict = {timestamp_: plot_rgb(data_rgb, timestamp_) for timestamp_ in timestamps}

        return hv.HoloMap(rgb_dict, kdims=['time'])


def plot_rgb(eopatch_da, timestamp):  # OK
    return eopatch_da.sel(time=timestamp).drop('time').hvplot(x='x', y='y')


def plot_raster(eopatch, feature_type, feature_name, alpha):
    data_da = array_to_dataframe(eopatch, feature_type, feature_name)
    data_min = data_da.values.min()
    data_max = data_da.values.max()
    data_levels = len(np.unique(data_da))
    data_levels = 11 if data_levels > 11 else data_levels
    data_da = data_da.where(data_da > 0).fillna(-1)
    vis = data_da.hvplot(x='x', y='y', crs=ccrs.UTM(33)).opts(clim=(data_min, data_max),
                                                              clipping_colors={'min': 'transparent'},
                                                              color_levels=data_levels,
                                                              alpha=alpha)
    return vis


def plot_vector(eopatch, feature_name, alpha):
    timestamps = eopatch.timestamp
    data_gpd = fill_vector(eopatch, FeatureType.VECTOR, feature_name)
    shapes_dict = {timestamp_: plot_shapes(data_gpd, timestamp_, alpha) for timestamp_ in timestamps}
    vis = hv.HoloMap(shapes_dict, kdims=['time'])
    return vis


def fill_vector(eopatch, feature_type, feature_name):
    vector = eopatch[feature_type][feature_name]
    eopatch_timestamps = eopatch.timestamp
    vector_timestamps = list(vector['TIMESTAMP'])
    blank_timestamps = [timestamp for timestamp in eopatch_timestamps if timestamp not in vector_timestamps]

    temp_df = pd.DataFrame(list(zip(blank_timestamps,
                                    len(blank_timestamps) * [1],
                                    len(blank_timestamps) * [eopatch.bbox.geometry])),
                           columns=vector.columns)

    final_vector = gpd.GeoDataFrame(pd.concat((vector, temp_df), ignore_index=True))
    return final_vector


def plot_scalar_label(eopatch, feature_type, feature_name):
    data_da = array_to_dataframe(eopatch, feature_type, feature_name)
    if data_da.dtype == np.bool:
        data_da = data_da.astype(np.int8)
    vis = data_da.hvplot()
    return vis


def plot_shapes(data_gpd, timestamp, alpha):  # OK
    out = data_gpd.loc[data_gpd['TIMESTAMP'] == timestamp]
    return gv.Polygons(out, crs=ccrs.UTM(33)).opts(alpha=alpha)


def plot_vector_timeless(eopatch, feature_name, alpha):
    data_gpd = eopatch[FeatureType.VECTOR_TIMELESS][feature_name]
    vis = gv.Polygons(data_gpd, crs=ccrs.UTM(33), vdims=['LULC_ID']).opts(alpha=alpha)
    return vis


def get_data(layer, eopatch, eopatch_ds, alpha, rgb):
    feature_type = layer[0]
    feature_name = layer[1]
    if feature_type in [FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]:
        return eopatch[feature_type][feature_name]
    elif rgb:
        return eopatch_ds_to_rgb(eopatch_ds=eopatch_ds, feature_name=feature_name, rgb=rgb)
    else:
        return eopatch_ds[feature_name]


def eopatch_ds_to_rgb(eopatch_ds, feature_name, rgb):
    timestamps = eopatch_ds.coords['time'].values
    bands = eopatch_ds[feature_name][..., rgb] * 3.5
    bands = bands.rename({feature_name.replace('-', '_') + '_dim': 'band'}).transpose('time', 'band', 'y', 'x')
    xs, ys = new_coordinates(eopatch_ds, CRS(32633), CRS(3857))
    eopatch_rgb = xr.DataArray(data=np.clip(bands.data, 0, 1),
                               coords={'time': timestamps,
                                       'band': rgb,
                                       'y': np.flip(ys),
                                       'x': xs},
                               dims=('time', 'band', 'y', 'x'))
    return eopatch_rgb


def eopatch_da_to_rgb(eopatch_da, feature_name, rgb):
    timestamps = eopatch_da.coords['time'].values
    bands = eopatch_da[..., rgb] * 3.5
    bands = bands.rename({feature_name.replace('-', '_') + '_dim': 'band'}).transpose('time', 'band', 'y', 'x')
    xs, ys = new_coordinates(eopatch_da, CRS(32633), CRS(3857))
    eopatch_rgb = xr.DataArray(data=np.clip(bands.data, 0, 1),
                               coords={'time': timestamps,
                                       'band': rgb,
                                       'y': np.flip(ys),
                                       'x': xs},
                               dims=('time', 'band', 'y', 'x'))
    return eopatch_rgb

