"""
A module implementing EOPatch merging utility

Credits:
Copyright (c) 2018-2020 William Ouellette
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import functools
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from .constants import FeatureType
from .utilities import FeatureParser


def merge_eopatches(*eopatches, features=..., time_dependent_op=None, timeless_op=None):
    """ Merge features of given EOPatches into a new EOPatch

    :param eopatches: Any number of EOPatches to be merged together
    :type eopatches: EOPatch
    :param features: A collection of features to be merged together. By default all features will be merged.
    :type features: object
    :param time_dependent_op: An operation to be used to join data for any time-dependent raster feature. Before
        joining time slices of all arrays will be sorted. Supported options are:
        - None (default): If time slices with matching timestamps have the same values, take one. Raise an error
          otherwise.
        - 'concatenate': Keep all time slices, even the ones with matching timestamps
        - 'min': Join time slices with matching timestamps by taking minimum values. Ignore NaN values.
        - 'max': Join time slices with matching timestamps by taking maximum values. Ignore NaN values.
        - 'mean': Join time slices with matching timestamps by taking mean values. Ignore NaN values.
        - 'median': Join time slices with matching timestamps by taking median values. Ignore NaN values.
    :type time_dependent_op: str or Callable or None
    :param timeless_op: An operation to be used to join data for any timeless raster feature. Supported options
        are:
        - None (default): If arrays are the same, take one. Raise an error otherwise.
        - 'concatenate': Join arrays over the last (i.e. bands) dimension
        - 'min': Join arrays by taking minimum values. Ignore NaN values.
        - 'max': Join arrays by taking maximum values. Ignore NaN values.
        - 'mean': Join arrays by taking mean values. Ignore NaN values.
        - 'median': Join arrays by taking median values. Ignore NaN values.
    :type timeless_op: str or Callable or None
    :return: A dictionary with EOPatch features and values
    :rtype: Dict[(FeatureType, str), object]
    """
    reduce_timestamps = time_dependent_op != 'concatenate'
    time_dependent_op = _parse_operation(time_dependent_op, is_timeless=False)
    timeless_op = _parse_operation(timeless_op, is_timeless=True)

    all_features = {feature for eopatch in eopatches for feature in FeatureParser(features)(eopatch)}
    eopatch_content = {}

    timestamps, sort_mask, split_mask = _merge_timestamps(eopatches, reduce_timestamps)
    eopatch_content[FeatureType.TIMESTAMP] = timestamps

    for feature in all_features:
        feature_type, feature_name = feature

        if feature_type.is_raster():
            if feature_type.is_time_dependent():
                eopatch_content[feature] = _merge_time_dependent_raster_feature(
                    eopatches, feature, time_dependent_op, sort_mask, split_mask
                )
            else:
                eopatch_content[feature] = _merge_timeless_raster_feature(eopatches, feature,
                                                                          timeless_op)

        if feature_type.is_vector():
            eopatch_content[feature] = _merge_vector_feature(eopatches, feature)

        if feature_type is FeatureType.META_INFO:
            eopatch_content[feature] = _select_meta_info_feature(eopatches, feature_name)

        if feature_type is FeatureType.BBOX:
            eopatch_content[feature] = _get_common_bbox(eopatches)

    return eopatch_content


def _parse_operation(operation_input, is_timeless):
    """ Transforms operation's instruction (i.e. an input string) into a function that can be applied to a list of
    arrays. If the input already is a function it returns it.
    """
    if isinstance(operation_input, Callable):
        return operation_input

    try:
        return {
            None: _return_if_equal_operation,
            'concatenate': functools.partial(np.concatenate, axis=-1 if is_timeless else 0),
            'mean': functools.partial(np.nanmean, axis=0),
            'median': functools.partial(np.nanmedian, axis=0),
            'min': functools.partial(np.nanmin, axis=0),
            'max': functools.partial(np.nanmax, axis=0)
        }[operation_input]
    except KeyError as exception:
        raise ValueError(f'Merge operation {operation_input} is not supported') from exception


def _return_if_equal_operation(arrays):
    """ Checks if arrays are all equal and returns first one of them. If they are not equal it raises an error.
    """
    if _all_equal(arrays):
        return arrays[0]
    raise ValueError('Cannot merge given arrays because their values are not the same, please define a different '
                     'merge operation')


def _merge_timestamps(eopatches, reduce_timestamps):
    """ Merges together timestamps from EOPatches. It also prepares masks on how to sort and join data in any
    time-dependent raster feature.
    """
    all_timestamps = [timestamp for eopatch in eopatches for timestamp in eopatch.timestamp
                      if eopatch.timestamp is not None]

    if not all_timestamps:
        return [], None, None

    sort_mask = np.argsort(all_timestamps)
    all_timestamps = sorted(all_timestamps)

    if not reduce_timestamps:
        return all_timestamps, sort_mask, None

    split_mask = [
        index + 1 for index, (timestamp, next_timestamp) in enumerate(zip(all_timestamps[:-1], all_timestamps[1:]))
        if timestamp != next_timestamp
    ]
    reduced_timestamps = [timestamp for index, timestamp in enumerate(all_timestamps)
                          if index == 0 or timestamp != all_timestamps[index - 1]]

    return reduced_timestamps, sort_mask, split_mask


def _merge_time_dependent_raster_feature(eopatches, feature, operation, sort_mask, split_mask):
    """ Merges numpy arrays of a time-dependent raster feature with a given operation and masks on how to sort and join
    time raster's time slices.
    """
    arrays = _extract_feature_values(eopatches, feature)

    merged_array = np.concatenate(arrays, axis=0)
    del arrays

    if sort_mask is None:
        return merged_array
    merged_array = merged_array[sort_mask]

    if split_mask is None or len(split_mask) == merged_array.shape[0] - 1:
        return merged_array

    split_arrays = np.split(merged_array, split_mask)
    del merged_array

    split_arrays = [operation(array_chunk) for array_chunk in split_arrays]
    return np.array(split_arrays)


def _merge_timeless_raster_feature(eopatches, feature, operation):
    """ Merges numpy arrays of a timeless raster feature with a given operation.
    """
    arrays = _extract_feature_values(eopatches, feature)

    if len(arrays) == 1:
        return arrays[0]
    return operation(arrays)


def _merge_vector_feature(eopatches, feature):
    """ Merges GeoDataFrames of a vector feature.
    """
    dataframes = _extract_feature_values(eopatches, feature)

    if len(dataframes) == 1:
        return dataframes[0]

    crs_list = [dataframe.crs for dataframe in dataframes if dataframe.crs is not None]
    if not crs_list:
        crs_list = [None]
    if not _all_equal(crs_list):
        raise ValueError(f'Cannot merge feature {feature} because dataframes are defined for '
                         f'different CRS')

    merged_dataframe = GeoDataFrame(pd.concat(dataframes, ignore_index=True), crs=crs_list[0])
    merged_dataframe = merged_dataframe.drop_duplicates(ignore_index=True)
    # In future a support for vector operations could be added here

    return merged_dataframe


def _select_meta_info_feature(eopatches, feature_name):
    """ Selects a value for a meta info feature of a merged EOPatch. By default the value is the first one.
    """
    values = _extract_feature_values(eopatches, (FeatureType.META_INFO, feature_name))

    if not _all_equal(values):
        message = f'EOPatches have different values of meta info feature {feature_name}. The first value will be ' \
                  f'used in a merged EOPatch'
        warnings.warn(message, category=UserWarning)

    return values[0]


def _get_common_bbox(eopatches):
    """ Makes sure that all EOPatches, which define a bounding box and CRS, define the same ones.
    """
    bboxes = [eopatch.bbox for eopatch in eopatches if eopatch.bbox is not None]

    if not bboxes:
        return None

    if _all_equal(bboxes):
        return bboxes[0]
    raise ValueError('Cannot merge EOPatches because they are defined for different bounding boxes')


def _extract_feature_values(eopatches, feature):
    """ A helper function that extracts a feature values from those EOPatches where a feature exists.
    """
    feature_type, feature_name = feature
    return [eopatch[feature] for eopatch in eopatches if feature_name in eopatch[feature_type]]


def _all_equal(values):
    """ A helper function that checks if all values in a given list are equal to each other.
    """
    first_value = values[0]

    if isinstance(first_value, np.ndarray):
        return all(np.array_equal(first_value, array, equal_nan=True) for array in values[1:])

    return all(first_value == value for value in values[1:])
