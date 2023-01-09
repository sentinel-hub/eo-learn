"""
A module implementing EOPatch merging utility

Credits:
Copyright (c) 2018-2020 William Ouellette
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import datetime as dt
import functools
import itertools as it
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from sentinelhub import BBox

from .constants import FeatureType
from .exceptions import EORuntimeWarning
from .types import FeatureSpec, FeaturesSpecification, Literal
from .utils.parsing import FeatureParser

if TYPE_CHECKING:
    from .eodata import EOPatch

OperationInputType = Union[Literal[None, "concatenate", "min", "max", "mean", "median"], Callable]


def merge_eopatches(
    *eopatches: EOPatch,
    features: FeaturesSpecification = ...,
    time_dependent_op: OperationInputType = None,
    timeless_op: OperationInputType = None,
) -> Dict[FeatureSpec, Any]:
    """Merge features of given EOPatches into a new EOPatch.

    :param eopatches: Any number of EOPatches to be merged together
    :param features: A collection of features to be merged together. By default, all features will be merged.
    :param time_dependent_op: An operation to be used to join data for any time-dependent raster feature. Before
        joining time slices of all arrays will be sorted. Supported options are:

        - None (default): If time slices with matching timestamps have the same values, take one. Raise an error
          otherwise.
        - 'concatenate': Keep all time slices, even the ones with matching timestamps
        - 'min': Join time slices with matching timestamps by taking minimum values. Ignore NaN values.
        - 'max': Join time slices with matching timestamps by taking maximum values. Ignore NaN values.
        - 'mean': Join time slices with matching timestamps by taking mean values. Ignore NaN values.
        - 'median': Join time slices with matching timestamps by taking median values. Ignore NaN values.

    :param timeless_op: An operation to be used to join data for any timeless raster feature. Supported options
        are:

        - None (default): If arrays are the same, take one. Raise an error otherwise.
        - 'concatenate': Join arrays over the last (i.e. bands) dimension
        - 'min': Join arrays by taking minimum values. Ignore NaN values.
        - 'max': Join arrays by taking maximum values. Ignore NaN values.
        - 'mean': Join arrays by taking mean values. Ignore NaN values.
        - 'median': Join arrays by taking median values. Ignore NaN values.

    :return: Merged EOPatch
    """
    reduce_timestamps = time_dependent_op != "concatenate"
    time_dependent_operation = _parse_operation(time_dependent_op, is_timeless=False)
    timeless_operation = _parse_operation(timeless_op, is_timeless=True)

    feature_parser = FeatureParser(features)
    all_features = {feature for eopatch in eopatches for feature in feature_parser.get_features(eopatch)}
    eopatch_content: Dict[FeatureSpec, object] = {}

    timestamps, order_mask_per_eopatch = _merge_timestamps(eopatches, reduce_timestamps)
    optimize_raster_temporal = _check_if_optimize(eopatches, time_dependent_op)

    for feature in all_features:
        feature_type, feature_name = feature

        if feature_type.is_raster():
            if feature_type.is_temporal():
                eopatch_content[feature] = _merge_time_dependent_raster_feature(
                    eopatches, feature, time_dependent_operation, order_mask_per_eopatch, optimize_raster_temporal
                )
            else:
                eopatch_content[feature] = _merge_timeless_raster_feature(eopatches, feature, timeless_operation)

        if feature_type.is_vector():
            eopatch_content[feature] = _merge_vector_feature(eopatches, feature)

        if feature_type is FeatureType.TIMESTAMP:
            eopatch_content[feature] = timestamps

        if feature_type is FeatureType.META_INFO:
            feature_name = cast(str, feature_name)  # parser makes sure of it
            eopatch_content[feature] = _select_meta_info_feature(eopatches, feature_name)

        if feature_type is FeatureType.BBOX:
            eopatch_content[feature] = _get_common_bbox(eopatches)

    return eopatch_content


def _parse_operation(operation_input: OperationInputType, is_timeless: bool) -> Callable:
    """Transforms operation's instruction (i.e. an input string) into a function that can be applied to a list of
    arrays. If the input already is a function it returns it.
    """
    defaults: Dict[Optional[str], Callable] = {
        None: _return_if_equal_operation,
        "concatenate": functools.partial(np.concatenate, axis=-1 if is_timeless else 0),
        "mean": functools.partial(np.nanmean, axis=0),
        "median": functools.partial(np.nanmedian, axis=0),
        "min": functools.partial(np.nanmin, axis=0),
        "max": functools.partial(np.nanmax, axis=0),
    }
    if operation_input in defaults:
        return defaults[operation_input]  # type: ignore[index]

    if isinstance(operation_input, Callable):  # type: ignore[arg-type]  #mypy 0.981 has issues with callable
        return cast(Callable, operation_input)
    raise ValueError(f"Merge operation {operation_input} is not supported")


def _return_if_equal_operation(arrays: np.ndarray) -> bool:
    """Checks if arrays are all equal and returns first one of them. If they are not equal it raises an error."""
    if _all_equal(arrays):
        return arrays[0]
    raise ValueError("Cannot merge given arrays because their values are not the same.")


def _merge_timestamps(
    eopatches: Sequence[EOPatch], reduce_timestamps: bool
) -> Tuple[List[dt.datetime], List[np.ndarray]]:
    """Merges together timestamps from EOPatches. It also prepares a list of masks, one for each EOPatch, how
    timestamps should be ordered and joined together.
    """
    timestamps_per_eopatch = [eopatch.timestamp for eopatch in eopatches]
    all_timestamps = [timestamp for eopatch_timestamps in timestamps_per_eopatch for timestamp in eopatch_timestamps]

    if not all_timestamps:
        return [], [np.array([], dtype=np.int32) for _ in range(len(eopatches))]

    if reduce_timestamps:
        unique_timestamps, order_mask = np.unique(all_timestamps, return_inverse=True)  # type: ignore[call-overload]
        ordered_timestamps = unique_timestamps.tolist()
    else:
        order_mask = np.argsort(all_timestamps)  # type: ignore[arg-type]
        ordered_timestamps = sorted(all_timestamps)

    order_mask = order_mask.tolist()

    order_mask_iter = iter(order_mask)
    order_mask_per_eopatch = [
        np.array(list(it.islice(order_mask_iter, len(eopatch_timestamps))), dtype=np.int32)
        for eopatch_timestamps in timestamps_per_eopatch
    ]

    return ordered_timestamps, order_mask_per_eopatch


def _check_if_optimize(eopatches: Sequence[EOPatch], operation_input: OperationInputType) -> bool:
    """Checks whether optimisation of `_merge_time_dependent_raster_feature` is possible"""
    if operation_input not in [None, "mean", "median", "min", "max"]:
        return False
    timestamp_list = [eopatch.timestamp for eopatch in eopatches]
    return _all_equal(timestamp_list)


def _merge_time_dependent_raster_feature(
    eopatches: Sequence[EOPatch],
    feature: FeatureSpec,
    operation: Callable,
    order_mask_per_eopatch: Sequence[np.ndarray],
    optimize: bool,
) -> np.ndarray:
    """Merges numpy arrays of a time-dependent raster feature with a given operation and masks on how to order and join
    time raster's time slices.
    """

    merged_array, merged_order_mask = _extract_and_join_time_dependent_feature_values(
        eopatches, feature, order_mask_per_eopatch, optimize
    )

    # Case where feature array is already in the correct order and doesn't need splitting, which includes a case
    # where array has a size 0
    if _is_strictly_increasing(merged_order_mask):
        return merged_array

    sort_mask = np.argsort(merged_order_mask)
    merged_array = merged_array[sort_mask]
    merged_order_mask = merged_order_mask[sort_mask]

    # Case where feature array has been sorted but doesn't need splitting
    if _is_strictly_increasing(merged_order_mask):
        return merged_array

    split_indices = np.nonzero(np.diff(merged_order_mask))[0] + 1
    split_arrays = np.split(merged_array, split_indices)
    del merged_array

    try:
        split_arrays = [operation(array_chunk) for array_chunk in split_arrays]
    except ValueError as exception:
        raise ValueError(
            f"Failed to merge {feature} with {operation}, try setting a different value for merging "
            "parameter time_dependent_op"
        ) from exception

    return np.array(split_arrays)


def _extract_and_join_time_dependent_feature_values(
    eopatches: Sequence[EOPatch],
    feature: FeatureSpec,
    order_mask_per_eopatch: Sequence[np.ndarray],
    optimize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collects feature arrays from EOPatches that have them and joins them together. It also joins together
    corresponding order masks.
    """
    arrays = []
    order_masks = []
    feature_type, feature_name = feature

    for eopatch, order_mask in zip(eopatches, order_mask_per_eopatch):
        if feature_name in eopatch[feature_type]:
            array = eopatch[feature_type, feature_name]
            if order_mask.size != array.shape[0]:
                raise ValueError(
                    f"Cannot merge a time-dependent feature {feature} because time dimension of an array "
                    f"in one EOPatch is {array.shape[0]} but EOPatch has {order_mask.size} timestamps"
                )

            arrays.append(array)
            order_masks.append(order_mask)

    if len(arrays) == 1 or (optimize and _all_equal(arrays)):
        return arrays[0], order_masks[0]
    return np.concatenate(arrays, axis=0), np.concatenate(order_masks)


def _is_strictly_increasing(array: np.ndarray) -> bool:
    """Checks if a 1D array of values is strictly increasing."""
    return (np.diff(array) > 0).all().astype(bool)


def _merge_timeless_raster_feature(
    eopatches: Sequence[EOPatch], feature: FeatureSpec, operation: Callable
) -> np.ndarray:
    """Merges numpy arrays of a timeless raster feature with a given operation."""
    arrays = _extract_feature_values(eopatches, feature)

    if len(arrays) == 1:
        return arrays[0]

    try:
        return operation(arrays)
    except ValueError as exception:
        raise ValueError(
            f"Failed to merge {feature} with {operation}, try setting a different value for merging "
            "parameter timeless_op."
        ) from exception


def _merge_vector_feature(eopatches: Sequence[EOPatch], feature: FeatureSpec) -> GeoDataFrame:
    """Merges GeoDataFrames of a vector feature."""
    dataframes = _extract_feature_values(eopatches, feature)

    if len(dataframes) == 1:
        return dataframes[0]

    crs_list = [dataframe.crs for dataframe in dataframes if dataframe.crs is not None]
    if not crs_list:
        crs_list = [None]
    if not _all_equal(crs_list):
        raise ValueError(f"Cannot merge feature {feature} because dataframes are defined for different CRS")

    merged_dataframe = GeoDataFrame(pd.concat(dataframes, ignore_index=True), crs=crs_list[0])
    merged_dataframe = merged_dataframe.drop_duplicates(ignore_index=True)
    # In future a support for vector operations could be added here

    return merged_dataframe


def _select_meta_info_feature(eopatches: Sequence[EOPatch], feature_name: str) -> Any:
    """Selects a value for a meta info feature of a merged EOPatch. By default, the value is the first one."""
    values = _extract_feature_values(eopatches, (FeatureType.META_INFO, feature_name))

    if not _all_equal(values):
        message = (
            f"EOPatches have different values of meta info feature {feature_name}. The first value will be "
            "used in a merged EOPatch."
        )
        warnings.warn(message, category=EORuntimeWarning)

    return values[0]


def _get_common_bbox(eopatches: Sequence[EOPatch]) -> Optional[BBox]:
    """Makes sure that all EOPatches, which define a bounding box and CRS, define the same ones."""
    bboxes = [eopatch.bbox for eopatch in eopatches if eopatch.bbox is not None]

    if not bboxes:
        return None

    if _all_equal(bboxes):
        return bboxes[0]
    raise ValueError("Cannot merge EOPatches because they are defined for different bounding boxes.")


def _extract_feature_values(eopatches: Sequence[EOPatch], feature: FeatureSpec) -> List[Any]:
    """A helper function that extracts a feature values from those EOPatches where a feature exists."""
    feature_type, feature_name = feature
    return [eopatch[feature] for eopatch in eopatches if feature_name in eopatch[feature_type]]


def _all_equal(values: Union[Sequence[Any], np.ndarray]) -> bool:
    """A helper function that checks if all values in a given list are equal to each other."""
    first_value = values[0]

    if isinstance(first_value, np.ndarray):
        is_numeric_dtype = np.issubdtype(first_value.dtype, np.number)
        return all(np.array_equal(first_value, array, equal_nan=is_numeric_dtype) for array in values[1:])

    return all(first_value == value for value in values[1:])
