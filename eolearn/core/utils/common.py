"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import uuid
from typing import Callable, Mapping, Sequence, cast

import geopandas as gpd
import numpy as np
from geopandas.testing import assert_geodataframe_equal


def deep_eq(fst_obj: object, snd_obj: object) -> bool:
    """Compares whether fst_obj and snd_obj are deeply equal.

    In case when both fst_obj and snd_obj are of type np.ndarray or either np.memmap, they are compared using
    np.array_equal(fst_obj, snd_obj). Otherwise, when they are lists or tuples, they are compared for length and then
    deep_eq is applied component-wise. When they are dict, they are compared for key set equality, and then deep_eq is
    applied value-wise. For all other data types that are not list, tuple, dict, or np.ndarray, the method falls back
    to the __eq__ method.

    Because np.ndarray is not a hashable object, it is impossible to form a set of numpy arrays, hence deep_eq works
    correctly.

    :param fst_obj: First object compared
    :param snd_obj: Second object compared
    :return: `True` if objects are deeply equal, `False` otherwise
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(fst_obj, type(snd_obj)):
        return False

    if isinstance(fst_obj, np.ndarray):
        snd_obj = cast(np.ndarray, snd_obj)
        if fst_obj.dtype != snd_obj.dtype:
            return False
        fst_nan_mask = np.isnan(fst_obj)
        snd_nan_mask = np.isnan(snd_obj)
        return np.array_equal(fst_obj[~fst_nan_mask], snd_obj[~snd_nan_mask]) and np.array_equal(
            fst_nan_mask, snd_nan_mask
        )

    if isinstance(fst_obj, gpd.GeoDataFrame):
        try:
            # We allow differences in index types and in dtypes of columns
            assert_geodataframe_equal(fst_obj, snd_obj, check_index_type=False, check_dtype=False)
            return True
        except AssertionError:
            return False

    if isinstance(fst_obj, (tuple, list)):
        snd_obj = cast(Sequence, snd_obj)

        return len(fst_obj) == len(snd_obj) and all(map(deep_eq, fst_obj, snd_obj))

    if isinstance(fst_obj, (dict, Mapping)):
        snd_obj = cast(dict, snd_obj)

        if fst_obj.keys() != snd_obj.keys():
            return False

        return all(deep_eq(fst_obj[key], snd_obj[key]) for key in fst_obj)

    return fst_obj == snd_obj


def generate_uid(prefix: str) -> str:
    """Generates a (sufficiently) unique ID starting with the `prefix`.

    The ID is composed of the prefix, a hexadecimal string obtained from the current time and a random hexadecimal
    string. This makes the uid sufficiently unique.
    """
    time_uid = uuid.uuid1(node=0).hex[:-12]
    random_uid = uuid.uuid4().hex[:12]
    return f"{prefix}-{time_uid}-{random_uid}"


def is_discrete_type(number_type: np.dtype | type) -> bool:
    """Checks if a given `numpy` type is a discrete numerical type."""
    return np.issubdtype(number_type, np.integer) or np.issubdtype(number_type, bool)


def _apply_to_spatial_axes(
    function: Callable[[np.ndarray], np.ndarray], data: np.ndarray, spatial_axes: tuple[int, int]
) -> np.ndarray:
    """Helper function for applying a 2D -> 2D function to a higher dimension array

    Recursively slices data into smaller-dimensional ones, until only the spatial axes remain. The indices of spatial
    axes have to be adjusted if the recursion-axis is smaller than either one, e.g. spatial axes (1, 2) become (0, 1)
    after splitting the 3D data along axis 0 into 2D arrays.

    After achieving 2D data slices the mapping function is applied. The data is then reconstructed into original form.
    """

    ax1, ax2 = spatial_axes
    if ax1 >= ax2:
        raise ValueError(
            f"For parameter `spatial_axes` the second axis must be greater than first, got {spatial_axes}."
        )

    if ax2 >= data.ndim:
        raise ValueError(
            f"Values in `spatial_axes` must be smaller than `data.ndim`, got {spatial_axes} for data of dimension"
            f" {data.ndim}."
        )

    if data.ndim <= 2:
        return function(data)

    axis = next(i for i in range(data.ndim) if i not in spatial_axes)
    data = np.moveaxis(data, axis, 0)

    ax1, ax2 = (ax if axis > ax else ax - 1 for ax in spatial_axes)

    mapped_slices = [_apply_to_spatial_axes(function, data_slice, (ax1, ax2)) for data_slice in data]
    return np.moveaxis(np.stack(mapped_slices), 0, axis)
