"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import uuid
from typing import Union

import geopandas as gpd
import numpy as np
from geopandas.testing import assert_geodataframe_equal


def deep_eq(fst_obj, snd_obj):
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

    if isinstance(fst_obj, (list, tuple)):
        if len(fst_obj) != len(snd_obj):
            return False

        for element_fst, element_snd in zip(fst_obj, snd_obj):
            if not deep_eq(element_fst, element_snd):
                return False
        return True

    if isinstance(fst_obj, dict):
        if fst_obj.keys() != snd_obj.keys():
            return False

        for key in fst_obj:
            if not deep_eq(fst_obj[key], snd_obj[key]):
                return False
        return True

    return fst_obj == snd_obj


def constant_pad(array, multiple_of, up_down_rule="even", left_right_rule="even", pad_value=0):
    """Function pads an image of shape (rows, columns, channels) with zeros.

    It pads an image so that the shape becomes (rows + padded_rows, columns + padded_columns, channels), where
    padded_rows = (int(rows/multiple_of[0]) + 1) * multiple_of[0] - rows

    Same rule is applied to columns.

    :type array: array of shape (rows, columns, channels) or (rows, columns)
    :param multiple_of: make array' rows and columns multiple of this tuple
    :type multiple_of: tuple (rows, columns)
    :param up_down_rule: Add padded rows evenly to the top/bottom of the image, or up (top) / down (bottom) only
    :type up_down_rule: up_down_rule: string, (even, up, down)
    :param up_down_rule: Add padded columns evenly to the left/right of the image, or left / right only
    :type up_down_rule: up_down_rule: string, (even, left, right)
    :param pad_value: Value to be assigned to padded rows and columns
    :type pad_value: int
    """
    shape = array.shape

    row_padding, col_padding = 0, 0

    if shape[0] % multiple_of[0]:
        row_padding = (int(shape[0] / multiple_of[0]) + 1) * multiple_of[0] - shape[0]

    if shape[1] % multiple_of[1]:
        col_padding = (int(shape[1] / multiple_of[1]) + 1) * multiple_of[1] - shape[1]

    row_padding_up, row_padding_down, col_padding_left, col_padding_right = 0, 0, 0, 0

    if row_padding > 0:
        if up_down_rule == "up":
            row_padding_up = row_padding
        elif up_down_rule == "down":
            row_padding_down = row_padding
        elif up_down_rule == "even":
            row_padding_up = int(row_padding / 2)
            row_padding_down = row_padding_up + (row_padding % 2)
        else:
            raise ValueError("Padding rule for rows not supported. Choose between even, down or up!")

    if col_padding > 0:
        if left_right_rule == "left":
            col_padding_left = col_padding
        elif left_right_rule == "right":
            col_padding_right = col_padding
        elif left_right_rule == "even":
            col_padding_left = int(col_padding / 2)
            col_padding_right = col_padding_left + (col_padding % 2)
        else:
            raise ValueError("Padding rule for columns not supported. Choose between even, left or right!")

    return np.lib.pad(
        array,
        ((row_padding_up, row_padding_down), (col_padding_left, col_padding_right)),
        "constant",
        constant_values=((pad_value, pad_value), (pad_value, pad_value)),
    )


def generate_uid(prefix: str):
    """Generates a (sufficiently) unique ID starting with the `prefix`.

    The ID is composed of the prefix, a hexadecimal string obtained from the current time and a random hexadecimal
    string. This makes the uid sufficiently unique.
    """
    time_uid = uuid.uuid1(node=0).hex[:-12]
    random_uid = uuid.uuid4().hex[:12]
    return f"{prefix}-{time_uid}-{random_uid}"


def is_discrete_type(number_type: Union[np.dtype, type]) -> bool:
    """Checks if a given `numpy` type is a discrete numerical type."""
    number_dtype = np.dtype(number_type)
    return issubclass(number_dtype.type, (np.integer, np.bool_))
