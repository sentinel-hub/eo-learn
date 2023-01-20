"""
Useful utilities for working with raster data, typically `numpy` arrays.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Domagoj Korais, Matic Lubej, Žiga Lukšič (Sinergise)
Copyright (c) 2017-2022 Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from typing import Tuple

import numpy as np

from ..types import Literal


def fast_nanpercentile(data: np.ndarray, percentile: float, *, method: str = "linear") -> np.ndarray:
    """This is an alternative implementation of `numpy.nanpercentile`. For cases where the size of the first dimension
    is relatively small compared to the size of the entire array it works much faster than the original.

    The algorithm divides pixel data over the first axis into groups by how many NaN values they have. In each group
    NaN values are removed and `numpy.percentile` function is applied. If any series contains only NaN values also any
    percentile of that series will be NaN.

    This function differs from `numpy` implementations only in the following:

    - In case the size of the first dimension of `data` is `0` this method will still return an output array with
      all NaN values. This matches with `numpy.nanpercentile` while `numpy.percentile` raises an error.
    - The output dtype of this method will be always the same as the input dtype while `numpy` implementations
      in many cases use `float64` as the output dtype.

    :param data: An array for which percentiles will be calculated along the first axis.
    :param percentile: A percentile to compute, which must be between `0` and `100` inclusive.
    :param method: A method for estimating the percentile. This parameter is propagated to `numpy.percentile`.
    :return: An array of percentiles and a shape equal to the shape of `data` array without the first dimension.
    """
    method_kwargs = {"method" if np.__version__ >= "1.22.0" else "interpolation": method}

    combined_data = np.zeros(data.shape[1:], dtype=data.dtype)

    no_data_counts = np.count_nonzero(np.isnan(data), axis=0)
    for no_data_num in np.unique(no_data_counts):
        mask = no_data_counts == no_data_num

        chunk = data[..., mask]
        time_size, sample_size = chunk.shape

        if time_size == no_data_num:
            result = np.full(sample_size, np.nan, dtype=chunk.dtype)
        else:
            chunk = chunk.flatten(order="F")
            chunk = chunk[~np.isnan(chunk)]
            chunk = chunk.reshape((time_size - no_data_num, sample_size), order="F")

            result = np.percentile(chunk, q=percentile, axis=0, **method_kwargs)  # type: ignore[call-overload]

        combined_data[mask] = result

    return combined_data


def constant_pad(
    array: np.ndarray,
    multiple_of: Tuple[int, int],
    up_down_rule: Literal["even", "up", "down"] = "even",
    left_right_rule: Literal["even", "left", "right"] = "even",
    pad_value: float = 0,
) -> np.ndarray:
    """Function pads an image of shape (rows, columns, channels) with zeros.

    It pads an image so that the shape becomes (rows + padded_rows, columns + padded_columns, channels), where
    padded_rows = (int(rows/multiple_of[0]) + 1) * multiple_of[0] - rows

    Same rule is applied to columns.

    :param array: Array with shape `(rows, columns, ...)` to be padded.
    :param multiple_of: make array' rows and columns multiple of this tuple
    :param up_down_rule: Add padded rows evenly to the top/bottom of the image, or up (top) / down (bottom) only
    :param left_right_rule: Add padded columns evenly to the left/right of the image, or left / right only
    :param pad_value: Value to be assigned to padded rows and columns
    """
    rows, columns = array.shape[:2]
    row_padding, col_padding = 0, 0

    if rows % multiple_of[0]:
        row_padding = (int(rows / multiple_of[0]) + 1) * multiple_of[0] - rows

    if columns % multiple_of[1]:
        col_padding = (int(columns / multiple_of[1]) + 1) * multiple_of[1] - columns

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
