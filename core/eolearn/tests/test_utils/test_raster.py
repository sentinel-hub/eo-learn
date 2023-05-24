"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import warnings
from typing import Literal, Optional, Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core.utils.raster import constant_pad, fast_nanpercentile

# ruff: noqa: NPY002


@pytest.mark.parametrize("size", [0, 5])
@pytest.mark.parametrize("percentile", [0, 1.5, 50, 80.99, 100])
@pytest.mark.parametrize("nan_ratio", [0, 0.05, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16])
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_fast_nanpercentile(size: int, percentile: float, nan_ratio: float, dtype: type, method: str):
    data_shape = (size, 3, 2, 4)
    data = np.random.rand(*data_shape)
    data[data < nan_ratio] = np.nan

    if np.issubdtype(dtype, np.integer):
        data *= 1000
    data = data.astype(dtype)

    method_kwargs = {"method" if np.__version__ >= "1.22.0" else "interpolation": method}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        expected_result = np.nanpercentile(data, q=percentile, axis=0, **method_kwargs).astype(data.dtype)

    result = fast_nanpercentile(data, percentile, method=method)

    assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    argnames="array, multiple_of, up_down_rule, left_right_rule, pad_value, expected_result",
    argvalues=[
        (np.arange(2).reshape((1, 2)), (3, 3), "even", "right", 5, np.array([[5, 5, 5], [0, 1, 5], [5, 5, 5]])),
        (np.arange(2).reshape((1, 2)), (3, 3), "up", "even", 5, np.array([[5, 5, 5], [5, 5, 5], [0, 1, 5]])),
        (np.arange(4).reshape((2, 2)), (3, 3), "down", "left", 7, np.array([[7, 0, 1], [7, 2, 3], [7, 7, 7]])),
        (np.arange(20).reshape((4, 5)), (3, 3), "down", "left", 3, None),
        (np.arange(60).reshape((6, 10)), (11, 11), "even", "even", 3, None),
        (np.ones((167, 210)), (256, 256), "even", "even", 3, None),
        (np.arange(6).reshape((2, 3)), (2, 2), "down", "even", 9, np.array([[0, 1, 2, 9], [3, 4, 5, 9]])),
        (
            np.arange(6).reshape((3, 2)),
            (4, 4),
            "down",
            "even",
            9,
            np.array([[9, 0, 1, 9], [9, 2, 3, 9], [9, 4, 5, 9], [9, 9, 9, 9]]),
        ),
    ],
)
def test_constant_pad(
    array: np.ndarray,
    multiple_of: Tuple[int, int],
    up_down_rule: Literal["even", "up", "down"],
    left_right_rule: Literal["even", "left", "right"],
    pad_value: float,
    expected_result: Optional[np.ndarray],
):
    """Checks that the function pads correctly and minimally. In larger cases only the shapes are checked."""
    padded = constant_pad(array, multiple_of, up_down_rule, left_right_rule, pad_value)
    if expected_result is not None:
        assert_array_equal(padded, expected_result)

    # correct amount of padding is present
    assert np.sum(padded == pad_value) - np.sum(array == pad_value) == np.prod(padded.shape) - np.prod(array.shape)
    for dim in (0, 1):
        assert (padded.shape[dim] - array.shape[dim]) // multiple_of[dim] == 0  # least amount of padding
        assert padded.shape[dim] % multiple_of[dim] == 0  # is divisible
