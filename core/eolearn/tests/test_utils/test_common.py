"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import warnings
from functools import partial
from typing import Callable, Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core.utils.common import _apply_to_spatial_axes, is_discrete_type

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    DTYPE_TEST_CASES = [
        (int, True),
        (bool, True),
        (float, False),
        (str, False),
        (bytes, False),
        (complex, False),
        (np.number, False),
        (np.int, True),
        (np.byte, True),
        (np.bool_, True),
        (np.bool, True),
        (np.bool8, True),
        (np.integer, True),
        (np.dtype("uint16"), True),
        (np.int8, True),
        (np.longlong, True),
        (np.int64, True),
        (np.double, False),
        (np.float64, False),
        (np.datetime64, False),
        (np.object_, False),
        (np.generic, False),
    ]


@pytest.mark.parametrize("number_type, is_discrete", DTYPE_TEST_CASES)
def test_is_discrete_type(number_type, is_discrete):
    """Checks the given type and its numpy dtype against the expected answer."""
    assert is_discrete_type(number_type) is is_discrete

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        numpy_dtype = np.dtype(number_type)

    assert is_discrete_type(numpy_dtype) is is_discrete


@pytest.mark.parametrize(
    "function, data, spatial_axes, expected",
    [
        pytest.param(
            partial(np.resize, new_shape=(2, 2)), np.zeros(shape=0), (1, 2), np.zeros((2, 2)), id="minimal test"
        ),
        pytest.param(
            partial(np.resize, new_shape=(3, 2)),
            np.ones((2, 3, 4, 1)),
            (0, 1),
            np.ones((3, 2, 4, 1)),
            id="common test case",
        ),
        pytest.param(
            partial(np.resize, new_shape=(5, 6)),
            np.ones((2, 3, 4, 1)),
            (0, 2),
            np.ones((5, 3, 6, 1)),
            id="test bigger new shape",
        ),
        pytest.param(
            partial(np.resize, new_shape=(1, 2)),
            np.ones((2, 3, 4, 1)),
            (3, 2),
            np.ones((2, 3, 1, 2)),
            id="test reverse axis",
        ),
        pytest.param(
            partial(np.resize, new_shape=(2, 2)),
            np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
            (1, 2),
            np.array([[[[0], [1]], [[2], [3]]], [[[12], [13]], [[14], [15]]]]),
            id="test applying function",
        ),
        pytest.param(
            partial(np.flip, axis=0),
            np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
            (1, 2),
            np.array(
                [
                    [[[8], [9], [10], [11]], [[4], [5], [6], [7]], [[0], [1], [2], [3]]],
                    [[[20], [21], [22], [23]], [[16], [17], [18], [19]], [[12], [13], [14], [15]]],
                ]
            ),
            id="test applying axis dependent function",
        ),
        pytest.param(
            lambda x: x + 1,
            np.arange(24).reshape((2, 3, 4, 1)),
            (1, 2),
            np.arange(1, 25).reshape((2, 3, 4, 1)),
            id="test applying not numpy function",
        ),
    ],
)
def test_apply_to_spatial_axes(
    function: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    spatial_axes: Tuple[int, int],
    expected: np.ndarray,
) -> None:
    assert_array_equal(_apply_to_spatial_axes(function, data, spatial_axes), expected)
