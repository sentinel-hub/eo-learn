"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import dataclasses
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


@dataclasses.dataclass
class ApplyToTestCase:
    function: Callable[[np.ndarray], np.ndarray]
    data: np.ndarray
    spatial_axes: Tuple[int, int]
    expected: np.ndarray


APPLY_TO_TEST_CASES = [
    ApplyToTestCase(
        function=partial(np.resize, new_shape=(2, 2)),
        data=np.zeros(shape=0),
        spatial_axes=(1, 2),
        expected=np.zeros((2, 2)),
    ),
    ApplyToTestCase(
        function=partial(np.resize, new_shape=(3, 2)),
        data=np.ones((2, 3, 4, 1)),
        spatial_axes=(0, 1),
        expected=np.ones((3, 2, 4, 1)),
    ),
    ApplyToTestCase(
        function=partial(np.resize, new_shape=(5, 6)),
        data=np.ones((2, 3, 4, 1)),
        spatial_axes=(0, 2),
        expected=np.ones((5, 3, 6, 1)),
    ),
    ApplyToTestCase(
        function=partial(np.resize, new_shape=(2, 4)),
        data=np.ones((2, 3, 4, 1)),
        spatial_axes=(3, 2),
        expected=np.ones((2, 3, 2, 4)),
    ),
    ApplyToTestCase(
        function=partial(np.resize, new_shape=(2, 4)),
        data=np.ones((2, 3, 4, 1)),
        spatial_axes=(2, 3),
        expected=np.ones((2, 3, 2, 4)),
    ),
    ApplyToTestCase(
        function=partial(np.resize, new_shape=(2, 2)),
        data=np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
        spatial_axes=(1, 2),
        expected=np.array([[[[0], [1]], [[2], [3]]], [[[12], [13]], [[14], [15]]]]),
    ),
    ApplyToTestCase(
        function=partial(np.flip, axis=0),
        data=np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
        spatial_axes=(1, 2),
        expected=np.array(
            [
                [[[8], [9], [10], [11]], [[4], [5], [6], [7]], [[0], [1], [2], [3]]],
                [[[20], [21], [22], [23]], [[16], [17], [18], [19]], [[12], [13], [14], [15]]],
            ]
        ),
    ),
    ApplyToTestCase(
        function=partial(np.flip, axis=0),
        data=np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
        spatial_axes=(2, 1),
        expected=np.array(
            [
                [[[8], [9], [10], [11]], [[4], [5], [6], [7]], [[0], [1], [2], [3]]],
                [[[20], [21], [22], [23]], [[16], [17], [18], [19]], [[12], [13], [14], [15]]],
            ]
        ),
    ),
    ApplyToTestCase(
        function=lambda x: x + 1,
        data=np.arange(24).reshape((2, 3, 4, 1)),
        spatial_axes=(1, 2),
        expected=np.arange(1, 25).reshape((2, 3, 4, 1)),
    ),
]


@pytest.mark.parametrize("test_case", APPLY_TO_TEST_CASES)
def test_apply_to_spatial_axes(test_case: ApplyToTestCase) -> None:
    image = _apply_to_spatial_axes(test_case.function, test_case.data, test_case.spatial_axes)
    assert_array_equal(image, test_case.expected)
