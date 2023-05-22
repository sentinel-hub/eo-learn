"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import dataclasses
import warnings
from functools import partial
from typing import Callable, Optional, Tuple

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
        (np.byte, True),
        (np.bool_, True),
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


@pytest.mark.parametrize(("number_type", "is_discrete"), DTYPE_TEST_CASES)
def test_is_discrete_type(number_type, is_discrete):
    """Checks the given type and its numpy dtype against the expected answer."""
    assert is_discrete_type(number_type) is is_discrete

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        numpy_dtype = np.dtype(number_type)

    assert is_discrete_type(numpy_dtype) is is_discrete


@dataclasses.dataclass
class ApplyToAxesTestCase:
    function: Callable[[np.ndarray], np.ndarray]
    data: np.ndarray
    spatial_axes: Tuple[int, int]
    expected: Optional[np.ndarray] = None


APPLY_TO_TEST_CASES = [
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(3, 2)),
        data=np.ones((2, 3)),
        spatial_axes=(0, 1),
        expected=np.ones((3, 2)),
    ),
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(4, 3)),
        data=np.ones((2, 3, 2, 3, 1)),
        spatial_axes=(1, 2),
        expected=np.ones((2, 4, 3, 3, 1)),
    ),
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(5, 6)),
        data=np.ones((2, 3, 4)),
        spatial_axes=(0, 2),
        expected=np.ones((5, 3, 6)),
    ),
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(2, 4)),
        data=np.ones((2, 3, 4, 1)),
        spatial_axes=(2, 3),
        expected=np.ones((2, 3, 2, 4)),
    ),
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(2, 2)),
        data=np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
        spatial_axes=(1, 2),
        expected=np.array([[[[0], [1]], [[2], [3]]], [[[12], [13]], [[14], [15]]]]),
    ),
    ApplyToAxesTestCase(
        function=partial(np.flip, axis=0),
        data=np.arange(2 * 3 * 4).reshape((2, 3, 4)),
        spatial_axes=(1, 2),
        expected=np.array(
            [[[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]], [[20, 21, 22, 23], [16, 17, 18, 19], [12, 13, 14, 15]]]
        ),
    ),
    ApplyToAxesTestCase(
        function=lambda x: x + 1,
        data=np.arange(24).reshape((2, 3, 4, 1)),
        spatial_axes=(1, 2),
        expected=np.arange(1, 25).reshape((2, 3, 4, 1)),
    ),
]


@pytest.mark.parametrize("test_case", APPLY_TO_TEST_CASES)
def test_apply_to_spatial_axes(test_case: ApplyToAxesTestCase) -> None:
    image = _apply_to_spatial_axes(test_case.function, test_case.data, test_case.spatial_axes)
    assert_array_equal(image, test_case.expected)


APPLY_TO_FAIL_TEST_CASES = [
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(2, 2)),
        data=np.zeros(shape=0),
        spatial_axes=(1, 2),
    ),
    ApplyToAxesTestCase(
        function=partial(np.resize, new_shape=(2, 4)),
        data=np.ones((2, 3, 4, 1)),
        spatial_axes=(3, 2),
    ),
    ApplyToAxesTestCase(
        function=partial(np.flip, axis=0),
        data=np.arange(2 * 3 * 4).reshape((2, 3, 4, 1)),
        spatial_axes=(2, 2),
    ),
    ApplyToAxesTestCase(
        function=lambda x: x + 1,
        data=np.arange(24).reshape((2, 3, 4, 1)),
        spatial_axes=(1,),
    ),
    ApplyToAxesTestCase(
        function=lambda x: x + 1,
        data=np.arange(24).reshape((2, 3, 4, 1)),
        spatial_axes=(1, 2, 3),
    ),
]


@pytest.mark.parametrize("test_case", APPLY_TO_FAIL_TEST_CASES)
def test_apply_to_spatial_axes_fails(test_case: ApplyToAxesTestCase) -> None:
    with pytest.raises(ValueError):
        _apply_to_spatial_axes(test_case.function, test_case.data, test_case.spatial_axes)
