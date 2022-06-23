"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import warnings

import numpy as np
import pytest

from eolearn.core.utils.common import is_discrete_type

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
