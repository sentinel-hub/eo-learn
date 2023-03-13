"""
Module with global fixtures

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import os

import pytest

from eolearn.core import EOPatch

pytest.register_assert_rewrite("sentinelhub.testing_utils")  # makes asserts in helper functions work with pytest

EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data")
EXAMPLE_EOPATCH_PATH = os.path.join(EXAMPLE_DATA_PATH, "TestEOPatch")


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    return EOPatch.load(EXAMPLE_EOPATCH_PATH, lazy_loading=True)
