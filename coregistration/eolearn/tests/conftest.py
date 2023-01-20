"""
Module with global fixtures
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
