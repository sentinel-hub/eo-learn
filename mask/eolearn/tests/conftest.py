"""
Module with global fixtures
"""
import os

import pytest

from eolearn.core import EOPatch

EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data")
TEST_EOPATCH_PATH = os.path.join(EXAMPLE_DATA_PATH, "TestEOPatch")


@pytest.fixture(name="test_eopatch")
def test_eopatch_fixture():
    return EOPatch.load(TEST_EOPATCH_PATH, lazy_loading=True)
