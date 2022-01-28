"""
Module with global fixtures
"""
import os

import pytest

from eolearn.core import EOPatch


@pytest.fixture(scope="session", name="test_eopatch_path")
def test_eopatch_path_fixture():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data", "TestEOPatch")


@pytest.fixture(name="test_eopatch")
def test_eopatch_fixture(test_eopatch_path):
    return EOPatch.load(test_eopatch_path)
