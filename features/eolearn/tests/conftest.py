"""
Module with global fixtures
"""
import os

import numpy as np
import pytest

from eolearn.core import EOPatch

EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data")
EXAMPLE_EOPATCH_PATH = os.path.join(EXAMPLE_DATA_PATH, "TestEOPatch")


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    return EOPatch.load(EXAMPLE_EOPATCH_PATH, lazy_loading=True)


@pytest.fixture(name="small_ndvi_eopatch")
def small_ndvi_eopatch_fixture(example_eopatch):
    ndvi = example_eopatch.data["NDVI"][:10, :20, :20]
    ndvi[np.isnan(ndvi)] = 0
    example_eopatch.data["NDVI"] = ndvi
    return example_eopatch
