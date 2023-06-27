"""
Module with global fixtures

Copyright (c) 2017- Sinergise and contributors

For the full list of contributors, see the CREDITS file in the root directory of this source tree.
This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from eolearn.core import EOPatch

pytest.register_assert_rewrite("sentinelhub.testing_utils")  # makes asserts in helper functions work with pytest


@pytest.fixture(scope="session", name="example_data_path")
def example_data_path_fixture() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data")


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data", "TestEOPatch")
    return EOPatch.load(path, lazy_loading=True)


@pytest.fixture(name="small_ndvi_eopatch")
def small_ndvi_eopatch_fixture(example_eopatch: EOPatch):
    ndvi = example_eopatch.data["NDVI"][:, :20, :20]
    ndvi[np.isnan(ndvi)] = 0
    example_eopatch.data["NDVI"] = ndvi
    example_eopatch.consolidate_timestamps(example_eopatch.get_timestamps()[:10])
    return example_eopatch
