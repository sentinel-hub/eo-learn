"""
Module with global fixtures

Copyright (c) 2017- Sinergise and contributors

For the full list of contributors, see the CREDITS file in the root directory of this source tree.
This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import pytest

from eolearn.core import EOPatch


@pytest.fixture(scope="session", name="test_eopatch_path")
def test_eopatch_path_fixture() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "example_data", "TestEOPatch")


@pytest.fixture(name="test_eopatch")
def test_eopatch_fixture(test_eopatch_path) -> EOPatch:
    return EOPatch.load(test_eopatch_path)
