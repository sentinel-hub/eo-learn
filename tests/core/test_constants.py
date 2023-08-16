"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import warnings

import pytest

from eolearn.core.constants import FeatureType
from eolearn.core.exceptions import EODeprecationWarning

# ruff: noqa:B018

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    TEST_CASES = [
        (FeatureType.TIMESTAMP, FeatureType.TIMESTAMPS),
        (FeatureType["TIMESTAMP"], FeatureType["TIMESTAMPS"]),
        (FeatureType("timestamp"), FeatureType("timestamps")),
    ]


@pytest.mark.parametrize(
    ("old_ftype", "new_ftype"),
    TEST_CASES,
    ids=["attribute access", "name access", "value access"],
)
def test_timestamp_featuretype(old_ftype, new_ftype) -> None:
    assert old_ftype is new_ftype


def test_timestamps_bbox_deprecation() -> None:
    with warnings.catch_warnings():  # make warnings errors
        warnings.simplefilter("error")

        FeatureType.DATA
        FeatureType["MASK"]
        FeatureType("label")
        FeatureType(FeatureType.META_INFO)

        with pytest.warns(EODeprecationWarning):
            FeatureType.TIMESTAMPS
        with pytest.warns(EODeprecationWarning):
            FeatureType["BBOX"]
        with pytest.warns(EODeprecationWarning):
            FeatureType("bbox")
