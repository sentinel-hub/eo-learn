"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

import pytest

from eolearn.core import FeatureType


@pytest.mark.parametrize(
    ("old_ftype", "new_ftype"),
    [
        (FeatureType.TIMESTAMP, FeatureType.TIMESTAMPS),
        (FeatureType["TIMESTAMP"], FeatureType["TIMESTAMPS"]),
        (FeatureType("timestamp"), FeatureType("timestamps")),
    ],
    ids=["attribute access", "name access", "value access"],
)
def test_timestamp_featuretype(old_ftype, new_ftype) -> None:
    assert old_ftype is new_ftype
