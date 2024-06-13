"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType
from eolearn.mask import SnowMaskTask


@pytest.mark.parametrize(
    ("task", "result"),
    [
        (SnowMaskTask((FeatureType.DATA, "BANDS-S2-L1C"), [2, 3, 7, 11], mask_name="TEST_SNOW_MASK"), (50468, 1405)),
    ],
)
def test_snow_coverage(task, result, test_eopatch):
    resulting_eopatch = task(test_eopatch)
    output = resulting_eopatch[task.mask_feature]

    assert output.ndim == 4
    assert output.shape[:-1] == test_eopatch.data["BANDS-S2-L1C"].shape[:-1]
    assert output.shape[-1] == 1

    assert output.dtype == bool

    snow_pixels = np.sum(output, axis=(1, 2, 3))
    assert np.sum(snow_pixels) == result[0], "Sum of snowy pixels does not match"
    assert snow_pixels[-4] == result[1], "Snowy pixels on specified frame do not match"


def test_snow_empty_eopatch(test_eopatch):
    bands = test_eopatch.data["BANDS-S2-L1C"]
    empty_bands_array = np.array([], dtype=bands.dtype).reshape((0, *bands.shape[1:]))
    empty_eopatch = EOPatch(bbox=test_eopatch.bbox, timestamps=[], data={"BANDS-S2-L1C": empty_bands_array})
    task = SnowMaskTask((FeatureType.DATA, "BANDS-S2-L1C"), [2, 3, 7, 11], mask_name="TEST_SNOW_MASK")
    resulting_eopatch = task(empty_eopatch)  # checks if the task runs without errors
    assert resulting_eopatch.mask["TEST_SNOW_MASK"].shape[0] == 0
