"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
from eolearn.features import HaralickTask

FEATURE = (FeatureType.DATA, "NDVI", "haralick")


@pytest.mark.parametrize(
    "task, expected_min, expected_max, expected_mean, expected_median",
    (
        [
            HaralickTask(FEATURE, texture_feature="contrast", angle=0, levels=255, window_size=3),
            3.5,
            9079.0,
            965.8295,
            628.5833,
        ],
        [
            HaralickTask(FEATURE, texture_feature="sum_of_square_variance", angle=np.pi / 2, levels=8, window_size=5),
            0.96899,
            48.7815,
            23.0229,
            23.8987,
        ],
        [
            HaralickTask(FEATURE, texture_feature="sum_entropy", angle=-np.pi / 2, levels=8, window_size=7),
            0,
            1.7463,
            0.5657,
            0.5055,
        ],
    ),
)
def test_haralick(small_ndvi_eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in small_ndvi_eopatch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4

    haralick = eopatch.data["haralick"]
    assert np.min(haralick) == approx(expected_min, abs=delta)
    assert np.max(haralick) == approx(expected_max, abs=delta)
    assert np.mean(haralick) == approx(expected_mean, abs=delta)
    assert np.median(haralick) == approx(expected_median, abs=delta)
