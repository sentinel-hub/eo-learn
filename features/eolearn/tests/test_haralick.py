"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2019 Matej Aleksandrov (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy

import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_array_equal

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
from eolearn.features import HaralickTask


@pytest.fixture(name='eopatch')
def eopatch_fixture(test_eopatch):
    ndvi = test_eopatch.data['ndvi'][:10]
    ndvi[np.isnan(ndvi)] = 0
    test_eopatch.data['ndvi'] = ndvi
    return test_eopatch


FEATURE = (FeatureType.DATA, 'ndvi', 'haralick')


@pytest.mark.parametrize('task, expected_min, expected_max, expected_mean, expected_median', (
    [
        HaralickTask(
            FEATURE, texture_feature='contrast', angle=0, levels=255, window_size=3
        ),
        0.0, 15620.8333, 1585.0905, 1004.9167
    ],
    [
        HaralickTask(
            FEATURE, texture_feature='sum_of_square_variance', angle=np.pi/2, levels=8, window_size=5
        ),
        7.7174, 48.7814, 31.9490, 25.0357
    ],
    [
        HaralickTask(
            FEATURE, texture_feature='sum_entropy', angle=-np.pi/2, levels=8, window_size=7
        ),
        0, 1.2971, 0.3898, 0.4019
    ],
))
def test_haralick(eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    initial_patch = copy.deepcopy(eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in initial_patch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4

    haralick = eopatch.data['haralick']
    assert np.min(haralick) == approx(expected_min, abs=delta)
    assert np.max(haralick) == approx(expected_max, abs=delta)
    assert np.mean(haralick) == approx(expected_mean, abs=delta)
    assert np.median(haralick) == approx(expected_median, abs=delta)
