"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2019 Matej Aleksandrov, Devis Peresutti (Sinergise)

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
from eolearn.features import LocalBinaryPatternTask


@pytest.fixture(name='eopatch')
def eopatch_fixture(test_eopatch):
    ndvi = test_eopatch.data['ndvi'][:10]
    ndvi[np.isnan(ndvi)] = 0
    test_eopatch.data['ndvi'] = ndvi
    return test_eopatch


@pytest.mark.parametrize('task, expected_min, expected_max, expected_mean, expected_median', (
    [
        LocalBinaryPatternTask((FeatureType.DATA, 'ndvi', 'lbp'), nb_points=24, radius=3),
        0.0, 25.0, 22.3147, 24.0
    ],
))
def test_local_binary_pattern(eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    initial_patch = copy.deepcopy(eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in initial_patch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4

    haralick = eopatch.data['lbp']
    assert np.min(haralick) == approx(expected_min, abs=delta)
    assert np.max(haralick) == approx(expected_max, abs=delta)
    assert np.mean(haralick) == approx(expected_mean, abs=delta)
    assert np.median(haralick) == approx(expected_median, abs=delta)
