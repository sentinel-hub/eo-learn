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
from eolearn.features import HOGTask


@pytest.fixture(name='eopatch')
def eopatch_fixture(test_eopatch):
    ndvi = test_eopatch.data['ndvi'][:10]
    ndvi[np.isnan(ndvi)] = 0
    test_eopatch.data['ndvi'] = ndvi
    return test_eopatch


def test_hog(eopatch):
    task = HOGTask(
        (FeatureType.DATA, 'ndvi', 'hog'), orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
        visualize=True, visualize_feature_name='hog_visu'
    )

    initial_patch = copy.deepcopy(eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in initial_patch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4
    for feature, expected_min, expected_max, expected_mean, expected_median in [
        ('hog', 0.0, 0.4427, 0.0564, 0.0),
        ('hog_visu', 0.0, 0.1386, 0.0052, 0.0),
    ]:
        hog = eopatch.data[feature]
        assert np.min(hog) == approx(expected_min, abs=delta)
        assert np.max(hog) == approx(expected_max, abs=delta)
        assert np.mean(hog) == approx(expected_mean, abs=delta)
        assert np.median(hog) == approx(expected_median, abs=delta)
