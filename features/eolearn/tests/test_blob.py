"""
Module for computing blobs in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2019 Matej Aleksandrov, Devis Peresutti (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy
import sys

import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_array_equal
from skimage.feature import blob_dog

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
from eolearn.features import BlobTask, DoGBlobTask, LoGBlobTask, DoHBlobTask


FEATURE = (FeatureType.DATA, 'ndvi', 'blob')


@pytest.fixture(name='eopatch')
def eopatch_fixture(test_eopatch):
    ndvi = test_eopatch.data['ndvi'][:10]
    ndvi[np.isnan(ndvi)] = 0
    test_eopatch.data['ndvi'] = ndvi
    return test_eopatch


def test_blob_feature(eopatch):
    BlobTask(FEATURE, blob_dog, sigma_ratio=1.6, min_sigma=1, max_sigma=30, overlap=0.5, threshold=0)(eopatch)
    DoGBlobTask((FeatureType.DATA, 'ndvi', 'blob_dog'), threshold=0)(eopatch)
    assert eopatch.data['blob'] == approx(eopatch.data['blob_dog']), \
        'DoG derived class result not equal to base class result'


HARALICK_TESTS = [
    [DoGBlobTask(FEATURE, threshold=0), 0.0, 37.9625, 0.0545, 0.0],
    [DoHBlobTask(FEATURE, num_sigma=5, threshold=0), 0.0, 1.4142, 0.0007, 0.0]
]
if sys.version_info >= (3, 8):  # For Python 3.7 scikit-image returns less accurate result for this test
    HARALICK_TESTS.append([LoGBlobTask(FEATURE, log_scale=True, threshold=0), 0, 13.65408, 0.05768, 0.0])


@pytest.mark.parametrize('task, expected_min, expected_max, expected_mean, expected_median', HARALICK_TESTS)
def test_haralick(eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    initial_patch = copy.deepcopy(eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in initial_patch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4

    blob = eopatch.data[FEATURE[-1]]
    assert np.min(blob) == approx(expected_min, abs=delta)
    assert np.max(blob) == approx(expected_max, abs=delta)
    assert np.mean(blob) == approx(expected_mean, abs=delta)
    assert np.median(blob) == approx(expected_median, abs=delta)
