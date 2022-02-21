"""
Module for computing blobs in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx
from skimage.feature import blob_dog

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
from eolearn.features import BlobTask, DoGBlobTask, DoHBlobTask, LoGBlobTask

FEATURE = (FeatureType.DATA, "NDVI", "blob")


def test_blob_feature(small_ndvi_eopatch):
    eopatch = small_ndvi_eopatch
    BlobTask(FEATURE, blob_dog, sigma_ratio=1.6, min_sigma=1, max_sigma=30, overlap=0.5, threshold=0)(eopatch)
    DoGBlobTask((FeatureType.DATA, "NDVI", "blob_dog"), threshold=0)(eopatch)
    assert eopatch.data["blob"] == approx(
        eopatch.data["blob_dog"]
    ), "DoG derived class result not equal to base class result"


BLOB_TESTS = [
    [DoGBlobTask(FEATURE, threshold=0), 0.0, 37.9625, 0.0854, 0.0],
]
if sys.version_info >= (3, 8):  # For Python 3.7 scikit-image returns less accurate result for this test
    BLOB_TESTS.append([DoHBlobTask(FEATURE, num_sigma=5, threshold=0), 0.0, 21.9203, 0.05807, 0.0])
    BLOB_TESTS.append([LoGBlobTask(FEATURE, log_scale=True, threshold=0), 0, 42.4264, 0.0977, 0.0])


@pytest.mark.parametrize("task, expected_min, expected_max, expected_mean, expected_median", BLOB_TESTS)
def test_blob(small_ndvi_eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in small_ndvi_eopatch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4

    blob = eopatch.data[FEATURE[-1]]
    assert np.min(blob) == approx(expected_min, abs=delta)
    assert np.max(blob) == approx(expected_max, abs=delta)
    assert np.mean(blob) == approx(expected_mean, abs=delta)
    assert np.median(blob) == approx(expected_median, abs=delta)
