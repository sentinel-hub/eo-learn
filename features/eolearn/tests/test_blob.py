"""
Module for computing blobs in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import copy
import sys

import pytest
from pytest import approx
from skimage.feature import blob_dog

from sentinelhub.testing_utils import assert_statistics_match

from eolearn.core import FeatureType
from eolearn.features import BlobTask, DoGBlobTask, DoHBlobTask, LoGBlobTask

FEATURE = (FeatureType.DATA, "NDVI", "blob")
BLOB_FEATURE = (FeatureType.DATA, "blob")


def test_dog_blob_task(small_ndvi_eopatch):
    eopatch = small_ndvi_eopatch
    BlobTask(FEATURE, blob_dog, sigma_ratio=1.6, min_sigma=1, max_sigma=30, overlap=0.5, threshold=0)(eopatch)
    DoGBlobTask((FeatureType.DATA, "NDVI", "blob_dog"), threshold=0)(eopatch)
    assert eopatch[BLOB_FEATURE] == approx(eopatch.data["blob_dog"])


BLOB_TESTS = [
    (DoGBlobTask(FEATURE, threshold=0), {"exp_min": 0.0, "exp_max": 37.9625, "exp_mean": 0.08545, "exp_median": 0.0}),
]
if sys.version_info >= (3, 8):  # For Python 3.7 scikit-image returns less accurate result for this test
    BLOB_TESTS.extend(
        [
            (
                DoHBlobTask(FEATURE, num_sigma=5, threshold=0),
                {"exp_min": 0.0, "exp_max": 21.9203, "exp_mean": 0.05807, "exp_median": 0.0},
            ),
            (
                LoGBlobTask(FEATURE, log_scale=True, threshold=0),
                {"exp_min": 0, "exp_max": 42.4264, "exp_mean": 0.09767, "exp_median": 0.0},
            ),
        ]
    )


@pytest.mark.parametrize("task, expected_statistics", BLOB_TESTS)
def test_blob_task(small_ndvi_eopatch, task, expected_statistics):
    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    assert_statistics_match(eopatch[BLOB_FEATURE], exp_shape=(10, 20, 20, 1), **expected_statistics, abs_delta=1e-4)

    del eopatch[BLOB_FEATURE]
    assert small_ndvi_eopatch == eopatch, "Other features of the EOPatch were affected."
