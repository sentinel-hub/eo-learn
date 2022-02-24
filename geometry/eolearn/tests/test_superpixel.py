"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import numpy as np
import pytest

from eolearn.core import FeatureType
from eolearn.geometry import FelzenszwalbSegmentationTask, SlicSegmentationTask, SuperpixelSegmentationTask

SUPERPIXEL_FEATURE = FeatureType.MASK_TIMELESS, "SP_FEATURE"


@pytest.mark.parametrize(
    "task, expected_min, expected_max, expected_mean, expected_median",
    (
        [
            SuperpixelSegmentationTask(
                (FeatureType.DATA, "BANDS-S2-L1C"), SUPERPIXEL_FEATURE, scale=100, sigma=0.5, min_size=100
            ),
            0,
            25,
            10.6809,
            11,
        ],
        [
            FelzenszwalbSegmentationTask(
                (FeatureType.DATA_TIMELESS, "MAX_NDVI"), SUPERPIXEL_FEATURE, scale=21, sigma=1.0, min_size=52
            ),
            0,
            22,
            8.5302,
            7,
        ],
        [
            FelzenszwalbSegmentationTask((FeatureType.MASK, "CLM"), SUPERPIXEL_FEATURE, scale=1, sigma=0, min_size=15),
            0,
            171,
            86.46267,
            90,
        ],
        [
            SlicSegmentationTask(
                (FeatureType.DATA, "CLP"),
                SUPERPIXEL_FEATURE,
                n_segments=55,
                compactness=25.0,
                max_num_iter=20,
                sigma=0.8,
            ),
            0,
            48,
            24.6072,
            25,
        ],
        [
            SlicSegmentationTask(
                (FeatureType.MASK_TIMELESS, "RANDOM_UINT8"),
                SUPERPIXEL_FEATURE,
                n_segments=231,
                compactness=15.0,
                max_num_iter=7,
                sigma=0.2,
            ),
            0,
            195,
            100.1844,
            101,
        ],
    ),
)
def test_superpixel(test_eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    task.execute(test_eopatch)
    result = test_eopatch[SUPERPIXEL_FEATURE]

    assert result.dtype == np.int64, "Expected int64 dtype for result"

    delta = 1e-3

    assert np.amin(result) == pytest.approx(expected_min, delta), "Minimum values do not match."
    assert np.amax(result) == pytest.approx(expected_max, delta), "Maxmum values do not match."
    assert np.mean(result) == pytest.approx(expected_mean, delta), "Mean values do not match."
    assert np.median(result) == pytest.approx(expected_median, delta), "Median values do not match."
