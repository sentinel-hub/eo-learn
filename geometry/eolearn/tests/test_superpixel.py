"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import dataclasses

import numpy as np
import pytest

from eolearn.core import FeatureType, EOTask
from eolearn.geometry import SuperpixelSegmentation, FelzenszwalbSegmentation, SlicSegmentation


SUPERPIXEL_FEATURE = FeatureType.MASK_TIMELESS, 'SP_FEATURE'


@dataclasses.dataclass(frozen=True)
class SuperpixelTestCase:
    """Class for keeping specifics of each test case."""
    name: str
    task: EOTask
    mask_mean: float
    mask_median: int
    mask_max: int
    mask_min: int = 0


TEST_CASES = (
    SuperpixelTestCase(
        name='base superpixel segmentation',
        task=SuperpixelSegmentation(
            (FeatureType.DATA, 'BANDS-S2-L1C'), SUPERPIXEL_FEATURE, scale=100, sigma=0.5, min_size=100
        ),
        mask_max=25, mask_mean=10.6809, mask_median=11
    ),
    SuperpixelTestCase(
        name='Felzenszwalb segmentation',
        task=FelzenszwalbSegmentation(
            (FeatureType.DATA_TIMELESS, 'MAX_NDVI'), SUPERPIXEL_FEATURE, scale=21, sigma=1.0, min_size=52
        ),
        mask_max=22, mask_mean=8.5302, mask_median=7
        ),
    SuperpixelTestCase(
        name='Felzenszwalb segmentation on mask',
        task=FelzenszwalbSegmentation(
            (FeatureType.MASK, 'CLM'), SUPERPIXEL_FEATURE, scale=1, sigma=0, min_size=15
        ),
        mask_max=171, mask_mean=86.46267, mask_median=90
        ),
    SuperpixelTestCase(
        name='SLIC segmentation',
        task=SlicSegmentation(
            (FeatureType.DATA, 'CLP'), SUPERPIXEL_FEATURE, n_segments=55, compactness=25.0, max_iter=20, sigma=0.8
        ),
        mask_max=48, mask_mean=24.6072, mask_median=25
        ),
    SuperpixelTestCase(
        name='SLIC segmentation on mask',
        task=SlicSegmentation(
            (FeatureType.MASK_TIMELESS, 'RANDOM_UINT8'), SUPERPIXEL_FEATURE,
            n_segments=231, compactness=15.0, max_iter=7, sigma=0.2
        ),
        mask_max=195, mask_mean=100.1844, mask_median=101
        ),
)


@pytest.mark.parametrize('test_case', TEST_CASES)
def test_superpixel(test_eopatch, test_case):
    test_case.task.execute(test_eopatch)
    result = test_eopatch[SUPERPIXEL_FEATURE]

    assert result.dtype == np.int64, "Expected int64 dtype for result"

    delta = 1e-3

    assert np.amin(result) == pytest.approx(test_case.mask_min, delta), 'Minimum values do not match.'
    assert np.amax(result) == pytest.approx(test_case.mask_max, delta), 'Maxmum values do not match.'
    assert np.mean(result) == pytest.approx(test_case.mask_mean, delta), 'Mean values do not match.'
    assert np.median(result) == pytest.approx(test_case.mask_median, delta), 'Median values do not match.'
