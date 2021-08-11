"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc (Sinergise)
Copyright (c) 2017-2019 Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os

import pytest
import numpy as np
from eolearn.core import FeatureType
from eolearn.mask import SnowMask, TheiaSnowMask


@pytest.mark.parametrize('params', [
    {'dem_params': (100, 100, 100)},
    {'red_params': 45},
    {'ndsi_params': (0.2, 3)}
])
def test_raises_errors(params, test_eopatch):
    with pytest.raises(ValueError):
        theia_mask = TheiaSnowMask(
            (FeatureType.DATA, 'BANDS-S2-L1C'), [2, 3, 11], (FeatureType.MASK, 'CLM'),
            (FeatureType.DATA_TIMELESS, 'DEM'), **params
        )
        theia_mask(test_eopatch)


@pytest.mark.parametrize('task, result', [
    (SnowMask((FeatureType.DATA, 'BANDS-S2-L1C'), [2, 3, 7, 11], mask_name='TEST_SNOW_MASK'),
     (50468, 1405)),
    (TheiaSnowMask((FeatureType.DATA, 'BANDS-S2-L1C'), [2, 3, 11], (FeatureType.MASK, 'CLM'),
                   (FeatureType.DATA_TIMELESS, 'DEM'), b10_index=10, mask_name='TEST_THEIA_SNOW_MASK'),
     (60682, 10088))
])
def test_snow_coverage(task, result, test_eopatch):
    resulting_eopatch = task(test_eopatch)
    output = resulting_eopatch[task.mask_feature]

    assert output.ndim == 4
    assert output.shape[:-1] == test_eopatch.data['BANDS-S2-L1C'].shape[:-1]
    assert output.shape[-1] == 1

    assert output.dtype == bool

    snow_pixels = np.sum(output, axis=(1, 2, 3))
    assert np.sum(snow_pixels) == result[0], 'Sum of snowy pixels does not match'
    assert snow_pixels[-4] == result[1], 'Snowy pixels on specified frame do not match'
