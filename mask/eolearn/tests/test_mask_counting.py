"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2019 Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pytest
import numpy as np

from eolearn.mask import ClassFrequencyTask
from eolearn.core import EOPatch, FeatureType

IN_FEATURE = (FeatureType.MASK, 'TEST')
OUT_FEATURE = (FeatureType.DATA_TIMELESS, 'FREQ')


@pytest.mark.parametrize('classes, no_data_value', (
    (['a', 'b'], 0), (4, 0), (None, 0), ([1, 2, 3], 2)
))
def test_value_error(classes, no_data_value):
    with pytest.raises(ValueError):
        ClassFrequencyTask(IN_FEATURE, OUT_FEATURE, classes=classes, no_data_value=no_data_value)


def test_class_frequency():
    shape = (20, 5, 5, 2)

    data = np.random.randint(0, 5, size=shape)
    data[:, 0, 0, 0] = 0
    data[:, 0, 1, 0] = 2

    eopatch = EOPatch()
    eopatch[IN_FEATURE] = data

    eopatch = ClassFrequencyTask(IN_FEATURE, OUT_FEATURE, [2, 1, 55])(eopatch)
    result = eopatch[OUT_FEATURE]

    # all zeros through the temporal dimension should result in nan
    assert np.isnan(result[0, 0, 0])

    # frequency of 2 should be 1
    assert result[0, 1, 0] == 1

    # frequency of 55 should be 0
    assert np.all(result[1:, :, -2] == 0)

    assert result.shape == (5, 5, 6)
