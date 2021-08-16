"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from eolearn.core import FeatureType
from eolearn.geometry import ErosionTask


CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


@pytest.mark.parametrize('invalid_input', [None, 0, 'a'])
def test_erosion_value_error(invalid_input):
    with pytest.raises(ValueError):
        ErosionTask((FeatureType.MASK_TIMELESS, 'LULC', 'TEST'), disk_radius=invalid_input)


def test_erosion_full(test_eopatch):
    mask_before = test_eopatch.mask_timeless['LULC'].copy()

    erosion_task = ErosionTask((FeatureType.MASK_TIMELESS, 'LULC', 'LULC_ERODED'), 1)
    eopatch = erosion_task.execute(test_eopatch)

    mask_after = eopatch.mask_timeless['LULC_ERODED'].copy()

    assert not np.all(mask_before == mask_after)

    for label in CLASSES:
        if label == 0:
            assert np.sum(mask_after == label) >= np.sum(mask_before == label), 'Error in the erosion process'
        else:
            assert np.sum(mask_after == label) <= np.sum(mask_before == label), 'Error in the erosion process'


def test_erosion_partial(test_eopatch):
    mask_before = test_eopatch.mask_timeless['LULC'].copy()

    # skip forest and artificial surface
    specific_labels = [0, 1, 3, 4]
    erosion_task = ErosionTask(
        mask_feature=(FeatureType.MASK_TIMELESS, 'LULC', 'LULC_ERODED'),
        disk_radius=1,
        erode_labels=specific_labels
    )
    eopatch = erosion_task.execute(test_eopatch)

    mask_after = eopatch.mask_timeless['LULC_ERODED'].copy()

    assert not np.all(mask_before == mask_after)

    for label in CLASSES:
        if label == 0:
            assert np.sum(mask_after == label) >= np.sum(mask_before == label), 'Error in the erosion process'
        elif label in specific_labels:
            assert np.sum(mask_after == label) <= np.sum(mask_before == label), 'Error in the erosion process'
        else:
            assert_array_equal(mask_after == label, mask_before == label, err_msg='Error in the erosion process')
