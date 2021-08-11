"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging

import pytest
import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.ml_tools import MorphologicalOperations, MorphologicalStructFactory, MorphologicalFilterTask

logging.basicConfig(level=logging.DEBUG)

MASK_FEATURE = FeatureType.MASK, 'mask'
MASK_TIMELESS_FEATURE = FeatureType.MASK_TIMELESS, 'timeless_mask'


@pytest.fixture(name='test_eopatch', scope='module')
def test_eopatch_fixture():
    patch = EOPatch()

    mask = np.random.randint(20, size=(10, 100, 100, 3))
    mask_timeless = np.random.randint(20, 50, size=(100, 100, 5))

    patch[MASK_FEATURE] = mask
    patch[MASK_TIMELESS_FEATURE] = mask_timeless
    return patch


@pytest.mark.parametrize('morph_operation', MorphologicalOperations)
@pytest.mark.parametrize('feature', [MASK_FEATURE, MASK_TIMELESS_FEATURE])
@pytest.mark.parametrize('struct_element', [
    None, MorphologicalStructFactory.get_disk(5), MorphologicalStructFactory.get_rectangle(5, 6)
])
def test_postprocessing(test_eopatch, feature, morph_operation, struct_element):
    task = MorphologicalFilterTask(feature, morph_operation, struct_element)
    task.execute(test_eopatch)
