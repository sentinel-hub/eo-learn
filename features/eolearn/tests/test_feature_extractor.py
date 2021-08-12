"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import numpy as np

from eolearn.features import feature_extractor as fe
from eolearn.features import FeatureExtractionTask
from eolearn.core import EOPatch, FeatureType


def test_simple():
    x = [1.0] * 13
    fee = fe.FeatureExtendedExtractor(
        "B8A ; B09 ; B08 ; I(B02, B03) ; S(B05, B03) ; R(B01, B02) ; D(B01, B02, B03) ; I(B8A, B04)"
        )
    assert fee(x) == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0]


def test_nested():
    x = [1.0] * 13
    fee = fe.FeatureExtendedExtractor("D(D(B8, B7, B2), D(B1, B2, B3), R(B10, B8A))")
    assert fee(x) == [4.0]


def test_add_ndvi():
    array = np.arange(2 * 3 * 3 * 13).reshape(2, 3, 3, 13)
    eopatch = EOPatch()
    eopatch[FeatureType.DATA]['bands'] = array

    eotask_ndvi = FeatureExtractionTask((FeatureType.DATA, 'bands', 'ndvi'), 'I(B4, B8A)')

    eopatch_ndvi = eotask_ndvi(eopatch)

    in_shape = eopatch.data['bands'].shape
    out_shape = in_shape[:-1] + (1,)

    assert eopatch_ndvi.data['ndvi'].shape == out_shape
