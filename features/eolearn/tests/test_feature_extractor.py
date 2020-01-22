"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest

from eolearn.features import feature_extractor as fe
from eolearn.features import FeatureExtractionTask
from eolearn.core import EOPatch, FeatureType

import numpy as np


class TestFeatureExtendedExtractor(unittest.TestCase):
    def test_simple(self):
        x = [1.0] * 13
        fee = fe.FeatureExtendedExtractor("B8A ; B09 ; B08 ; I(B02, B03) ; S(B05, B03) ; R(B01, B02) ; D(B01, B02, B03)"
                                          " ; I(B8A, B04)")
        self.assertEqual(fee(x), [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0])

    def test_nested(self):
        x = [1.0] * 13
        fee = fe.FeatureExtendedExtractor("D(D(B8, B7, B2), D(B1, B2, B3), R(B10, B8A))")
        self.assertEqual(fee(x), [4.0])


class TestFeatureExtractionTask(unittest.TestCase):
    def test_add_ndvi(self):
        a = np.arange(2 * 3 * 3 * 13).reshape(2, 3, 3, 13)
        eop = EOPatch()
        eop[FeatureType.DATA]['bands'] = a

        eotask_ndvi = FeatureExtractionTask((FeatureType.DATA, 'bands', 'ndvi'), 'I(B4, B8A)')

        eop_ndvi = eotask_ndvi(eop)

        in_shape = eop.data['bands'].shape
        out_shape = in_shape[:-1] + (1,)

        self.assertEqual(eop_ndvi.data['ndvi'].shape, out_shape)


if __name__ == '__main__':
    unittest.main()
