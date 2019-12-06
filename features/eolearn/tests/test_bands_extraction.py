"""
A collection of bands extraction EOTasks

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.features import EuclideanNormTask, NormalizedDifferenceIndexTask


class TestBandsExtraction(unittest.TestCase):
    def test_euclidean_norm(self):
        eopatch = EOPatch()

        data = np.zeros(5*10*10*7).reshape(5, 10, 10, 7)
        bands = [0, 1, 2, 4, 6]
        data[..., bands] = 1

        eopatch.data['TEST'] = data

        eopatch = EuclideanNormTask((FeatureType.DATA, 'TEST'), (FeatureType.DATA, 'NORM'), bands)(eopatch)
        self.assertTrue(np.all(eopatch.data['NORM'] == np.sqrt(len(bands))))

    def test_ndi(self):
        eopatch = EOPatch(data={'TEST': np.zeros((4, 3, 3, 9))})

        f_in, f_out = (FeatureType.DATA, 'TEST'), (FeatureType.DATA, 'NDI')
        self.assertRaises(ValueError, NormalizedDifferenceIndexTask, f_in, f_out, bands=[1, 2, 3])
        self.assertRaises(ValueError, NormalizedDifferenceIndexTask, f_in, f_out, bands='test')

        band_a, band_b = 4.123, 3.321
        eopatch.data['TEST'][..., 0] = band_a
        eopatch.data['TEST'][..., 1] = band_b
        eopatch = NormalizedDifferenceIndexTask(f_in, f_out, bands=[0, 1])(eopatch)
        self.assertTrue(np.all(eopatch.data['NDI'] == ((band_a - band_b) / (band_a + band_b))))

        eopatch.data['TEST'][..., 5] = np.nan
        eopatch.data['TEST'][..., 7] = np.inf
        eopatch = NormalizedDifferenceIndexTask(
            (FeatureType.DATA, 'TEST'), (FeatureType.DATA, 'NAN_INF_INPUT'), bands=[5, 7]
        ).execute(eopatch)
        self.assertTrue(np.all(np.isnan(eopatch.data['NAN_INF_INPUT'])))

        eopatch.data['TEST'][..., 1] = 1
        eopatch.data['TEST'][..., 3] = -1
        eopatch = NormalizedDifferenceIndexTask(
            (FeatureType.DATA, 'TEST'), (FeatureType.DATA, 'DIV_ZERO_NAN'), bands=[1, 3]
        ).execute(eopatch)

        self.assertTrue(np.all(np.isnan(eopatch.data['DIV_ZERO_NAN'])))

        eopatch = NormalizedDifferenceIndexTask(
            (FeatureType.DATA, 'TEST'), (FeatureType.DATA, 'DIV_ZERO_INT'), bands=[1, 3], undefined_value=123
        ).execute(eopatch)

        self.assertTrue(np.all(eopatch.data['DIV_ZERO_INT'] == 123))


if __name__ == '__main__':
    unittest.main()
