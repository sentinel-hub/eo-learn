"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import unittest

import numpy as np
from eolearn.core import EOPatch, FeatureType
from eolearn.mask import CloudMaskTask


class TestAddMultiCloudMaskTask(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    FEATURES_TO_LOAD = [
        (FeatureType.DATA, 'BANDS-S2-L1C', 'CLP_S2C'),
        (FeatureType.MASK, 'CLM_S2C', 'IS_DATA'),
        (FeatureType.LABEL, 'IS_CLOUDLESS'),
        FeatureType.META_INFO,
        FeatureType.TIMESTAMP,
        FeatureType.BBOX
    ]

    @classmethod
    def setUpClass(cls):
        cls.eop = EOPatch.load(cls.TEST_PATCH_FILENAME, features=cls.FEATURES_TO_LOAD)
        cls.eop.rename_feature(FeatureType.DATA, 'BANDS-S2-L1C', 'ALL_DATA')

    def test_raises_errors(self):
        add_tcm = CloudMaskTask(data_feature='bands')
        self.assertRaises(ValueError, add_tcm, self.eop)

    def _check_shape(self, output, data):
        """Checks that shape of data and output match."""

        self.assertTrue(output.ndim == 4)
        self.assertTrue(output.shape[:-1] == data.shape[:-1])
        self.assertTrue(output.shape[-1] == 1)

    def test_cloud_coverage(self):
        # Classifier is run on same resolution as data array
        add_tcm = CloudMaskTask(data_feature='ALL_DATA',
                                all_bands=True,
                                is_data_feature='IS_DATA',
                                mono_features=('CLP_TEST', 'CLM_TEST'),
                                mask_feature=None,
                                average_over=4,
                                dilation_size=2,
                                mono_threshold=0.4)
        eop_clm = add_tcm(self.eop)

        # the masks and pseudo-probabilities should be equal
        self.assertTrue(np.array_equal(eop_clm.mask['CLM_TEST'], eop_clm.mask['CLM_S2C']))
        self.assertTrue(np.array_equal(eop_clm.data['CLP_TEST'], eop_clm.data['CLP_S2C']))

    def test_cloud_coverage_downscaled(self):
        # Classifier is run on downscaled version of data array
        add_tcm = CloudMaskTask(data_feature='ALL_DATA',
                                processing_resolution=120,
                                mono_features=('CLP_TEST', 'CLM_TEST'),
                                mask_feature=None,
                                average_over=8,
                                dilation_size=4)
        eop_clm = add_tcm(self.eop)

        # Check shape and type
        self._check_shape(eop_clm.mask['CLM_TEST'], eop_clm.data['ALL_DATA'])
        self._check_shape(eop_clm.data['CLP_TEST'], eop_clm.data['ALL_DATA'])
        self.assertTrue(eop_clm.mask['CLM_TEST'].dtype == bool)
        self.assertTrue(eop_clm.data['CLP_TEST'].dtype == np.float32)

        # Compare mean cloud coverage with provided reference
        self.assertAlmostEqual(np.mean(eop_clm.mask['CLM_TEST']), np.mean(eop_clm.mask['CLM_S2C']), places=1)
        self.assertAlmostEqual(np.mean(eop_clm.data['CLP_TEST']), np.mean(eop_clm.data['CLP_S2C']), places=1)

        # Check if most of the same times are flagged as cloudless
        cloudless = np.mean(eop_clm.mask['CLM_TEST'], axis=(1, 2, 3)) == 0
        self.assertTrue(np.mean(cloudless == eop_clm.label['IS_CLOUDLESS'][:, 0]) > 0.94)


if __name__ == '__main__':
    unittest.main()
