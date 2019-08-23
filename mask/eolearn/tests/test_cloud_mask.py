"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pytest
import unittest
import os
import numpy as np

from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector
from eolearn.core import EOPatch, FeatureType


class TestAddSentinelHubCloudMaskTask(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    FEATURES_TO_LOAD = [
        (FeatureType.DATA, 'BANDS-S2-L1C', 'CLP'),
        (FeatureType.MASK, 'CLM', 'IS_DATA'),
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
        classifier = get_s2_pixel_cloud_detector(all_bands=True)
        add_cm = AddCloudMaskTask(classifier, 'bands', cmask_feature='clm')
        self.assertRaises(ValueError, add_cm, self.eop)

    def _check_shape(self, output, data):
        """Checks that shape of data and output match."""

        self.assertTrue(output.ndim == 4)
        self.assertTrue(output.shape[:-1] == data.shape[:-1])
        self.assertTrue(output.shape[-1] == 1)

    def test_cloud_coverage(self):
        classifier = get_s2_pixel_cloud_detector(all_bands=True)
        # Classifier is run on same resolution as data array
        add_cm = AddCloudMaskTask(classifier, 'ALL_DATA', cmask_feature='CLM_TEST', cprobs_feature='CLP_TEST')
        eop_clm = add_cm(self.eop)

        # Check shape and type
        self._check_shape(eop_clm.mask['CLM_TEST'], eop_clm.data['ALL_DATA'])
        self._check_shape(eop_clm.data['CLP_TEST'], eop_clm.data['ALL_DATA'])
        self.assertTrue(eop_clm.mask['CLM_TEST'].dtype == np.bool)
        self.assertTrue(eop_clm.data['CLP_TEST'].dtype == np.float32)

        # Compare mean cloud coverage with provided reference
        mean_clm_provided = np.mean(eop_clm.mask['CLM'])
        mean_clp_provided = np.mean(eop_clm.data['CLP'])
        self.assertAlmostEqual(np.mean(eop_clm.mask['CLM_TEST']), mean_clm_provided, places=1)
        self.assertAlmostEqual(np.mean(eop_clm.data['CLP_TEST']), mean_clp_provided, places=2)

        # Classifier is run on downscaled version of data array
        add_cm = AddCloudMaskTask(classifier, 'ALL_DATA', cm_size_y='20m', cm_size_x='20m',
                                  cmask_feature='CLM_TEST', cprobs_feature='CLP_TEST')
        eop_clm = add_cm(self.eop)
        
        # Check shape and type
        self._check_shape(eop_clm.mask['CLM_TEST'], eop_clm.data['ALL_DATA'])
        self._check_shape(eop_clm.data['CLP_TEST'], eop_clm.data['ALL_DATA'])
        self.assertTrue(eop_clm.mask['CLM_TEST'].dtype == np.bool)
        self.assertTrue(eop_clm.data['CLP_TEST'].dtype == np.float32)

        # Compare mean cloud coverage with provided reference
        mean_clm_provided = np.mean(eop_clm.mask['CLM'])
        mean_clp_provided = np.mean(eop_clm.data['CLP'])
        self.assertAlmostEqual(np.mean(eop_clm.mask['CLM_TEST']), mean_clm_provided, places=2)
        self.assertAlmostEqual(np.mean(eop_clm.data['CLP_TEST']), mean_clp_provided, places=2)

        # Check if same times are flagged as cloudless
        cloudless = np.mean(eop_clm.mask['CLM_TEST'], axis=(1,2,3)) == 0
        self.assertTrue(np.all(cloudless == eop_clm.label['IS_CLOUDLESS'][:,0]))

    @pytest.mark.xfail
    def test_wms_request(self):
        classifier = get_s2_pixel_cloud_detector(all_bands=False)
        # Classifier is run on new request of data array
        add_cm = AddCloudMaskTask(classifier, 'BANDS-S2-L1C', cm_size_x='20m', cm_size_y='20m',
                                  cmask_feature='CLM_TEST', cprobs_feature='CLP_TEST')
        eop_clm = add_cm(self.eop)

        # Check shape and type
        self._check_shape(eop_clm.mask['CLM_TEST'], eop_clm.data['ALL_DATA'])
        self._check_shape(eop_clm.data['CLP_TEST'], eop_clm.data['ALL_DATA'])
        self.assertTrue(eop_clm.mask['CLM_TEST'].dtype == np.bool)
        self.assertTrue(eop_clm.data['CLP_TEST'].dtype == np.float32)

        # Compare mean cloud coverage with provided reference
        mean_clm_provided = np.mean(eop_clm.mask['CLM'])
        mean_clp_provided = np.mean(eop_clm.data['CLP'])
        self.assertAlmostEqual(np.mean(eop_clm.mask['CLM_TEST']), mean_clm_provided, places=2)
        self.assertAlmostEqual(np.mean(eop_clm.data['CLP_TEST']), mean_clp_provided, places=2)


if __name__ == '__main__':
    unittest.main()
