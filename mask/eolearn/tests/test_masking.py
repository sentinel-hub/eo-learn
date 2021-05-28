"""
Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Jernej Puc, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy
import os
import unittest

import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.mask import MaskFeature


class TestMaskFeature(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    @classmethod
    def setUpClass(cls):
        cls.eop = EOPatch.load(cls.TEST_PATCH_FILENAME)

        cls.bands_feature = FeatureType.DATA, 'BANDS-S2-L1C'
        cls.ndvi_feature = FeatureType.DATA, 'NDVI'
        cls.cloud_mask_feature = FeatureType.MASK, 'CLM'
        cls.lulc_feature = FeatureType.MASK_TIMELESS, 'LULC'

    def test_bands_with_clm(self):
        eop = copy.copy(self.eop)
        new_feature = FeatureType.DATA, 'BANDS-S2-L1C_MASKED'

        mask_task = MaskFeature(self.bands_feature, self.cloud_mask_feature, mask_values=[True], no_data_value=-1)
        eop = mask_task(eop)

        masked_count = np.count_nonzero(eop[new_feature] == -1)
        clm_count = np.count_nonzero(eop[self.cloud_mask_feature])
        bands_num = eop[self.bands_feature].shape[-1]
        self.assertEqual(masked_count, clm_count * bands_num)

    def test_ndvi_with_clm(self):
        eop = copy.copy(self.eop)
        new_feature = FeatureType.DATA, 'NDVI_MASKED'

        mask_task = MaskFeature(self.ndvi_feature, self.cloud_mask_feature, mask_values=[True])
        eop = mask_task(eop)

        masked_count = np.count_nonzero(np.isnan(eop[new_feature]))
        clm_count = np.count_nonzero(eop[self.cloud_mask_feature])
        self.assertEqual(masked_count, clm_count)

    def test_clm_with_lulc(self):
        eop = copy.copy(self.eop)
        new_feature = FeatureType.MASK, 'CLM_MASKED'

        mask_task = MaskFeature(self.cloud_mask_feature, self.lulc_feature, mask_values=[2], no_data_value=255)
        eop = mask_task(eop)

        masked_count = np.count_nonzero(eop[new_feature] == 255)
        lulc_count = np.count_nonzero(eop[self.lulc_feature] == 2)
        bands_num = eop[self.cloud_mask_feature].shape[-1]
        time_num = eop[self.cloud_mask_feature].shape[0]
        self.assertEqual(masked_count, lulc_count * time_num * bands_num)

    def test_lulc_with_lulc(self):
        eop = copy.copy(self.eop)
        new_feature = FeatureType.MASK_TIMELESS, 'LULC_MASKED'

        mask_task = MaskFeature(self.lulc_feature, self.lulc_feature, mask_values=[1], no_data_value=100)
        eop = mask_task(eop)

        masked_count = np.count_nonzero(eop[new_feature] == 100)
        lulc_count = np.count_nonzero(eop[self.lulc_feature] == 1)
        self.assertEqual(masked_count, lulc_count)

    def test_wrong_arguments(self):
        with self.assertRaises(ValueError):
            MaskFeature(self.bands_feature, self.cloud_mask_feature, mask_values=10)


if __name__ == '__main__':
    unittest.main()
