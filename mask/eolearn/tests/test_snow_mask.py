"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc (Sinergise)
Copyright (c) 2017-2019 Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pytest
import unittest
import os
import numpy as np

from eolearn.mask import SnowMask, TheiaSnowMask
from eolearn.core import EOPatch, FeatureType


class TestSnowMaskingTasks(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    FEATURES_TO_LOAD = [
        (FeatureType.DATA, 'BANDS-S2-L1C'),
        (FeatureType.MASK, 'CLM'),
        (FeatureType.DATA_TIMELESS, 'DEM'),
        FeatureType.META_INFO,
        FeatureType.TIMESTAMP,
        FeatureType.BBOX
    ]

    @classmethod
    def setUpClass(cls):
        cls.eop = EOPatch.load(cls.TEST_PATCH_FILENAME, features=cls.FEATURES_TO_LOAD)

    def test_raises_errors(self):
        test_params = [dict(dem_params=(100, 100, 100)),
                       dict(red_params=45),
                       dict(ndsi_params=(0.2, 3))]
        for test_param in test_params:
            with self.subTest(msg='Test case {}'.format(test_param.keys())):
                with self.assertRaises(ValueError):
                    theia_mask = TheiaSnowMask((FeatureType.DATA, 'BANDS-S2-L1C'),
                                               [2, 3, 11],
                                               (FeatureType.MASK, 'CLM'),
                                               (FeatureType.DATA_TIMELESS, 'DEM'),
                                               **test_param)
                    theia_mask(self.eop)

    def _check_shape(self, output, data):
        """Checks that shape of data and output match."""
        self.assertTrue(output.ndim == 4)
        self.assertTrue(output.shape[:-1] == data.shape[:-1])
        self.assertTrue(output.shape[-1] == 1)

    def test_snow_coverage(self):
        snow_mask = SnowMask((FeatureType.DATA, 'BANDS-S2-L1C'),
                             [2, 3, 7, 11],
                             mask_name='TEST_SNOW_MASK')
        theia_snow_mask = TheiaSnowMask((FeatureType.DATA, 'BANDS-S2-L1C'),
                                        [2, 3, 11],
                                        (FeatureType.MASK, 'CLM'),
                                        (FeatureType.DATA_TIMELESS, 'DEM'),
                                        b10_index=10,
                                        mask_name='TEST_THEIA_SNOW_MASK')

        test_results = [(50468, 1405), (60682, 10088)]

        for task, results in zip([snow_mask, theia_snow_mask], test_results):
            eop = task(self.eop)

            # Check shape and type
            self._check_shape(eop.mask[task.mask_feature[1]], self.eop.data['BANDS-S2-L1C'])
            self.assertTrue(eop.mask[task.mask_feature[1]].dtype == bool)

            snow_pixels = np.sum(eop.mask[task.mask_feature[1]], axis=(1, 2, 3))
            self.assertEqual(np.sum(snow_pixels), results[0],
                             'Sum of snowy pixels does not match for task {}'.format(task.mask_feature[1]))
            self.assertEqual(snow_pixels[-4], results[1],
                             'Snowy pixels on specified frame does not match for task {}'.format(task.mask_feature[1]))


if __name__ == '__main__':
    unittest.main()
