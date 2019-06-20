import unittest
import os

import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import SuperpixelSegmentation, FelzenszwalbSegmentation, SlicSegmentation


class TestSuperpixelSegmentation(unittest.TestCase):

    class TestCase:
        """
        Container for each test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                           'TestEOPatch')
        SUPERPIXEL_FEATURE = FeatureType.MASK_TIMELESS, 'SP_FEATURE'

        def __init__(self, name, task, mask_min=0, mask_max=0, mask_mean=0, mask_median=0):
            self.name = name
            self.task = task
            self.mask_min = mask_min
            self.mask_max = mask_max
            self.mask_mean = mask_mean
            self.mask_median = mask_median

            self.result = None

        def execute(self):
            eopatch = EOPatch.load(self.TEST_PATCH_FILENAME)

            eopatch = self.task.execute(eopatch)

            self.result = eopatch[self.SUPERPIXEL_FEATURE[0]][self.SUPERPIXEL_FEATURE[1]]

    @classmethod
    def setUpClass(cls):
        # Segmentation also works with nan values, although it is not tested
        cls.test_cases = [
            cls.TestCase('base superpixel segmentation',
                         SuperpixelSegmentation((FeatureType.DATA, 'BANDS-S2-L1C'), cls.TestCase.SUPERPIXEL_FEATURE,
                                                scale=100, sigma=0.5, min_size=100),
                         mask_min=0, mask_max=25, mask_mean=10.6809, mask_median=11),
            cls.TestCase('Felzenszwalb segmentation',
                         FelzenszwalbSegmentation((FeatureType.DATA_TIMELESS, 'MAX_NDVI'),
                                                  cls.TestCase.SUPERPIXEL_FEATURE, scale=21, sigma=1.0, min_size=52),
                         mask_min=0, mask_max=22, mask_mean=8.5302, mask_median=7),
            cls.TestCase('Felzenszwalb segmentation on mask',
                         FelzenszwalbSegmentation((FeatureType.MASK, 'CLM'), cls.TestCase.SUPERPIXEL_FEATURE,
                                                  scale=1, sigma=0, min_size=15),
                         mask_min=0, mask_max=171, mask_mean=86.46267, mask_median=90),
            cls.TestCase('SLIC segmentation',
                         SlicSegmentation((FeatureType.DATA, 'CLP'), cls.TestCase.SUPERPIXEL_FEATURE,
                                          n_segments=55, compactness=25.0, max_iter=20, sigma=0.8),
                         mask_min=0, mask_max=48, mask_mean=24.6072, mask_median=25),
            cls.TestCase('SLIC segmentation on mask',
                         SlicSegmentation((FeatureType.MASK_TIMELESS, 'RANDOM_UINT8'), cls.TestCase.SUPERPIXEL_FEATURE,
                                          n_segments=231, compactness=15.0, max_iter=7, sigma=0.2),
                         mask_min=0, mask_max=195, mask_mean=100.1844, mask_median=101),
        ]

        for test_case in cls.test_cases:
            test_case.execute()

    def test_output_type(self):
        for test_case in self.test_cases:
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertTrue(test_case.result.dtype == np.int64, "Expected int64 dtype")

    def test_stats(self):
        for test_case in self.test_cases:
            delta = 1e-3
            data = test_case.result

            min_val = np.amin(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mask_min, min_val, delta=delta,
                                       msg="Expected min {}, got {}".format(test_case.mask_min, min_val))

            max_val = np.amax(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mask_max, max_val, delta=delta,
                                       msg="Expected max {}, got {}".format(test_case.mask_max, max_val))

            mean_val = np.mean(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mask_mean, mean_val, delta=delta,
                                       msg="Expected mean {}, got {}".format(test_case.mask_mean, mean_val))

            median_val = np.median(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mask_median, median_val, delta=delta,
                                       msg="Expected median {}, got {}".format(test_case.mask_median, median_val))


if __name__ == '__main__':
    unittest.main()
