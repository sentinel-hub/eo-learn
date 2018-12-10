import unittest
import os.path
import numpy as np
from datetime import datetime

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.mask import MaskFeature
from eolearn.features import ReferenceScenes, BaseCompositing, BlueCompositing, HOTCompositing, MaxNDVICompositing, \
    MaxNDWICompositing, MaxRatioCompositing, HistogramMatching


class TestRadiometricNormalization(unittest.TestCase):

    class RadiometricNormalizationTestCase:
        """
        Container for each interpolation test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                           'radiometricNormalization_TestEOPatch')

        def __init__(self, name, task, img_min=None, img_max=None, img_mean=None, img_median=None):
            self.name = name
            self.task = task
            self.img_min = img_min
            self.img_max = img_max
            self.img_mean = img_mean
            self.img_median = img_median
            self.result = None

        def execute(self):
            patch = EOPatch.load(self.TEST_PATCH_FILENAME)
            self.result = self.task.execute(patch)

    @classmethod
    def setUpClass(cls):

        cls.test_cases = [
            cls.RadiometricNormalizationTestCase('mask feature', MaskFeature(
                (FeatureType.DATA, 'S2-L1C-10-BANDS', 'TEST'), (FeatureType.MASK, 'SCL'),
                mask_values=[0, 1, 2, 3, 8, 9, 10, 11]), img_min=0.0084, img_max=1.0503, img_mean=0.19780958,
                img_median=0.1597),

            cls.RadiometricNormalizationTestCase('reference scene', ReferenceScenes(
                (FeatureType.DATA, 'MASKED', 'TEST'), (FeatureType.SCALAR, 'VALID_FRAC'),
                max_scene_number=5), img_min=0.0084, img_max=1.0503, img_mean=0.19995609,
                img_median=0.161),

            cls.RadiometricNormalizationTestCase('blue compositing', BlueCompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                blue_idx=0, interpolation='geoville'), img_min=0.0086, img_max=1.0503, img_mean=0.18888658,
                img_median=0.1475),

            cls.RadiometricNormalizationTestCase('hot compositing', HOTCompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                blue_idx=0, red_idx=2, interpolation='geoville'), img_min=0.0084, img_max=0.8064, img_mean=0.18883,
                img_median=0.147),

            cls.RadiometricNormalizationTestCase('max ndvi compositing', MaxNDVICompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                red_idx=2, nir_idx=7, interpolation='geoville'), img_min=0.0091, img_max=0.9463,
                img_mean=0.20268114, img_median=0.1602),

            cls.RadiometricNormalizationTestCase('max ndwi compositing', MaxNDWICompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                nir_idx=6, swir1_idx=8, interpolation='geoville'), img_min=0.0084, img_max=0.8064,
                img_mean=0.19160628, img_median=0.1465),

            cls.RadiometricNormalizationTestCase('max ratio compositing', MaxRatioCompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                blue_idx=0, nir_idx=6, swir1_idx=8, interpolation='geoville'), img_min=0.0093, img_max=0.9463,
                img_mean=0.20803407, img_median=0.1654),

            cls.RadiometricNormalizationTestCase('histogram matching', HistogramMatching(
                (FeatureType.DATA, 'MASKED', 'TEST'),
                (FeatureType.DATA_TIMELESS, 'REFERENCE_COMPOSITE_BLUE')),
                img_min=-0.1530194, img_max=1.1012005, img_mean=0.1901219, img_median=0.14299975)
        ]

        for test_case in cls.test_cases:
            test_case.execute()

    def test_output_type(self):
        for test_case in self.test_cases:
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertTrue(isinstance(test_case.result.timestamp, list), "Expected a list of timestamps")
                self.assertTrue(isinstance(test_case.result.timestamp[0], datetime),
                                "Expected timestamps of type datetime.datetime")

    def test_stats(self):
        for tci, test_case in enumerate(self.test_cases):
            delta = 1e-3

            if tci not in [2, 3, 4, 5, 6]:
                if test_case.img_min is not None:
                    min_val = np.nanmin(test_case.result.data['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_min, min_val, delta=delta,
                                               msg="Expected min {}, got {}".format(test_case.img_min, min_val))
                if test_case.img_max is not None:
                    max_val = np.nanmax(test_case.result.data['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_max, max_val, delta=delta,
                                               msg="Expected max {}, got {}".format(test_case.img_max, max_val))
                if test_case.img_mean is not None:
                    mean_val = np.nanmean(test_case.result.data['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_mean, mean_val, delta=delta,
                                               msg="Expected mean {}, got {}".format(test_case.img_mean, mean_val))
                if test_case.img_median is not None:
                    median_val = np.nanmedian(test_case.result.data['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_median, median_val, delta=delta,
                                               msg="Expected median {}, got {}".format(test_case.img_median,
                                                                                       median_val))

            else:
                if test_case.img_min is not None:
                    min_val = np.nanmin(test_case.result.data_timeless['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_min, min_val, delta=delta,
                                               msg="Expected min {}, got {}".format(test_case.img_min, min_val))
                if test_case.img_max is not None:
                    max_val = np.nanmax(test_case.result.data_timeless['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_max, max_val, delta=delta,
                                               msg="Expected max {}, got {}".format(test_case.img_max, max_val))
                if test_case.img_mean is not None:
                    mean_val = np.nanmean(test_case.result.data_timeless['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_mean, mean_val, delta=delta,
                                               msg="Expected mean {}, got {}".format(test_case.img_mean, mean_val))
                if test_case.img_median is not None:
                    median_val = np.nanmedian(test_case.result.data_timeless['TEST'])
                    with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_median, median_val, delta=delta,
                                               msg="Expected median {}, got {}".format(test_case.img_median,
                                                                                       median_val))


if __name__ == '__main__':
    unittest.main()
