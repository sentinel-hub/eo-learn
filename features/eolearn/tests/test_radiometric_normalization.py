import unittest
import os.path
import numpy as np
from datetime import datetime

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.mask import MaskFeature
from eolearn.features import ReferenceScenes, BlueCompositing, HOTCompositing, MaxNDVICompositing, \
    MaxNDWICompositing, MaxRatioCompositing, HistogramMatching


class TestRadiometricNormalization(unittest.TestCase):

    class RadiometricNormalizationTestCase:
        """
        Container for each interpolation test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                           'TestEOPatch')

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
                (FeatureType.DATA, 'BANDS-S2-L1C', 'TEST'), (FeatureType.MASK, 'SCL'),
                mask_values=[0, 1, 2, 3, 8, 9, 10, 11]), img_min=0.0004, img_max=1.0678, img_mean=0.12345793,
                img_median=0.1028),

            cls.RadiometricNormalizationTestCase('reference scene', ReferenceScenes(
                (FeatureType.DATA, 'MASKED', 'TEST'), (FeatureType.SCALAR, 'VALID_FRAC'),
                max_scene_number=5), img_min=0.0006, img_max=0.982, img_mean=0.11606364,
                img_median=0.1053),

            cls.RadiometricNormalizationTestCase('blue compositing', BlueCompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                blue_idx=0, interpolation='geoville'), img_min=0.0006, img_max=0.6152, img_mean=0.09875322,
                img_median=0.095),

            cls.RadiometricNormalizationTestCase('hot compositing', HOTCompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                blue_idx=0, red_idx=2, interpolation='geoville'), img_min=0.0006, img_max=0.8131, img_mean=0.09900281,
                img_median=0.0952),

            cls.RadiometricNormalizationTestCase('max ndvi compositing', MaxNDVICompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                red_idx=2, nir_idx=7, interpolation='geoville'), img_min=0.0006, img_max=0.6152,
                img_mean=0.10077792, img_median=0.0966),

            cls.RadiometricNormalizationTestCase('max ndwi compositing', MaxNDWICompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                nir_idx=6, swir1_idx=8, interpolation='geoville'), img_min=0.0006, img_max=0.982,
                img_mean=0.12849501, img_median=0.1096),

            cls.RadiometricNormalizationTestCase('max ratio compositing', MaxRatioCompositing(
                (FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA_TIMELESS, 'TEST'),
                blue_idx=0, nir_idx=6, swir1_idx=8, interpolation='geoville'), img_min=0.0006, img_max=0.9385,
                img_mean=0.111010134, img_median=0.1045),

            cls.RadiometricNormalizationTestCase('histogram matching', HistogramMatching(
                (FeatureType.DATA, 'MASKED', 'TEST'),
                (FeatureType.DATA_TIMELESS, 'REFERENCE_COMPOSITE_BLUE')),
                img_min=-0.13022107, img_max=0.5565749, img_mean=0.09766118, img_median=0.09366062)
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
