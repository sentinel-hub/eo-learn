
import unittest
import os.path
import numpy as np
from datetime import datetime

from eolearn.core import EOPatch, FeatureType
from eolearn.features import LinearInterpolation, CubicInterpolation, SplineInterpolation, BSplineInterpolation, \
    AkimaInterpolation, LinearResampling, CubicResampling, NearestResampling


class TestInterpolation(unittest.TestCase):

    class InterpolationTestCase:
        """
        Container for each interpolation test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                           'TestEOPatch')

        def __init__(self, name, task, result_len, img_min=None, img_max=None, img_mean=None, img_median=None,
                     nan_replace=None):
            self.name = name
            self.task = task
            self.result_len = result_len
            self.img_min = img_min
            self.img_max = img_max
            self.img_mean = img_mean
            self.img_median = img_median
            self.nan_replace = nan_replace

            self.result = None

        def execute(self):
            patch = EOPatch.load(self.TEST_PATCH_FILENAME)
            self.result = self.task.execute(patch)
            patch = self.result

            if self.nan_replace is not None:
                feature_type, feature_name, _ = next(self.task.feature(patch))
                patch[feature_type][feature_name][np.isnan(patch[feature_type][feature_name])] = self.nan_replace

    @classmethod
    def setUpClass(cls):

        cls.test_cases = [
            cls.InterpolationTestCase('linear', LinearInterpolation('NDVI', result_interval=(0.0, 1.0),
                                                                    mask_feature=(FeatureType.MASK, 'IS_VALID'),
                                                                    unknown_value=10),
                                      result_len=68, img_min=0.0, img_max=10.0, img_mean=0.720405,
                                      img_median=0.59765935),

            cls.InterpolationTestCase('linear_change_timescale',
                                      LinearInterpolation('NDVI', result_interval=(0.0, 1.0),
                                                          mask_feature=(FeatureType.MASK, 'IS_VALID'),
                                                          unknown_value=10, scale_time=1),
                                      result_len=68, img_min=0.0, img_max=10.0, img_mean=0.720405,
                                      img_median=0.59765935),

            cls.InterpolationTestCase('cubic', CubicInterpolation('NDVI', result_interval=(0.0, 1.0),
                                                                  mask_feature=(FeatureType.MASK, 'IS_VALID'),
                                                                  resample_range=('2015-01-01', '2018-01-01', 16),
                                                                  unknown_value=5, bounds_error=False),
                                      result_len=69, img_min=0.0, img_max=5.0, img_mean=1.3592644,
                                      img_median=0.6174331),

            cls.InterpolationTestCase('spline', SplineInterpolation('NDVI', result_interval=(-0.3, 1.0),
                                                                    mask_feature=(FeatureType.MASK, 'IS_VALID'),
                                                                    resample_range=('2016-01-01', '2018-01-01', 5),
                                                                    spline_degree=3, smoothing_factor=0,
                                                                    unknown_value=0),
                                      result_len=147, img_min=-0.3, img_max=1.0, img_mean=0.492752,
                                      img_median=0.53776133),

            cls.InterpolationTestCase('bspline', BSplineInterpolation('NDVI', unknown_value=-3,
                                                                      mask_feature=(FeatureType.MASK, 'IS_VALID'),
                                                                      resample_range=('2017-01-01', '2017-02-01', 50),
                                                                      spline_degree=5),
                                      result_len=1, img_min=-0.032482587, img_max=0.701796, img_mean=0.42080238,
                                      img_median=0.42889267),

            cls.InterpolationTestCase('akima', AkimaInterpolation('NDVI', unknown_value=0,
                                                                  mask_feature=(FeatureType.MASK, 'IS_VALID')),
                                      result_len=68, img_min=-0.13793105, img_max=0.860242, img_mean=0.53159297,
                                      img_median=0.59087014),

            cls.InterpolationTestCase('nearest resample', NearestResampling('NDVI', result_interval=(0.0, 1.0),
                                                                            resample_range=('2016-01-01',
                                                                                            '2018-01-01', 5)),
                                      result_len=147, img_min=-0.2, img_max=0.860242, img_mean=0.35143828,
                                      img_median=0.37481314, nan_replace=-0.2),

            cls.InterpolationTestCase('linear resample', LinearResampling('NDVI', result_interval=(0.0, 1.0),
                                                                          resample_range=('2016-01-01',
                                                                                          '2018-01-01', 5)),
                                      result_len=147, img_min=-0.2, img_max=0.8480114, img_mean=0.350186,
                                      img_median=0.3393997, nan_replace=-0.2),

            cls.InterpolationTestCase('cubic resample', CubicResampling('NDVI', result_interval=(-0.2, 1.0),
                                                                        resample_range=('2015-01-01', '2018-01-01', 16),
                                                                        unknown_value=5),
                                      result_len=69, img_min=-0.2, img_max=5.0, img_mean=1.2348738,
                                      img_median=0.4656453, nan_replace=-0.2),

            cls.InterpolationTestCase('linear custom list', LinearInterpolation(
                'NDVI', result_interval=(-0.2, 1.0),
                resample_range=('2015-09-01', '2016-01-01', '2016-07-01', '2017-01-01', '2017-07-01'),
                unknown_value=-2),
                                      result_len=5, img_min=-0.032482587, img_max=0.8427637, img_mean=0.5108417,
                                      img_median=0.5042224)

        ]

        for test_case in cls.test_cases:
            test_case.execute()

    def test_output_type(self):
        for test_case in self.test_cases:
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertTrue(isinstance(test_case.result.timestamp, list), "Expected a list of timestamps")
                self.assertTrue(isinstance(test_case.result.timestamp[0], datetime),
                                "Expected timestamps of type datetime.datetime")
                ts_len = len(test_case.result.timestamp)
                self.assertEqual(ts_len, test_case.result_len,
                                 msg="Expected timestamp len {}, got {}".format(test_case.result_len, ts_len))
                data_shape = test_case.result.data['NDVI'].shape
                exp_shape = (test_case.result_len, 101, 100, 1)
                self.assertEqual(data_shape, exp_shape,
                                 "Expected data shape {}, got {}".format(exp_shape, data_shape))

    def test_stats(self):
        for test_case in self.test_cases:
            delta = 1e-3

            if test_case.img_min is not None:
                min_val = np.amin(test_case.result.data['NDVI'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                    self.assertAlmostEqual(test_case.img_min, min_val, delta=delta,
                                           msg="Expected min {}, got {}".format(test_case.img_min, min_val))
            if test_case.img_max is not None:
                max_val = np.amax(test_case.result.data['NDVI'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                    self.assertAlmostEqual(test_case.img_max, max_val, delta=delta,
                                           msg="Expected max {}, got {}".format(test_case.img_max, max_val))
            if test_case.img_mean is not None:
                mean_val = np.mean(test_case.result.data['NDVI'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                    self.assertAlmostEqual(test_case.img_mean, mean_val, delta=delta,
                                           msg="Expected mean {}, got {}".format(test_case.img_mean, mean_val))
            if test_case.img_median is not None:
                median_val = np.median(test_case.result.data['NDVI'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                    self.assertAlmostEqual(test_case.img_median, median_val, delta=delta,
                                           msg="Expected median {}, got {}".format(test_case.img_median,
                                                                                   median_val))


if __name__ == '__main__':
    unittest.main()
