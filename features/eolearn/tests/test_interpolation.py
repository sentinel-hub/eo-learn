
import unittest
import os.path
import numpy as np
from datetime import datetime

from eolearn.core import EOPatch
from eolearn.features import LinearInterpolation, CubicInterpolation, SplineInterpolation, BSplineInterpolation, \
    AkimaInterpolation, LinearResampling, CubicResampling, NearestResampling


class TestInterpolation(unittest.TestCase):

    class InterpolationTestCase:
        """
        Container for each interpolation test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

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
            if self.nan_replace is not None:
                patch.data[self.task.feature_name][np.isnan(patch.data[self.task.feature_name])] = self.nan_replace
            self.result = self.task.execute(patch)

    @classmethod
    def setUpClass(cls):

        cls.test_cases = [
            cls.InterpolationTestCase('linear', LinearInterpolation('ndvi', result_interval=(-0.3, -0.1),
                                                                    unknown_value=10),
                                      result_len=180, img_min=-0.3000, img_max=10.0, img_mean=-0.147169,
                                      img_median=-0.21132),
            cls.InterpolationTestCase('cubic', CubicInterpolation('ndvi', result_interval=(-0.3, -0.1),
                                                                  resample_range=('2015-01-01', '2018-01-01', 16),
                                                                  unknown_value=5),
                                      result_len=69, img_min=-0.3000, img_max=5.0, img_mean=1.30387,
                                      img_median=-0.1007357),
            cls.InterpolationTestCase('spline', SplineInterpolation('ndvi', result_interval=(-10, 10),
                                                                    resample_range=('2016-01-01', '2018-01-01', 5),
                                                                    spline_degree=3, smoothing_factor=0.5),
                                      result_len=147, img_min=-0.6867819, img_max=0.26210105, img_mean=-0.1886001,
                                      img_median=-0.18121514),
            cls.InterpolationTestCase('bspline', BSplineInterpolation('ndvi', unknown_value=-3,
                                                                      resample_range=('2017-01-01', '2017-02-01', 50),
                                                                      spline_degree=5),
                                      result_len=1, img_min=-0.20366311, img_max=0.03428878, img_mean=-0.04307341,
                                      img_median=-0.040708482),
            cls.InterpolationTestCase('akima', AkimaInterpolation('ndvi', unknown_value=0),
                                      result_len=180, img_min=-0.4821199, img_max=0.2299331, img_mean=-0.20141865,
                                      img_median=-0.213559),
            cls.InterpolationTestCase('nearest resample', NearestResampling('ndvi', result_interval=(-0.3, -0.1),
                                                                            resample_range=('2016-01-01',
                                                                                            '2018-01-01', 5)),
                                      result_len=147, img_min=-0.3000, img_max=-0.1000, img_mean=-0.19651729,
                                      img_median=-0.20000, nan_replace=-0.2),
            cls.InterpolationTestCase('linear resample', LinearResampling('ndvi', result_interval=(-0.3, -0.1),
                                                                          resample_range=('2016-01-01',
                                                                                          '2018-01-01', 5)),
                                      result_len=147, img_min=-0.3000, img_max=-0.1000, img_mean=-0.1946315,
                                      img_median=-0.20000, nan_replace=-0.2),
            cls.InterpolationTestCase('cubic resample', CubicResampling('ndvi', result_interval=(-0.3, -0.1),
                                                                        resample_range=('2015-01-01', '2018-01-01', 16),
                                                                        unknown_value=5),
                                      result_len=69, img_min=-0.3000, img_max=5.0, img_mean=1.22359,
                                      img_median=-0.181167, nan_replace=-0.2)
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
                data_shape = test_case.result.data['ndvi'].shape
                exp_shape = (test_case.result_len, 20, 20, 1)
                self.assertEqual(data_shape, exp_shape,
                                 "Expected data shape {}, got {}".format(exp_shape, data_shape))

    def test_stats(self):
        for test_case in self.test_cases:
            delta = 1e-3

            if test_case.img_min is not None:
                min_val = np.amin(test_case.result.data['ndvi'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_min, min_val, delta=delta,
                                               msg="Expected min {}, got {}".format(test_case.img_min, min_val))
            if test_case.img_max is not None:
                max_val = np.amax(test_case.result.data['ndvi'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_max, max_val, delta=delta,
                                               msg="Expected max {}, got {}".format(test_case.img_max, max_val))
            if test_case.img_mean is not None:
                mean_val = np.mean(test_case.result.data['ndvi'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_mean, mean_val, delta=delta,
                                               msg="Expected mean {}, got {}".format(test_case.img_mean, mean_val))
            if test_case.img_median is not None:
                median_val = np.median(test_case.result.data['ndvi'])
                with self.subTest(msg='Test case {}'.format(test_case.name)):
                        self.assertAlmostEqual(test_case.img_median, median_val, delta=delta,
                                               msg="Expected median {}, got {}".format(test_case.img_median,
                                                                                       median_val))


if __name__ == '__main__':
    unittest.main()
