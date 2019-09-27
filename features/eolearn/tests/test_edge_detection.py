"""
Credits:
Copyright (c) 2018-2019 Mark Bogataj, Filip Koprivec (Jožef Stefan Institute)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import numpy as np
from eolearn.core import EOPatch, FeatureType
from eolearn.features import AdaptiveThresholdMethod, SimpleThresholdMethod, ThresholdType, \
    Thresholding, OperatorEdgeDetection, SobelOperator, ScharrOperator, ScharrFourierOperator, \
    Prewitt3Operator, Prewitt4Operator, RobertsCrossOperator, KayyaliOperator, KirschOperator


import os.path


class TestEdgeDetectionTasks(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.patch)

        Thresholding((FeatureType.DATA, "random"), [0,2,1]).execute(cls.patch)

        cls.initial_patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.initial_patch)


    @staticmethod
    def _prepare_patch(patch):
        ndvi = patch.data['ndvi'][:10]
        ndvi[np.isnan(ndvi)] = 0
        patch.data['ndvi'] = ndvi

    def test_new_features(self):
        delta = 1e-4
        random_mask = self.patch.data_timeless['random_mask']

        test_min = np.min(random_mask)
        exp_min = 0.0
        self.assertAlmostEqual(test_min, exp_min, delta=delta, msg="Expected min {}, got {}".format(exp_min, test_min))

        test_max = np.max(random_mask)
        exp_max = 255.0
        self.assertAlmostEqual(test_max, exp_max, delta=delta, msg="Expected max {}, got {}".format(exp_max, test_max))

        test_mean = np.mean(random_mask)
        exp_mean = 126.8625
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta, msg="Expected mean {}, got {}".format(exp_mean, test_mean))

        test_median = np.median(random_mask)
        exp_median = 0.0
        self.assertAlmostEqual(test_median, exp_median, delta=delta, msg="Expected median {}, got {}".format(exp_median, test_median))

        test_std = np.std(random_mask)
        exp_std = 127.4984
        self.assertAlmostEqual(test_std, exp_std, delta=delta, msg="Expected standard deviation {}, got {}".format(exp_std, test_std))

    def test_unchanged_features(self):
        for feature, value in self.initial_patch.data.items():
            self.assertTrue(np.array_equal(value, self.patch.data[feature]),
                            msg="EOPatch data feature '{}' was changed in the process".format(feature))


class TestOperatorEdgeDetection(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    class OperatorEdgeDetectionTestCase:

        def __init__(self, name, task, mag_min, mag_max, mag_mean, mag_median,
                     ang_min, ang_max, ang_mean, ang_median):
            self.name = name
            self.task = task
            self.mag_min = mag_min
            self.mag_max = mag_max
            self.mag_mean = mag_mean
            self.mag_median = mag_median
            self.ang_min = ang_min
            self.ang_max = ang_max
            self.ang_mean = ang_mean
            self.ang_median = ang_median

            self.result = None
            self.feature_type = None
            self.feature_name = None

        def execute(self, patch):
            self.feature_type, self.feature_name, _ = next(self.task.feature(patch))
            self.result = self.task.execute(patch)

    @classmethod
    def setUpClass(cls):

        cls.patch = EOPatch.load(TestOperatorEdgeDetection.TEST_PATCH_FILENAME)

        cls.test_cases = [
            cls.OperatorEdgeDetectionTestCase('Sobel operator', SobelOperator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0022429, mag_max=1.0000000, mag_mean=0.2150782, mag_median=0.0520476,
                                              ang_min=-3.1249206, ang_max=3.1332331, ang_mean=-0.0674717, ang_median=0.0191111),

            cls.OperatorEdgeDetectionTestCase('Scharr operator', ScharrOperator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0004208, mag_max=1.0000000, mag_mean=0.2017822, mag_median=0.0524957,
                                              ang_min=-3.1126597, ang_max=3.1366969, ang_mean=-0.0881438, ang_median=0.0167943),

            cls.OperatorEdgeDetectionTestCase('Scharr Fourier operator', ScharrFourierOperator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0007999, mag_max=1.0000000, mag_mean=0.2010398, mag_median=0.0524722,
                                              ang_min=-3.1252567, ang_max=3.1406779, ang_mean=-0.1018352, ang_median=0.0186267),

            cls.OperatorEdgeDetectionTestCase('Prewit 3x3 operator', Prewitt3Operator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0008755, mag_max=1.0000000, mag_mean=0.2145458, mag_median=0.0490427,
                                              ang_min=-3.1394975, ang_max=3.1404768, ang_mean=-0.0747916, ang_median=-0.0006886),

            cls.OperatorEdgeDetectionTestCase('Roberts 4x4 operator', Prewitt4Operator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0013428, mag_max=1.0000000, mag_mean=0.2493523, mag_median=0.0406065,
                                              ang_min=-3.1389329, ang_max=3.1395759, ang_mean=-0.1907657, ang_median=-0.0238701),

            cls.OperatorEdgeDetectionTestCase('Roberts cross operator', RobertsCrossOperator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0012565, mag_max=1.0000000, mag_mean=0.1344944, mag_median=0.0501696,
                                              ang_min=-3.1333714, ang_max=3.1415927, ang_mean=-0.0760171, ang_median=-0.1745898),

            cls.OperatorEdgeDetectionTestCase('Kayyali operator', KayyaliOperator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0000218, mag_max=1.0000000, mag_mean=0.0774692, mag_median=0.0580744,
                                              ang_min=-0.7853982, ang_max=2.3561945, ang_mean=0.8089601, ang_median=2.3561945),

            cls.OperatorEdgeDetectionTestCase('Kirsch operator', KirschOperator((FeatureType.DATA, "ndvi"), index=4),
                                              mag_min=0.0043820, mag_max=1.0000000, mag_mean=0.1774682, mag_median=0.0440887,
                                              ang_min=-3.0933584, ang_max=3.1145084, ang_mean=-0.1142567, ang_median=-0.0883852),

            cls.OperatorEdgeDetectionTestCase('Sobel operator bare bands',
                                              SobelOperator((FeatureType.DATA,"random"), index=4, sub_index=3),
                                              mag_min=0.0081409, mag_max=1.0000000, mag_mean=0.3824989, mag_median=0.3549343,
                                              ang_min=-3.1065221, ang_max=3.1050154, ang_mean=-0.0777507, ang_median=-0.0349679),

            cls.OperatorEdgeDetectionTestCase('Sobel operator grayscale',
                                              SobelOperator((FeatureType.DATA,"random"), index=4, sub_index=[4, 3, 2],
                                                            to_grayscale=True),
                                              mag_min=0.0058795, mag_max=1.0000000, mag_mean=0.3131372, mag_median=0.2580054,
                                              ang_min=-3.1371329, ang_max=3.1316391, ang_mean=-0.1175701, ang_median=-0.1166535),
        ]

    def test_stats(self):
        for test_case in self.test_cases:
            delta = 1e-5
            test_case.execute(self.patch)
            data = test_case.result.data_timeless[test_case.feature_name]

            magnitude = data[..., 0]
            angle = data[..., 1]

            mag_min_val = np.amin(magnitude)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mag_min, mag_min_val, delta=delta,
                                       msg="Expected magnitude min {}, got {}".format(test_case.mag_min, mag_min_val))

            mag_max_val = np.amax(magnitude)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mag_max, mag_max_val, delta=delta,
                                       msg="Expected magnitude max {}, got {}".format(test_case.mag_max, mag_max_val))

            mag_mean_val = np.mean(magnitude)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mag_mean, mag_mean_val, delta=delta,
                                       msg="Expected magnitude mean {}, got {}".format(test_case.mag_mean, mag_mean_val))

            mag_median_val = np.median(magnitude)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.mag_min, mag_min_val, delta=delta,
                                       msg="Expected magnitude median {}, got {}".format(test_case.mag_median, mag_median_val))

            ang_min_val = np.amin(angle)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.ang_min, ang_min_val, delta=delta,
                                       msg="Expected angle min {}, got {}".format(test_case.ang_min, ang_min_val))

            ang_max_val = np.amax(angle)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.ang_max, ang_max_val, delta=delta,
                                       msg="Expected angle max {}, got {}".format(test_case.ang_max, ang_max_val))

            ang_mean_val = np.mean(angle)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.ang_mean, ang_mean_val, delta=delta,
                                       msg="Expected angle mean {}, got {}".format(test_case.ang_mean, ang_mean_val))

            ang_median_val = np.median(angle)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.ang_min, ang_min_val, delta=delta,
                                       msg="Expected angle median {}, got {}".format(test_case.ang_median, ang_median_val))

if __name__ == '__main__':
    unittest.main()
