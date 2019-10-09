import unittest

import numpy as np
from eolearn.core import EOPatch
from eolearn.features.doubly_logistic_approximation import DoublyLogisticApproximationTask
from datetime import datetime


class TestDoublyLogisticApproximation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dates = [datetime(2017, 1, 1), datetime(2017, 1, 11), datetime(2017, 2, 7), datetime(2017, 2, 17),
                 datetime(2017, 2, 20), datetime(2017, 3, 9), datetime(2017, 3, 12), datetime(2017, 3, 22),
                 datetime(2017, 3, 29), datetime(2017, 4, 1), datetime(2017, 4, 11), datetime(2017, 4, 21),
                 datetime(2017, 5, 1), datetime(2017, 5, 8), datetime(2017, 5, 11), datetime(2017, 5, 18),
                 datetime(2017, 5, 21), datetime(2017, 5, 28), datetime(2017, 5, 31), datetime(2017, 6, 10),
                 datetime(2017, 6, 17), datetime(2017, 6, 20), datetime(2017, 6, 27), datetime(2017, 6, 30),
                 datetime(2017, 7, 2), datetime(2017, 7, 7), datetime(2017, 7, 10), datetime(2017, 7, 12),
                 datetime(2017, 7, 15), datetime(2017, 7, 17), datetime(2017, 7, 20), datetime(2017, 7, 22),
                 datetime(2017, 7, 25), datetime(2017, 7, 30), datetime(2017, 8, 1), datetime(2017, 8, 4),
                 datetime(2017, 8, 6), datetime(2017, 8, 9), datetime(2017, 8, 11), datetime(2017, 8, 21),
                 datetime(2017, 8, 24), datetime(2017, 8, 26), datetime(2017, 8, 29), datetime(2017, 8, 31),
                 datetime(2017, 9, 5), datetime(2017, 9, 18), datetime(2017, 9, 23), datetime(2017, 9, 28),
                 datetime(2017, 9, 30), datetime(2017, 10, 5), datetime(2017, 10, 8), datetime(2017, 10, 10),
                 datetime(2017, 10, 13), datetime(2017, 10, 15), datetime(2017, 10, 18), datetime(2017, 10, 20),
                 datetime(2017, 10, 25), datetime(2017, 10, 30), datetime(2017, 11, 4), datetime(2017, 11, 19),
                 datetime(2017, 11, 24), datetime(2017, 11, 27), datetime(2017, 12, 2), datetime(2017, 12, 7),
                 datetime(2017, 12, 17), datetime(2017, 12, 19), datetime(2017, 12, 22), datetime(2017, 12, 24),
                 datetime(2017, 12, 29)]

        values = [0.059783712, 0.06304545, np.NaN, 0.07635699, np.NaN, 0.07554149, np.NaN, np.NaN, 0.095581025,
                  0.09109493, np.NaN, 0.114999376, np.NaN, np.NaN, np.NaN, 0.12209105, 0.089099824, 0.12314385,
                  np.NaN, np.NaN, np.NaN, 0.13705324, np.NaN, np.NaN, np.NaN, 0.26388252, 0.19771394, 0.27082473,
                  np.NaN, 0.3250511, 0.24847558, 0.35576594, 0.3713841, 0.30338106, 0.4179747, 0.326186, np.NaN,
                  np.NaN, np.NaN, 0.69031024, 0.58324015, 0.68232125, 0.59130245, 0.655362, 0.7490687, np.NaN,
                  0.45670646, 0.39466715, 0.43877497, 0.40336365, 0.2609013, np.NaN, 0.13757367, 0.12950855,
                  0.13163383, 0.13427204, np.NaN, 0.117703386, 0.12337628, 0.013523403, 0.076479584, 0.058155004,
                  np.NaN, 0.04863815, 0.060789138, 0.063743524, np.NaN, 0.044066854, 0.11930587]

        cls.eopatch = EOPatch(timestamp=list(dates))
        cls.eopatch.timestamp = list(dates)
        cls.eopatch.data["TEST"] = np.array(values).reshape(-1, 1, 1, 1)
        cls.eopatch.mask["VALID_DATA"] = ~np.isnan(cls.eopatch.data["TEST"])
        cls.eopatch = DoublyLogisticApproximationTask('TEST', 'TEST_OUT', mask_data=True).execute(cls.eopatch)

    def test_parameters(self):
        c1, c2, a1, a2, a3, a4, a5 = self.eopatch.data_timeless['TEST_OUT'].squeeze()
        delta = 0.1

        exp_c1 = 0.03
        self.assertAlmostEqual(c1, exp_c1, delta=delta)

        exp_c2 = 0.56
        self.assertAlmostEqual(c2, exp_c2, delta=delta)

        exp_a1 = 0.64
        self.assertAlmostEqual(a1, exp_a1, delta=delta)

        exp_a2 = 0.15
        self.assertAlmostEqual(a2, exp_a2, delta=delta)

        exp_a3 = 3.07
        self.assertAlmostEqual(a3, exp_a3, delta=delta)

        exp_a4 = 0.15
        self.assertAlmostEqual(a4, exp_a4, delta=delta)

        exp_a5 = 1
        self.assertAlmostEqual(a5, exp_a5, delta=delta)


if __name__ == '__main__':
    unittest.main()
