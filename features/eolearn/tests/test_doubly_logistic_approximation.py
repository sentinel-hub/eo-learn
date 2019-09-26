import unittest
import numpy as np
from eolearn.core import EOPatch, FeatureType
from features.eolearn.features import DoublyLogisticApproximationTask


class TestDoublyLogisticApproximation(unittest.TestCase):
    patch = EOPatch()

    @staticmethod
    def setUpClass(cls):
        data = np.array([0, 0.05, 0.2, 0.5, 0.81, 1, 0.95, 0.51, 0.03, 0.001, 0.001])
        for i in range(3):
            data = np.expand_dims(data, axis=1)

        cls.eopatch.add_feature(FeatureType.DATA, 'TEST', data)
        cls.eopatch = DoublyLogisticApproximationTask('TEST', 'TEST_OUT')(cls.eopatch)

    def test_parameters(self):
        c1, c2, a1, a2, a3, a4, a5 = self.eopatch.data['TEST_OUT'].squeeze()
        delta = 0.1
        print(c1)

        exp_c1 = 1
        self.assertAlmostEqual(c1, exp_c1, delta=delta)

        exp_c2 = 8.50
        self.assertAlmostEqual(c2, exp_c2, delta=delta)

        exp_a1 = 0.88
        self.assertAlmostEqual(a1, exp_a1, delta=delta)

        exp_a2 = 5.51
        self.assertAlmostEqual(a2, exp_a2, delta=delta)

        exp_a3 = 16.43
        self.assertAlmostEqual(a3, exp_a3, delta=delta)

        exp_a4 = 0.46
        self.assertAlmostEqual(a4, exp_a4, delta=delta)

        exp_a5 = 7.02
        self.assertAlmostEqual(a5, exp_a5, delta=delta)


if __name__ == '__main__':
    unittest.main()
