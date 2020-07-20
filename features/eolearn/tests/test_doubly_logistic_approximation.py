import unittest
import os.path
import numpy as np
from eolearn.core import EOPatch, FeatureType
from eolearn.features.doubly_logistic_approximation import DoublyLogisticApproximationTask


class TestDoublyLogisticApproximation(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.patch)
        cls.patch = DoublyLogisticApproximationTask('ndvi',
                                                    'TEST_OUT',
                                                    mask_feature=(FeatureType.MASK, 'IS_VALID')
                                                    ).execute(cls.patch)

    @staticmethod
    def _prepare_patch(patch):
        patch.add_feature(FeatureType.DATA, 'ndvi', patch.data['ndvi'][:, 0, 0, 0].reshape(-1, 1, 1, 1))
        patch.add_feature(FeatureType.MASK, 'IS_VALID',
                          np.invert(patch.mask['IS_VALID'][:, 0, 0, 0].reshape(-1, 1, 1, 1)))

    def test_parameters(self):
        c1, c2, a1, a2, a3, a4, a5 = self.patch.data_timeless['TEST_OUT'].squeeze()
        delta = 0.1

        exp_c1 = -0.21
        self.assertAlmostEqual(c1, exp_c1, delta=delta)

        exp_c2 = 0.09
        self.assertAlmostEqual(c2, exp_c2, delta=delta)

        exp_a1 = 0.45
        self.assertAlmostEqual(a1, exp_a1, delta=delta)

        exp_a2 = 0.15
        self.assertAlmostEqual(a2, exp_a2, delta=delta)

        exp_a3 = 4.77
        self.assertAlmostEqual(a3, exp_a3, delta=delta)

        exp_a4 = 0.15
        self.assertAlmostEqual(a4, exp_a4, delta=delta)

        exp_a5 = 26.17
        self.assertAlmostEqual(a5, exp_a5, delta=delta)


if __name__ == '__main__':
    unittest.main()
