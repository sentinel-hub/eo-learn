import unittest
import os.path
import numpy as np
from eolearn.core import EOPatch, FeatureType
from eolearn.features.doubly_logistic_approximation import DoublyLogisticApproximationTask


class TestDoublyLogisticApproximation(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        data = cls.patch.data['NDVI']
        timestamps = cls.patch.timestamp
        mask = cls.patch.mask['IS_VALID']
        indices = list(np.nonzero([t.year == 2016 for t in timestamps])[0])
        cls.patch = EOPatch()
        cls.patch.timestamp = timestamps[indices[0]:(indices[-1] + 2)]
        cls.patch.data['TEST'] = np.reshape(data[indices[0]:(indices[-1] + 2), 0, 0, 0], (-1, 1, 1, 1))
        cls.patch.mask['IS_VALID'] = np.reshape(mask[indices[0]:(indices[-1] + 2), 0, 0, 0], (-1, 1, 1, 1))
        cls.patch = DoublyLogisticApproximationTask(feature='TEST', valid_mask=(FeatureType.MASK, 'IS_VALID'),
                                                    new_feature=f'TEST_OUT').execute(cls.patch)

    def test_parameters(self):
        c1, c2, a1, a2, a3, a4, a5 = self.patch.data_timeless['TEST_OUT'].squeeze()
        delta = 0.1

        exp_c1 = 0.207
        self.assertAlmostEqual(c1, exp_c1, delta=delta)

        exp_c2 = 0.464
        self.assertAlmostEqual(c2, exp_c2, delta=delta)

        exp_a1 = 0.686
        self.assertAlmostEqual(a1, exp_a1, delta=delta)

        exp_a2 = 0.222
        self.assertAlmostEqual(a2, exp_a2, delta=delta)

        exp_a3 = 1.204
        self.assertAlmostEqual(a3, exp_a3, delta=delta)

        exp_a4 = 0.406
        self.assertAlmostEqual(a4, exp_a4, delta=delta)

        exp_a5 = 15.701
        self.assertAlmostEqual(a5, exp_a5, delta=delta)


if __name__ == '__main__':
    unittest.main()
