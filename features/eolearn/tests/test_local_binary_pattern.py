"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2019 Matej Aleksandrov, Devis Peresutti (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import os.path
import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.features import LocalBinaryPatternTask


class TestLocalBinaryPattern(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.patch)

        LocalBinaryPatternTask((FeatureType.DATA, 'ndvi', 'lbp'), nb_points=24, radius=3).execute(cls.patch)

        cls.initial_patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.initial_patch)

    @staticmethod
    def _prepare_patch(patch):
        ndvi = patch.data['ndvi'][:10]
        ndvi[np.isnan(ndvi)] = 0
        patch.data['ndvi'] = ndvi

    def test_new_feature(self):
        lbp = self.patch.data['lbp']
        delta = 1e-4

        test_min = np.min(lbp)
        exp_min = 0.0
        self.assertAlmostEqual(test_min, exp_min, delta=delta, msg="Expected min {}, got {}".format(exp_min, test_min))

        test_max = np.max(lbp)
        exp_max = 25.0
        self.assertAlmostEqual(test_max, exp_max, delta=delta, msg="Expected max {}, got {}".format(exp_max, test_max))

        test_mean = np.mean(lbp)
        exp_mean = 22.3147
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta,
                               msg="Expected mean {}, got {}".format(exp_mean, test_mean))

        test_median = np.median(lbp)
        exp_median = 24.0
        self.assertAlmostEqual(test_median, exp_median, delta=delta,
                               msg="Expected median {}, got {}".format(exp_median, test_median))

    def test_unchanged_features(self):
        for feature, value in self.initial_patch.data.items():
            self.assertTrue(np.array_equal(value, self.patch.data[feature]),
                            msg="EOPatch data feature '{}' was changed in the process".format(feature))


if __name__ == '__main__':
    unittest.main()
