"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2019 Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import numpy as np

from eolearn.mask import ClassFrequencyTask
from eolearn.core import EOPatch, FeatureType


class TestMaskCounting(unittest.TestCase):
    def test_class_frequency(self):
        patch = EOPatch()

        shape = (20, 5, 5, 2)

        data = np.random.randint(0, 5, size=shape)
        data[:, 0, 0, 0] = 0
        data[:, 0, 1, 0] = 2

        patch[(FeatureType.MASK, 'TEST')] = data

        classes = [2, 1, 55]

        in_feature = (FeatureType.MASK, 'TEST')
        out_feature = (FeatureType.DATA_TIMELESS, 'FREQ')

        self.assertRaises(ValueError, ClassFrequencyTask, in_feature, out_feature, classes=['a', 'b'])
        self.assertRaises(ValueError, ClassFrequencyTask, in_feature, out_feature, classes=4)
        self.assertRaises(ValueError, ClassFrequencyTask, in_feature, out_feature, classes=None)
        self.assertRaises(ValueError, ClassFrequencyTask, in_feature, out_feature, classes=[1, 2, 3], no_data_value=2)

        patch = ClassFrequencyTask(in_feature, out_feature, classes)(patch)

        result = patch[out_feature]

        # all zeros through the temporal dimension should result in nan
        self.assertTrue(np.isnan(result[0, 0, 0]))

        # frequency of 2 should be 1
        self.assertEqual(result[0, 1, 0], 1)

        # frequency of 55 should be 0
        self.assertTrue(np.all(result[1:, :, -2] == 0))

        self.assertTrue(result.shape == (5, 5, 6))


if __name__ == '__main__':
    unittest.main()
