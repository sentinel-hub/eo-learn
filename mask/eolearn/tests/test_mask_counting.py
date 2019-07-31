import unittest
import numpy as np

from eolearn.mask import CountValidTask, ClassFrequencyTask
from eolearn.core import EOPatch, FeatureType


class TestMaskCounting(unittest.TestCase):
    def test_count_values(self):
        patch = EOPatch()

        shape = (5, 3, 3, 2)

        data = np.random.randint(0, 5, size=shape)
        data[:, 0, 0, 0] = 0
        data[:, 0, 1, 0] = 2

        patch[(FeatureType.MASK, 'TEST')] = data

        patch = CountValidTask((FeatureType.MASK, 'TEST'), (FeatureType.MASK_TIMELESS, 'COUNT'))(patch)

        result = patch[(FeatureType.MASK_TIMELESS, 'COUNT')]

        self.assertEqual(result[0, 0, 0], 0)
        self.assertEqual(result[0, 1, 0], shape[0])

    def test_class_frequency(self):
        patch = EOPatch()

        shape = (20, 5, 5, 2)

        data = np.random.randint(0, 5, size=shape)
        data[:, 0, 0, 0] = 0
        data[:, 0, 1, 0] = 2

        patch[(FeatureType.MASK, 'TEST')] = data

        classes = [2, 1, 55]

        input_feature = (FeatureType.MASK, 'TEST')
        output_feature = (FeatureType.DATA_TIMELESS, 'FREQ')

        self.assertRaises(ValueError, ClassFrequencyTask, input_feature, output_feature, classes=['a', 'b'])
        self.assertRaises(ValueError, ClassFrequencyTask, input_feature, output_feature, classes=4)
        self.assertRaises(ValueError, ClassFrequencyTask, input_feature, output_feature, classes=None)

        patch = ClassFrequencyTask(input_feature, output_feature, classes)(patch)

        result = patch[output_feature]

        # all zeros through the temporal dimension should result in nan
        self.assertTrue(np.isnan(result[0, 0, 0]))

        # frequency of 2 should be 1
        self.assertEqual(result[0, 1, 0], 1)

        # frequency of 55 should be 0
        self.assertTrue(np.all(result[1:, :, -2] == 0))

if __name__ == '__main__':
    unittest.main()
