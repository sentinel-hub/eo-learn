import unittest
import numpy as np

from eolearn.core import FeatureType, EOPatch
from eolearn.ml_tools import TrainSetMaskTask


class TestTrainSet(unittest.TestCase):

    def test_train_set_mask(self):
        new_name = 'TEST_TRAIN_MASK'

        input_mask_feature = (FeatureType.MASK_TIMELESS, 'TEST')
        input_label_feature = (FeatureType.LABEL_TIMELESS, 'TEST')
        new_mask_feature = (FeatureType.MASK_TIMELESS, new_name)
        new_label_feature = (FeatureType.LABEL_TIMELESS, new_name)

        self.assertRaises(ValueError, TrainSetMaskTask, input_mask_feature, None)
        self.assertRaises(ValueError, TrainSetMaskTask, input_mask_feature, 1.5)
        self.assertRaises(ValueError, TrainSetMaskTask, input_mask_feature, [0.5, 0.3, 0.7])

        shape = (8, 8, 3)
        size = np.prod(shape)

        data = np.ones(shape=shape, dtype=np.int)

        data[0:2, 0:2, :] = 11
        data[0:2, 2:4, :] = 22
        data[2:4, 0:2, :] = 33
        data[2:4, 2:4, :] = 44
        data[0:4, 4:8, :] = 55
        data[4:8, 0:4, :] = 66
        data[4:8, 4:8, :] = 77

        patch = EOPatch()
        patch[(FeatureType.MASK_TIMELESS, 'TEST')] = data
        patch[(FeatureType.LABEL_TIMELESS, 'TEST')] = data.copy().reshape((size,))

        bins = [0.2, 0.5, 0.8]

        patch = TrainSetMaskTask((*input_mask_feature, new_name), bins)(patch, seed=1)
        self.assertTrue(set(np.unique(patch[new_mask_feature])) <= set(range(len(bins) + 1)))

        patch = TrainSetMaskTask((*input_label_feature, new_name), bins)(patch, seed=1)
        self.assertTrue(set(np.unique(patch[new_label_feature])) <= set(range(len(bins) + 1)))

        shape = (10, 20, 20, 3)
        size = np.prod(shape)

        bins = [0.2, 0.5, 0.7, 0.8]
        patch[(FeatureType.DATA, 'TEST')] = np.random.randint(10, 30, size=shape)

        patch = TrainSetMaskTask((FeatureType.DATA, 'TEST', 'BINS'), bins)(patch)
        self.assertTrue(set(np.unique(patch[(FeatureType.DATA, 'BINS')])) <= set(range(len(bins) + 1)))


if __name__ == '__main__':
    unittest.main()
