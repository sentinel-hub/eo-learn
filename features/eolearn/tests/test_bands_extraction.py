import unittest
import os

import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.features import EuclideanNormTask

class TestErosion(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    def test_euclidean_norm(self):
        eopatch = EOPatch()

        data = np.zeros(5*10*10*7).reshape(5, 10, 10, 7)
        bands = [0, 1, 2, 4, 6]
        data[..., bands] = 1

        eopatch.data['TEST'] = data

        eopatch = EuclideanNormTask((FeatureType.DATA, 'TEST'), (FeatureType.DATA, 'NORM'), bands)(eopatch)
        self.assertTrue(np.all(eopatch.data['NORM'] == np.sqrt(len(bands))))

if __name__ == '__main__':
    unittest.main()
