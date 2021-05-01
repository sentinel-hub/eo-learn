import os
import unittest
import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.mask.edge_extraction import EdgeExtractionTask


class TestEdgeExtractionTask(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    @classmethod
    def setUpClass(cls):
        cls.eopatch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls.eopatch.data['BLUE'] = cls.eopatch.data['BANDS-S2-L1C'][..., 1, np.newaxis]
        EdgeExtractionTask({FeatureType.DATA: ['NDVI', 'BLUE']}).execute(cls.eopatch)

    def test_edge_mask(self):
        exp_slice = [True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, False,
                     False, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, False, False, True, True, True,
                     False, False, True, True, True, True, True, True, True, True, True, True,
                     False, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True]

        mask_slice = self.eopatch.mask_timeless['EDGES_INV'][50]
        for i in range(len(exp_slice)):
            self.assertEqual(mask_slice[i], exp_slice[i])

        exp_sum = 9211
        mask_sum = np.sum(self.eopatch.mask_timeless['EDGES_INV'])
        self.assertEqual(mask_sum, exp_sum)

    if __name__ == '__main__':
        unittest.main()
