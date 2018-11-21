import unittest
import os

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import ErosionTask

import numpy as np


class TestErosion(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

    @classmethod
    def setUpClass(cls):
        cls.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_point_raster_sampler(self):
        # test erosion on all classes
        eopatch = EOPatch.load(self.TEST_PATCH_FILENAME, lazy_loading=True)
        mask_before = eopatch.mask_timeless['LULC'].copy()

        erosion_task = ErosionTask((FeatureType.MASK_TIMELESS, 'LULC', 'LULC_ERODED'), 1)
        eopatch = erosion_task.execute(eopatch)

        mask_after = eopatch.mask_timeless['LULC_ERODED'].copy()

        self.assertFalse(np.all(mask_before == mask_after))

        for label in self.classes:
            if label == 0:
                self.assertGreaterEqual(np.sum(mask_after == label), np.sum(mask_before == label),
                                        msg="error in the erosion process")
            else:
                self.assertLessEqual(np.sum(mask_after == label), np.sum(mask_before == label),
                                     msg="error in the erosion process")


if __name__ == '__main__':
    unittest.main()
