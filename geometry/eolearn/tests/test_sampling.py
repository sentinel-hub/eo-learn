import pytest

import unittest
import os

from eolearn.geometry import PointSampler

import numpy as np


class TestPointSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raster_size = (100, 100)
        cls.nsamples = 50
        cls.raster = np.zeros(cls.raster_size, dtype=np.uint8)
        cls.raster[40:60, 40:60] = 1

    def test_point_sampler(self):
        ps = PointSampler(self.raster)
        self.assertEqual(ps.area(), np.prod(self.raster_size), msg="incorrect total area")
        self.assertAlmostEqual(ps.area(cc_index=0), 400, msg="incorrect area for label 1")
        self.assertAlmostEqual(ps.area(cc_index=1), 9600, msg="incorrect area for label 0")
        self.assertTrue(ps.geometries[0]['polygon'].envelope.bounds == (40, 40, 60, 60), msg="incorrect polygon bounds")
        self.assertTrue(ps.geometries[1]['polygon'].envelope.bounds == (0, 0, 100, 100), msg="incorrect polygon bounds")
        del ps
        ps = PointSampler(self.raster, no_data_value=0)
        self.assertEqual(ps.area(), 400, msg="incorrect total area")
        self.assertEqual(len(ps), 1, msg="incorrect handling of no data values")
        del ps
        ps = PointSampler(self.raster, ignore_labels=[1])
        self.assertEqual(ps.area(), 9600, msg="incorrect total area")
        self.assertEqual(len(ps), 1, msg="incorrect handling of no data values")
        del ps


if __name__ == '__main__':
    unittest.main()
