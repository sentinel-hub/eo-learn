import pytest

import unittest
import os

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import PointSampler, PointRasterSampler, PointSamplingTask

import numpy as np


class TestPointSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raster_size = (100, 100)
        cls.n_samples = 100
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

    def test_point_raster_sampler(self):
        ps = PointRasterSampler()
        # test if it raises ndim error
        with self.assertRaises(ValueError):
            ps.sample(np.ones((1000,)))
            ps.sample(np.ones((1000, 1000, 3)))
        labels, rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        self.assertEqual(len(labels), self.n_samples, msg="incorrect number of samples")
        self.assertEqual(len(rows), self.n_samples, msg="incorrect number of samples")
        self.assertEqual(len(cols), self.n_samples, msg="incorrect number of samples")
        # test number of sample is proportional to class frequency
        self.assertEqual(np.sum(labels == 1), int(self.n_samples*(400/np.prod(self.raster_size))),
                         msg="incorrect sampling distribution")
        self.assertEqual(np.sum(labels == 0), int(self.n_samples*(9600/np.prod(self.raster_size))),
                         msg="incorrect sampling distribution")
        # test sampling is correct
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")
        del ps
        # test erosion of both classes (0 and 1)
        ps = PointRasterSampler(disk_radius=2)
        labels, rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        self.assertLessEqual(np.sum(labels == 1), 3, msg="incorrect sampling distribution")
        self.assertGreaterEqual(np.sum(labels == 0), 97, msg="incorrect sampling distribution")
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")
        del ps
        # test even sampling of classes
        ps = PointRasterSampler(even_sampling=True)
        labels, rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        self.assertEqual(np.sum(labels == 1), self.n_samples//2, msg="incorrect sampling distribution")
        self.assertEqual(np.sum(labels == 0), self.n_samples//2, msg="incorrect sampling distribution")
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")
        del ps
        # test no data value option
        ps = PointRasterSampler(no_data_value=0)
        labels, rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        self.assertEqual(np.sum(labels == 1), self.n_samples, msg="incorrect handling of no data value")
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")
        del ps
        # test ignore labels option
        ps = PointRasterSampler(disk_radius=2, ignore_labels=[0])
        labels, rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        self.assertEqual(np.sum(labels == 1), self.n_samples, msg="incorrect handling of ignore labels")
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")
        del ps
        ps = PointRasterSampler(disk_radius=2, ignore_labels=[1])
        labels, rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        self.assertEqual(np.sum(labels == 0), self.n_samples, msg="incorrect handling of ignore labels")
        self.assertTrue((labels == self.raster[rows, cols]).all(), msg="incorrect sampling")
        del ps

    def test_point_sampling_task(self):
        # test PointSamplingTask
        t, h, w, d = 10, 100, 100, 5
        eop = EOPatch()
        eop.data['bands'] = np.arange(t*h*w*d).reshape(t, h, w, d)
        eop.mask_timeless['raster'] = self.raster.reshape(self.raster_size + (1,))
        task = PointSamplingTask(n_samples=self.n_samples, sample_raster_name='raster', data_feature_name='bands',
                                 even_sampling=True)
        task.execute(eop)
        # assert features, labels and sampled rows and cols are added to eopatch
        self.assertIn('LABELS', eop.label_timeless, msg="labels not added to eopatch")
        self.assertIn('SAMPLE_ROWS', eop.label_timeless, msg="sampled cols not added to eopatch")
        self.assertIn('SAMPLE_COLS', eop.label_timeless, msg="sampled rows not added to eopatch")
        self.assertIn('FEATS', eop.data, msg="features not added to eopatch")
        # check validity of sampling
        self.assertTupleEqual(eop.data['FEATS'].shape, (t, self.n_samples, 1, d), msg="incorrect features size")
        self.assertEqual(len(eop.label_timeless['LABELS']), self.n_samples, msg="incorrect number of samples")
        self.assertTrue((eop.label_timeless['LABELS'] ==
                         eop.mask_timeless['raster'][eop.label_timeless['SAMPLE_ROWS'],
                                                     eop.label_timeless['SAMPLE_COLS']].squeeze()).all(),
                        msg="incorrect sampling")
        self.assertTrue((eop.data['FEATS'] ==
                         eop.data['bands'][:, eop.label_timeless['SAMPLE_ROWS'],
                         eop.label_timeless['SAMPLE_COLS'], :].reshape(t, self.n_samples, 1, d)).all(),
                        msg="incorrect sampling")


if __name__ == '__main__':
    unittest.main()
