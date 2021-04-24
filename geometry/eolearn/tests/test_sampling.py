"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os
import unittest

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import PointSampler, PointRasterSampler, PointSamplingTask, BalancedClassSampler, \
    BalancedClassSamplerTask

import numpy as np


class TestSampling(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                       'TestEOPatch')

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
        ps = PointRasterSampler([0, 1])
        # test if it raises ndim error
        with self.assertRaises(ValueError):
            ps.sample(np.ones((1000,)))
            ps.sample(np.ones((1000, 1000, 3)))

        rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        labels = self.raster[rows, cols]
        self.assertEqual(len(labels), self.n_samples, msg="incorrect number of samples")
        self.assertEqual(len(rows), self.n_samples, msg="incorrect number of samples")
        self.assertEqual(len(cols), self.n_samples, msg="incorrect number of samples")
        # test number of sample is proportional to class frequency
        self.assertEqual(np.sum(labels == 1), int(self.n_samples * (400 / np.prod(self.raster_size))),
                         msg="incorrect sampling distribution")
        self.assertEqual(np.sum(labels == 0), int(self.n_samples * (9600 / np.prod(self.raster_size))),
                         msg="incorrect sampling distribution")
        # test sampling is correct
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")
        del ps
        # test even sampling of classes
        ps = PointRasterSampler([0, 1], even_sampling=True)
        rows, cols = ps.sample(self.raster, n_samples=self.n_samples)
        labels = self.raster[rows, cols]
        self.assertEqual(np.sum(labels == 1), self.n_samples // 2, msg="incorrect sampling distribution")
        self.assertEqual(np.sum(labels == 0), self.n_samples // 2, msg="incorrect sampling distribution")
        self.assertEqual((labels == self.raster[rows, cols]).all(), True, msg="incorrect sampling")

    def test_point_sampling_task(self):
        # test PointSamplingTask
        t, h, w, d = 10, 100, 100, 5
        eop = EOPatch()
        eop.data['bands'] = np.arange(t * h * w * d).reshape(t, h, w, d)
        eop.mask_timeless['raster'] = self.raster.reshape(self.raster_size + (1,))

        task = PointSamplingTask(n_samples=self.n_samples, ref_mask_feature='raster', ref_labels=[0, 1],
                                 sample_features=[(FeatureType.DATA, 'bands', 'SAMPLED_DATA'),
                                                  (FeatureType.MASK_TIMELESS, 'raster', 'SAMPLED_LABELS')],
                                 even_sampling=True)
        task.execute(eop)
        # assert features, labels and sampled rows and cols are added to eopatch
        self.assertIn('SAMPLED_LABELS', eop.mask_timeless, msg="labels not added to eopatch")
        self.assertIn('SAMPLED_DATA', eop.data, msg="features not added to eopatch")
        # check validity of sampling
        self.assertTupleEqual(eop.data['SAMPLED_DATA'].shape, (t, self.n_samples, 1, d), msg="incorrect features size")
        self.assertTupleEqual(eop.mask_timeless['SAMPLED_LABELS'].shape, (self.n_samples, 1, 1),
                              msg="incorrect number of samples")

    def test_balanced_class_sampler(self):
        sampling = BalancedClassSampler(class_feature='LULC',
                                        samples_amount=0.1,
                                        valid_mask='EDGES_INV',
                                        ignore_labels=8,
                                        features={FeatureType.DATA_TIMELESS: ['DEM', 'MAX_NDVI']},
                                        weak_classes=1,
                                        search_radius=3,
                                        samples_per_class=5,
                                        seed=123)

        sampling.sample_patch(self.TEST_PATCH_FILENAME)
        samples = sampling.get_balanced_data()
        frequency = sampling.get_prior_class_distribution()

        compare_frequency = {0: 20, 2: 822, 3: 138, 4: 29, 1: 2}
        for key in compare_frequency:
            self.assertEqual(frequency[key], compare_frequency[key])

        self.assertEqual(len(samples.index), 25)

        self.assertEqual(samples.iloc[1, 2], 79)
        self.assertEqual(samples.iloc[5, 0], 3)
        self.assertAlmostEqual(samples.iloc[10, 4], 664.0, delta=0.01)
        self.assertAlmostEqual(samples.iloc[18, 5], 0.763, delta=0.01)

    def test_balanced_class_sampler_task(self):
        sampling = BalancedClassSamplerTask(class_feature='LULC', seed=321)

        eopatch = EOPatch.load(self.TEST_PATCH_FILENAME)
        sampling.execute(eopatch)

        samples = sampling.get_balanced_data()
        frequency = sampling.get_prior_class_distribution()
        compare_frequency = {2: 779, 3: 163, 8: 12, 0: 10, 4: 44, 1: 2}
        for key in compare_frequency:
            self.assertEqual(frequency[key], compare_frequency[key])

        self.assertEqual(len(samples.index), 12)

        self.assertEqual(samples.iloc[1, 2], 56)
        self.assertEqual(samples.iloc[5, 0], 4)


if __name__ == '__main__':
    unittest.main()
