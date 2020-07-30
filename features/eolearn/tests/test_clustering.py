import unittest
import os.path
import logging

from eolearn.core import EOPatch, FeatureType

from eolearn.features import ClusteringTask

import numpy as np

logging.basicConfig(level=logging.DEBUG)


class TestEOPatch(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        test_features = {FeatureType.DATA_TIMELESS: ['feature1', 'feature2']}

        ClusteringTask(features=test_features,
                       new_feature_name='clusters_small',
                       n_clusters=100,
                       affinity='cosine',
                       linkage='single',
                       remove_small=3).execute(cls.patch)

        ClusteringTask(features=test_features,
                       new_feature_name='clusters_mask',
                       distance_threshold=0.1,
                       affinity='cosine',
                       linkage='average',
                       mask_name='mask').execute(cls.patch)

    def test_clustering(self):
        clusters = self.patch.data_timeless['clusters_small'].squeeze()
        delta = 1e-3

        test_unique = len(np.unique(clusters))
        exp_unique = 26
        self.assertEqual(test_unique, exp_unique,
                         msg=" Expected number of clusters {}, got {}.".format(exp_unique, test_unique))

        test_median = np.median(clusters)
        exp_median = 92
        self.assertEqual(test_median, exp_median,
                         msg="Expected median {}, got {}.".format(exp_median, test_median))

        test_mean = np.mean(clusters)
        exp_mean = 68.665
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta,
                               msg="Expected mean {}, got {}.".format(exp_mean, test_mean))

        clusters = self.patch.data_timeless['clusters_mask'].squeeze()
        delta = 1e-4

        test_unique = len(np.unique(clusters))
        exp_unique = 45
        self.assertEqual(test_unique, exp_unique,
                         msg="Expected number of clusters {}, got {}.".format(exp_unique, test_unique))

        test_median = np.median(clusters)
        exp_median = -0.5
        self.assertEqual(test_median, exp_median,
                         msg="Expected median {}, got {}.".format(exp_median, test_median))

        test_mean = np.mean(clusters)
        exp_mean = 3.7075
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta,
                               msg="Expected mean {}, got {}.".format(exp_mean, test_mean))

        test_mask = np.all(clusters[0:5, 0:20] == -1)
        exp_mask = True
        self.assertTrue(test_mask, msg="Expected area to be {}, got {}.".format(exp_mask, test_mask))


if __name__ == '__main__':
    unittest.main()
