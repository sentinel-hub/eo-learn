import unittest
import os.path
import numpy as np
from skimage.feature import blob_dog

from eolearn.core import EOPatch, FeatureType
from eolearn.features import BlobTask, DoGBlobTask, LoGBlobTask, DoHBlobTask


class TestBlob(unittest.TestCase):

    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.patch)

        BlobTask((FeatureType.DATA, 'ndvi', 'blob'), blob_dog, sigma_ratio=1.6, min_sigma=1, max_sigma=30,
                 overlap=0.5, threshold=0).execute(cls.patch)
        DoGBlobTask((FeatureType.DATA, 'ndvi', 'blob_dog'), threshold=0).execute(cls.patch)
        LoGBlobTask((FeatureType.DATA, 'ndvi', 'blob_log'), log_scale=True, threshold=0).execute(cls.patch)
        DoHBlobTask((FeatureType.DATA, 'ndvi', 'blob_doh'), num_sigma=5, threshold=0).execute(cls.patch)

        cls.initial_patch = EOPatch.load(cls.TEST_PATCH_FILENAME)
        cls._prepare_patch(cls.initial_patch)

    @staticmethod
    def _prepare_patch(patch):
        ndvi = patch.data['ndvi'][:10]
        ndvi[np.isnan(ndvi)] = 0
        patch.data['ndvi'] = ndvi

    def test_blob_feature(self):
        self.assertTrue(np.allclose(self.patch.data['blob'], self.patch.data['blob_dog']),
                        msg='DoG derived class result not equal to base class result')

    def test_dog_feature(self):
        blob = self.patch.data['blob_dog']
        delta = 1e-4

        test_min = np.min(blob)
        exp_min = 0.0
        self.assertAlmostEqual(test_min, exp_min, delta=delta, msg="Expected min {}, got {}".format(exp_min, test_min))

        test_max = np.max(blob)
        exp_max = 37.9625
        self.assertAlmostEqual(test_max, exp_max, delta=delta, msg="Expected max {}, got {}".format(exp_max, test_max))

        test_mean = np.mean(blob)
        exp_mean = 0.0545
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta,
                               msg="Expected mean {}, got {}".format(exp_mean, test_mean))

        test_median = np.median(blob)
        exp_median = 0.0
        self.assertAlmostEqual(test_median, exp_median, delta=delta,
                               msg="Expected median {}, got {}".format(exp_median, test_median))

    def test_log_feature(self):
        blob = self.patch.data['blob_log']
        delta = 1e-4

        test_min = np.min(blob)
        exp_min = 0.0
        self.assertAlmostEqual(test_min, exp_min, delta=delta, msg="Expected min {}, got {}".format(exp_min, test_min))

        test_max = np.max(blob)
        exp_max = 13.65408
        self.assertAlmostEqual(test_max, exp_max, delta=delta, msg="Expected max {}, got {}".format(exp_max, test_max))

        test_mean = np.mean(blob)
        exp_mean = 0.05728
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta,
                               msg="Expected mean {}, got {}".format(exp_mean, test_mean))

        test_median = np.median(blob)
        exp_median = 0.0
        self.assertAlmostEqual(test_median, exp_median, delta=delta,
                               msg="Expected median {}, got {}".format(exp_median, test_median))

    def test_doh_feature(self):
        blob = self.patch.data['blob_doh']
        delta = 1e-4

        test_min = np.min(blob)
        exp_min = 0.0
        self.assertAlmostEqual(test_min, exp_min, delta=delta, msg="Expected min {}, got {}".format(exp_min, test_min))

        test_max = np.max(blob)
        exp_max = 1.4142
        self.assertAlmostEqual(test_max, exp_max, delta=delta, msg="Expected max {}, got {}".format(exp_max, test_max))

        test_mean = np.mean(blob)
        exp_mean = 0.0007
        self.assertAlmostEqual(test_mean, exp_mean, delta=delta,
                               msg="Expected mean {}, got {}".format(exp_mean, test_mean))

        test_median = np.median(blob)
        exp_median = 0.0
        self.assertAlmostEqual(test_median, exp_median, delta=delta,
                               msg="Expected median {}, got {}".format(exp_median, test_median))

    def test_unchanged_features(self):
        for feature, value in self.initial_patch.data.items():
            self.assertTrue(np.array_equal(value, self.patch.data[feature]),
                            msg="EOPatch data feature '{}' was changed in the process".format(feature))


if __name__ == '__main__':
    unittest.main()
