import unittest
import logging
import numpy as np

from eolearn.ml_tools import ImagePatchClassifier, ImagePixelClassifier, ImagePixel2PatchClassifier


logging.basicConfig(level=logging.DEBUG)


class BadClassifier:
    @staticmethod
    def _predict(data):
        return np.max(data)

    @staticmethod
    def _predict_proba(data):
        return np.max(data), np.min(data)

    def predict_proba(self, data):
        return np.asarray([self._predict_proba(example) for example in data], dtype=np.float)


class DummyPixelClassifier:
    @staticmethod
    def _predict(data):
        return np.max(data)

    @staticmethod
    def _predict_proba(data):
        return np.max(data), np.min(data)

    def predict(self, data):
        return np.asarray([self._predict(example) for example in data], dtype=np.int)

    def predict_proba(self, data):
        return np.asarray([self._predict_proba(example) for example in data], dtype=np.float)


class DummyPatchClassifier:
    def __init__(self, receptive_field):
        self.receptive_field = receptive_field

    @staticmethod
    def _predict(data):
        return np.max(data)

    @staticmethod
    def _predict_proba(data):
        return np.max(data), np.min(data)

    def predict(self, data):
        if data.shape[1:3] == self.receptive_field:
            return np.asarray([self._predict(example) for example in data], dtype=np.int)
        else:
            raise ValueError('Dummy Classifier: input of incorrect shape')

    def predict_proba(self, data):
        if data.shape[1:3] == self.receptive_field:
            return np.asarray([self._predict_proba(example) for example in data], dtype=np.float)
        else:
            raise ValueError('Dummy Classifier: input of incorrect shape')


class TestImageClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.receptive_field = (2, 2)
        cls.bad_classifier = BadClassifier()
        cls.pixel_classifier = DummyPixelClassifier()
        cls.patch_classifier = DummyPatchClassifier(receptive_field=cls.receptive_field)

    def test_initialisation(self):
        # Test class raises exceptions if classifier does not implement image_predict and image_predict_proba
        with self.assertRaises(ValueError):
            ImagePatchClassifier(self.bad_classifier, self.receptive_field)
            ImagePixelClassifier(self.bad_classifier)
            ImagePixel2PatchClassifier(self.bad_classifier, (4, 4))

    def test_pixel_classifier(self):
        # Test a dummy pixel classifier example
        image_classifier = ImagePixelClassifier(self.pixel_classifier)
        data = np.ones((20, 40, 40, 3), dtype=np.uint8)
        values = np.random.randint(0, 100, (3,), dtype=np.uint8)
        data *= values[None, None, None, :]
        data_predicted = image_classifier.image_predict(data)
        data_prob_predicted = image_classifier.image_predict_proba(data)
        self.assertTrue(np.array_equal(data_predicted, np.max(values)*np.ones((20, 40, 40), dtype=np.uint8)))
        self.assertTrue(np.array_equal(data_prob_predicted[..., 0],
                                       np.max(values)*np.ones((20, 40, 40), dtype=np.uint8)))
        self.assertTrue(np.array_equal(data_prob_predicted[..., 1],
                                       np.min(values)*np.ones((20, 40, 40), dtype=np.uint8)))

    def test_patch_classifier(self):
        # Test a dummy patch classifier example
        image_classifier = ImagePatchClassifier(self.patch_classifier, self.receptive_field)
        data = np.ones((20, 40, 40, 3), dtype=np.uint8)
        values = np.random.randint(0, 100, (3,), dtype=np.uint8)
        data *= values[None, None, None, :]
        data_predicted = image_classifier.image_predict(data)
        data_prob_predicted = image_classifier.image_predict_proba(data)
        self.assertTrue(np.array_equal(data_predicted, np.max(values)*np.ones((20, 40, 40), dtype=np.uint8)))
        self.assertTrue(np.array_equal(data_prob_predicted[..., 0],
                                       np.max(values)*np.ones((20, 40, 40), dtype=np.uint8)))
        self.assertTrue(np.array_equal(data_prob_predicted[..., 1],
                                       np.min(values)*np.ones((20, 40, 40), dtype=np.uint8)))


if __name__ == '__main__':
    unittest.main()
