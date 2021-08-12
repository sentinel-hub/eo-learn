"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging

import pytest
import numpy as np
from numpy.testing import assert_array_equal

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
        return np.asarray([self._predict_proba(example) for example in data], dtype=float)


class DummyPixelClassifier:
    @staticmethod
    def _predict(data):
        return np.max(data)

    @staticmethod
    def _predict_proba(data):
        return np.max(data), np.min(data)

    def predict(self, data):
        return np.asarray([self._predict(example) for example in data], dtype=int)

    def predict_proba(self, data):
        return np.asarray([self._predict_proba(example) for example in data], dtype=float)


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
            return np.asarray([self._predict(example) for example in data], dtype=int)
        raise ValueError('Dummy Classifier: input of incorrect shape')

    def predict_proba(self, data):
        if data.shape[1:3] == self.receptive_field:
            return np.asarray([self._predict_proba(example) for example in data], dtype=float)
        raise ValueError('Dummy Classifier: input of incorrect shape')


RECEPTIVE_FIELD = (2, 2)
BAD_CLASSIFIER = BadClassifier()
PIXEL_CLASSIFIER = DummyPixelClassifier()
PATCH_CLASSIFIER = DummyPatchClassifier(receptive_field=RECEPTIVE_FIELD)


@pytest.mark.parametrize('classifier_class, faulty_params', [
    (ImagePatchClassifier, (BAD_CLASSIFIER, RECEPTIVE_FIELD)),
    (ImagePixelClassifier, (BAD_CLASSIFIER,)),
    (ImagePixel2PatchClassifier, (BAD_CLASSIFIER, (4, 4))),
])
def test_initialisation(classifier_class, faulty_params):
    # Test class raises exceptions if classifier does not implement image_predict and image_predict_proba
    with pytest.raises(ValueError):
        classifier_class(*faulty_params)


def test_pixel_classifier():
    # Test a dummy pixel classifier example
    image_classifier = ImagePixelClassifier(PIXEL_CLASSIFIER)
    data = np.ones((20, 40, 40, 3), dtype=np.uint8)
    values = np.random.randint(0, 100, (3,), dtype=np.uint8)
    data *= values[None, None, None, :]

    data_predicted = image_classifier.image_predict(data)
    data_prob_predicted = image_classifier.image_predict_proba(data)

    assert_array_equal(data_predicted, np.max(values)*np.ones((20, 40, 40), dtype=np.uint8))
    assert_array_equal(data_prob_predicted[..., 0], np.max(values) * np.ones((20, 40, 40), dtype=np.uint8))
    assert_array_equal(data_prob_predicted[..., 1], np.min(values) * np.ones((20, 40, 40), dtype=np.uint8))


def test_patch_classifier():
    # Test a dummy patch classifier example
    image_classifier = ImagePatchClassifier(PATCH_CLASSIFIER, RECEPTIVE_FIELD)
    data = np.ones((20, 40, 40, 3), dtype=np.uint8)
    values = np.random.randint(0, 100, (3,), dtype=np.uint8)
    data *= values[None, None, None, :]

    data_predicted = image_classifier.image_predict(data)
    data_prob_predicted = image_classifier.image_predict_proba(data)

    assert_array_equal(data_predicted, np.max(values)*np.ones((20, 40, 40), dtype=np.uint8))
    assert_array_equal(data_prob_predicted[..., 0], np.max(values) * np.ones((20, 40, 40), dtype=np.uint8))
    assert_array_equal(data_prob_predicted[..., 1], np.min(values) * np.ones((20, 40, 40), dtype=np.uint8))
