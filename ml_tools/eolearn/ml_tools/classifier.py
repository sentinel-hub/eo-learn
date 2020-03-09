"""
Module for classification helper classes and classification task.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
# pylint: disable=invalid-name

import itertools
from abc import ABC, abstractmethod

import numpy as np

from eolearn.core import EOTask

from .utilities import rolling_window


class ImageBaseClassifier(ABC):
    """
    Abstract class for image classifiers.

    Image Classifier extends the receptive field of trained classifier with smaller
    receptive field over entire image. The classifier's receptive field is
    usually small, i.e.:
        - pixel based classifier has receptive field (1,1)
        - patch based classifier has receptive field (num_pixels_y, num_pixels_x)

    Image Classifier divides the image into non-overlapping pieces of same size
    as trained classifier's receptive field and runs classifier over them thus
    producing a classification mask of the same size as image.

    The classifier can be of any type as long as it has the following two
    methods implemented:
        - predict(X)
        - predict_proba(X)

    This is true for all classifiers that follow scikit-learn's API.
    The APIs of scikit-learn's objects is described
    at: http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects.

    Parameters:
    -----------

    classifier:
        The actual trained classifier that will be executed over entire image

    receptive_field: tuple, (n_rows, n_columns)
        Sensitive area of the classifier ((1,1) for pixel based or (n,m) for patch base)

    """

    def __init__(self, classifier, receptive_field):
        self.receptive_field = receptive_field

        self._check_classifier(classifier)
        self.classifier = classifier

        self._samples = None
        self._image_size = None

    @staticmethod
    def _check_classifier(classifier):
        """
        Check if the classifier implements predict and predict_proba methods.
        """
        predict = getattr(classifier, "predict", None)
        if not callable(predict):
            raise ValueError('Classifier does not have predict method!')

        predict_proba = getattr(classifier, "predict_proba", None)
        if not callable(predict_proba):
            raise ValueError('Classifier does not have predict_proba method!')

    def _check_image(self, X):
        """
        Checks the image size and its compatibility with classifier's receptive field.

        At this moment it is required that image size = K * receptive_field. This will
        be relaxed in future with the introduction of padding.
        """

        if (len(X.shape) < 3) or (len(X.shape) > 4):
            raise ValueError('Input has to have shape [n_samples, n_pixels_y, n_pixels_x] '
                             'or [n_samples, n_pixels_y, n_pixels_x, n_bands].')

        self._samples = X.shape[0]
        self._image_size = X.shape[1:3]

        if (self._image_size[0] % self.receptive_field[0]) or (self._image_size[0] % self.receptive_field[0]):
            raise ValueError('Image (%d,%d) and receptive fields (%d,%d) mismatch.\n'
                             'Resize your image to be divisible with receptive field.'
                             % (self._image_size[0], self._image_size[0], self.receptive_field[0],
                                self.receptive_field[1]))

    @staticmethod
    def _transform_input(X):
        """
        Transform the input in the form expected by the classifier. For example reshape matrix to vector.
        """
        return X

    @abstractmethod
    def image_predict(self, X):
        """
        Predicts class label for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_y, n_pixels_x, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_y, n_pixels_x]
            Target labels or masks.
        """
        raise NotImplementedError

    @abstractmethod
    def image_predict_proba(self, X):
        """
        Predicts class probabilities for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_x, n_pixels_y, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_x, n_pixels_y, n_classes]
            Target probabilities
        """
        raise NotImplementedError


class ImagePixelClassifier(ImageBaseClassifier):
    """
    Pixel classifier divides the image into individual pixels,
    runs classifier and collects the result in the shape of the input image.

    Parameters:
    -----------

    classifier:
        The actual trained classifier that will be executed over entire image
    """

    def __init__(self, classifier):
        ImageBaseClassifier.__init__(self, classifier, (1, 1))

    def image_predict(self, X):
        """
        Predicts class label for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_y, n_pixels_x, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_y, n_pixels_x]
            Target labels or masks.
        """
        self._check_image(X)

        new_shape = (X.shape[0] * X.shape[1] * X.shape[2],)

        if len(X.shape) == 4:
            new_shape += (X.shape[3],)

        pixels = X.reshape(new_shape)

        predictions = self.classifier.predict(self._transform_input(pixels))

        return predictions.reshape(X.shape[0], X.shape[1], X.shape[2])

    def image_predict_proba(self, X):
        """
        Predicts class probabilities for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_x, n_pixels_y, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_x, n_pixels_y, n_classes]
            Target probabilities
        """
        self._check_image(X)

        new_shape = (X.shape[0] * X.shape[1] * X.shape[2],)

        if len(X.shape) == 4:
            new_shape += (X.shape[3],)

        pixels = X.reshape(new_shape)

        probabilities = self.classifier.predict_proba(self._transform_input(pixels))

        return probabilities.reshape(X.shape[0], X.shape[1], X.shape[2],
                                     probabilities.shape[1])


class ImagePatchClassifier(ImageBaseClassifier):
    """
    Patch classifier divides the image into non-overlapping patches of same size
    as trained classifier's receptieve field and runs classifier over them thus
    producing a classification mask of the same size as image.

    Parameters:
    -----------

    classifier:
        The actual trained classifier that will be executed over entire image

    receptive_field: tuple, (n_rows, n_columns)
        Sensitive area of the classifier ((1,1) for pixel based or (n,m) for patch based)

    """

    def _to_patches(self, X):
        """
        Reshapes input to patches of the size of classifier's receptive field.

        For example:

        input X shape: [n_samples, n_pixels_y, n_pixels_x, n_bands]

        output: [n_samples * n_pixels_y/receptive_field_y * n_pixels_x/receptive_field_x,
                 receptive_field_y, receptive_field_x, n_bands]
        """

        window = self.receptive_field
        asteps = self.receptive_field

        if len(X.shape) == 4:
            window += (0,)
            asteps += (1,)

        image_view = rolling_window(X, window, asteps)

        new_shape = image_view.shape

        # this makes a copy of the array? can we do without reshaping?
        image_view = image_view.reshape((new_shape[0] * new_shape[1] * new_shape[2],) + new_shape[3:])

        if len(X.shape) == 4:
            image_view = np.moveaxis(image_view, 1, -1)

        return image_view, new_shape

    def image_predict(self, X):
        """
        Predicts class label for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_y, n_pixels_x, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_y, n_pixels_x]
            Target labels or masks.
        """
        self._check_image(X)

        patches, patches_shape = self._to_patches(X)

        predictions = self.classifier.predict(self._transform_input(patches))

        image_predictions = predictions.reshape(patches_shape[0:3])

        image_results = np.zeros((self._samples,) + self._image_size)

        nx, ny = self.receptive_field
        row_steps = self._image_size[0] // nx
        col_steps = self._image_size[1] // ny

        # how can this be optimised?
        for i, j, k in itertools.product(range(row_steps), range(col_steps), range(self._samples)):
            image_results[k, nx * i:nx * (i + 1), ny * j:ny * (j + 1)] = image_predictions[k, i, j]

        return image_results

    def image_predict_proba(self, X):
        """
        Predicts class probabilities for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_x, n_pixels_y, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_x, n_pixels_y, n_classes]
            Target probabilities
        """
        self._check_image(X)

        patches, patches_shape = self._to_patches(X)

        probabilities = self.classifier.predict_proba(self._transform_input(patches))

        image_probabilities = probabilities.reshape(patches_shape[0:3] + (probabilities.shape[1],))

        image_results = np.zeros((self._samples,) + self._image_size + (probabilities.shape[1],))

        nx, ny = self.receptive_field
        row_steps = self._image_size[0] // nx
        col_steps = self._image_size[1] // ny

        # how can this be optimised?
        for i, j, k in itertools.product(range(row_steps), range(col_steps), range(self._samples)):
            image_results[k, nx * i:nx * (i + 1), ny * j:ny * (j + 1), :] = image_probabilities[k, i, j, :]

        return image_results


class ImagePixel2PatchClassifier(ImageBaseClassifier):
    """
    Pixel to patch classifier first performs classification on pixel level
    and then combines the results in user defined patches. In case of combining
    probabilities the weighted sum is taken over all pixels in a patch. In case
    of predictions the user defines what fraction of pixels within the patch
    has to belong to signal class ot be considered as signal.

    Parameters:
    -----------

    classifier:
        The actual trained classifier that will be executed over entire image

    patch_size: tuple, (n_rows, n_columns)
        Patch size

    target: int
        Target class value. Set the patch class to this target class if its fractional representation within
        this patch is above the target_threshols

    target_threshold: float
        See above

    """

    def __init__(self, classifier, patch_size, mode='mean_prob', target=None, target_threshold=None):
        self.pixel_classifier = ImagePixelClassifier(classifier)
        self.patch_size = patch_size
        self.target = target
        self.target_threshold = target_threshold

        self.mode = mode

        ImageBaseClassifier.__init__(self, classifier, (1, 1))

    def _to_patches(self, X):
        """
        Reshapes input to patches of the size of classifier's receptive field.

        For example:

        input X shape: [n_samples, n_pixels_y, n_pixels_x, n_bands]

        output: [n_samples * n_pixels_y/receptive_field_y * n_pixels_x/receptive_field_x,
                 receptive_field_y, receptive_field_x, n_bands]
        """

        window = self.patch_size
        asteps = self.patch_size

        if len(X.shape) == 4:
            window += (0,)
            asteps += (1,)

        image_view = rolling_window(X, window, asteps)

        new_shape = image_view.shape

        return image_view, new_shape

    def _target(self, array):
        unique, counts = np.unique(array, return_counts=True)
        valuecount = dict(zip(unique, counts))

        return 1 if self.target in valuecount.keys() and \
                    valuecount[self.target] / np.ma.size(array) >= self.target_threshold else 0

    def image_predict(self, X):
        """
        Predicts class label for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_y, n_pixels_x, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_y, n_pixels_x]
            Target labels or masks.
        """
        self._check_image(X)

        if self.mode == 'majority_class':
            predictions = self.pixel_classifier.image_predict(X)

        elif self.mode == 'mean_prob':
            probabilities = self.image_predict_proba(X)
            predictions = (probabilities[..., self.target] > self.target_threshold).astype(np.int)

        patches, _ = self._to_patches(predictions)

        row_steps = self._image_size[0] // self.patch_size[0]
        col_steps = self._image_size[1] // self.patch_size[1]

        # how can this be optimised?
        for i, j, k in itertools.product(range(row_steps), range(col_steps), range(self._samples)):
            patches[k, i, j] = self._target(patches[k, i, j])

        return predictions

    def image_predict_proba(self, X):
        """
        Predicts class probabilities for the entire image.

        Parameters:
        -----------

        X: array, shape = [n_samples, n_pixels_x, n_pixels_y, n_bands]
            Array of training images

        y: array, shape = [n_samples] or [n_samples, n_pixels_x, n_pixels_y, n_classes]
            Target probabilities
        """
        self._check_image(X)

        probabilities = self.pixel_classifier.image_predict_proba(X)

        patches, _ = self._to_patches(probabilities)

        row_steps = self._image_size[0] // self.patch_size[0]
        col_steps = self._image_size[1] // self.patch_size[1]

        ps = self.patch_size[0] * self.patch_size[1]

        # how can this be optimised?
        for i, j, k in itertools.product(range(row_steps), range(col_steps), range(self._samples)):
            patches[k, i, j, 0] = np.sum(patches[k, i, j, 0]) / ps
            patches[k, i, j, 1] = np.sum(patches[k, i, j, 1]) / ps

        return probabilities


class ImageClassificationMaskTask(EOTask):
    """
    This task applies pixel-based uni-temporal classifier to each image in the patch
    and appends to each image the classification mask.
    """
    def __init__(self, input_feature, output_feature, classifier):
        """ Run a classification task on a EOPatch feature

            Classifier is an instance of the ImageBaseClassifier that maps [w, h, d] numpy arrays (d-channel images)
            into [w, h, 1] numpy arrays (classification masks).

            :param input_feature: Feature which will be classified
            :type input_feature: (FeatureType, str)
            :param output_feature: Feature where classification results will be saved
            :type output_feature: (FeatureType, str)
            :param classifier: A classifier that works over [n, w, h, d]-dimensional numpy arrays.
            :type classifier: ImageBaseClassifier
        """
        self.input_feature = self._parse_features(input_feature)
        self.output_feature = self._parse_features(output_feature)
        self.classifier = classifier

    def execute(self, eopatch):
        """ Transforms [n, w, h, d] eopatch into a [n, w, h, 1] eopatch, adding it the classification mask.

            :param eopatch: An input EOPatch
            :type eopatch: EOPatch
            :return: Outputs EOPatch with n classification masks appended to out_feature_type with out_feature_name key
            :rtype: EOPatch
        """
        in_type, in_name = next(self.input_feature(eopatch))
        out_type, out_name = next(self.input_feature())

        eopatch[out_type][out_name] = self.classifier.image_predict(eopatch[in_type][in_name])

        return eopatch
