"""
Module for transforming reference labels

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np

# pylint: disable=invalid-name


class Mask2Label:
    """Transforms mask (shape n x m) into a single label."""

    def __init__(self, mode, target_value=1, target_threshold=0.5):
        """
        Transformation works in two modes:

        - majority: assign label to the class with the largest contribution
        - target: assign label to target if its contribution percentage is above or equal to target threshold.
          In other cases label is set to 0.

        The parameters target_value and target_threshold are taken into account only in target mode.

        :param mode: A conversion mode, options are `'majority'` or `'target'`
        :type mode: str
        :param target_value: A target value
        :type target_value: int
        :param target_threshold: Fraction of pixels in mask that need to belong to the target to be labeled as target
        :type target_threshold: float
        """
        self.mode_ = mode
        if self.mode_ not in ["majority", "target"]:
            print("Invalid mode! Set mode to majority or target.")

        self.target_ = target_value
        self.threshold_ = target_threshold

    def _target(self, mask):
        unique, counts = np.unique(mask, return_counts=True)
        value_count = dict(zip(unique, counts))

        return (
            1
            if self.target_ in value_count.keys() and value_count[self.target_] / np.ma.size(mask) >= self.threshold_
            else 0
        )

    @staticmethod
    def _majority(mask):
        label, count = np.unique(mask, return_counts=True)

        return label[np.argmax(count)]

    def transform(self, X):
        """
        :param X: An array in form of (n, m)
        :type X: np.ndarray
        """
        if self.mode_ == "target":
            return np.apply_along_axis(self._target, 1, np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])))

        if self.mode_ == "majority":
            return np.apply_along_axis(self._majority, 1, np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])))

        print("Invalid mode! Set mode to majority or target. Returning input.")
        return X


class Mask2TwoClass:
    """
    The masks can include non-exclusive-multi-class labels described with bit pattern.
    This transformer simplifies the mask in form of bit-pattern to two class labels.
    """

    def __init__(self, positive_class_definition):
        """
        :param positive_class_definition: A bit pattern, (e.g. '100001') defining the positive class
            if argument is an int then the mask is not interpreted as bit pattern
        :type positive_class_definition: int or string
        """
        if isinstance(positive_class_definition, str):
            self.definition_ = int(positive_class_definition, 2)
            self.binary_ = True
        else:
            self.definition_ = positive_class_definition
            self.binary_ = False

    def transform(self, X):
        """
        :param X: An array in form of (n, m)
        :type X: np.ndarray
        """

        if self.binary_:
            return (X & self.definition_ > 0).astype(int)
        return (X == self.definition_).astype(int)
