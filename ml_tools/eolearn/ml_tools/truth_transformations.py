"""
Module for transforming reference labels

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np

# pylint: disable=invalid-name


class Mask2Label:
    """
    Transforms mask (shape n x m) into a single label. Transformation works in two modes:
        - majority: assign label to the class with the largest contribution
        - target: assign label to target if its contribution percentage is above or equal to target threshold.
        In other cases label is set to 0.

    The parameters target_value and target_threshold are taken into account only in target mode.

    Parameters
    ----------
    mode: str ('majority', 'target')
        conversion mode

    target_value: int (default 1)
        target value

    target_threshold: float (default 0.5)
        fraction of pixels in mask that need to belong to the target to be labeled as target
    """

    def __init__(self, mode, target_value=1, target_threshold=0.5):

        self.mode_ = mode
        if self.mode_ not in ['majority', 'target']:
            print('Invalid mode! Set mode to majority or target.')

        self.target_ = target_value
        self.threshold_ = target_threshold

    def _target(self, mask):
        unique, counts = np.unique(mask, return_counts=True)
        value_count = dict(zip(unique, counts))

        return 1 if self.target_ in value_count.keys() and \
                    value_count[self.target_] / np.ma.size(mask) >= self.threshold_ else 0

    @staticmethod
    def _majority(mask):
        label, count = np.unique(mask, return_counts=True)

        return label[np.argmax(count)]

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape [n x m]
            The mask in form of n x m array.
        """
        if self.mode_ == 'target':
            return np.apply_along_axis(self._target, 1, np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])))

        if self.mode_ == 'majority':
            return np.apply_along_axis(self._majority, 1, np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])))

        print('Invalid mode! Set mode to majority or target. Returning input.')
        return X


class Mask2TwoClass:
    """
    The masks can include non-exclusive-multi-class labels described with bit pattern.
    This transformer simplifies the mask in form of bit-pattern to two class labels.

    Parameters
    ----------
    positive_class_definition: int or string
        int or string (bit pattern, i.e. '100001') defining the positive class
        if argument is an int then the mask is not interpreted as bit pattern
    """

    def __init__(self, positive_class_definition):

        if isinstance(positive_class_definition, str):
            self.definition_ = int(positive_class_definition, 2)
            self.binary_ = True
        else:
            self.definition_ = positive_class_definition
            self.binary_ = False

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape [n x m]
            The mask in form of n x m array.
        """

        if self.binary_:
            return (X & self.definition_ > 0).astype(int)
        return (X == self.definition_).astype(int)
