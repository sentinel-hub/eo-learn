"""
Tasks used for train set preparation

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc (Sinergise)
Copyright (c) 2017-2019 Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from enum import Enum
import numpy as np

from eolearn.core import EOTask, FeatureType


class TrainTestSplitType(Enum):
    """ An enum defining TrainTestSplitTask's methods of splitting the data into subsets
    """
    PER_PIXEL = 'per_pixel'
    PER_CLASS = 'per_class'
    PER_VALUE = 'per_value'


class TrainTestSplitTask(EOTask):
    """ Randomly assign each pixel or groups of pixels to multiple subsets (e.g., test/train/validate).

    Input pixels are defined by an input feature (e.g., MASK_TIMELESS with polygon ids, connected component ids, or
    similar), that groups together pixels with similar properties.

    There are three modes of split operation:

    - :attr:`PER_PIXEL<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_PIXEL>` (default), where pixels are
      assigned to a subset randomly, regardless of their value,

    - :attr:`PER_CLASS<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_CLASS>`, where pixels of the same value
      are assigned to the same subset,

    - :attr:`PER_VALUE<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_VALUE>`, where pixels of the same value
      are assigned to a the same subset consistently across eopatches. In other words, if a group of pixels of the same
      value lies on multiple eopatches, they are assigned to the same subset in all eopatches. In this case, the *seed*
      argument of the *execute* method is ignored.

    Classes are defined by a list of cumulative probabilities, passed as the *bins* argument, the same way as the *bins*
    argument in `numpy.digitize <https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html>`_. Valid
    classes are enumerated from 1 onward and if no_data_value is provided, all values equal to it get assigned to class
    0.

    To get a train/test split as 80/20, bins argument should be provided as `bins=[0.8]`.

    To get a train/val/test split as 60/20/20, bins argument should be provided as `bins=[0.6, 0.8]`.

    Splits can also be made into as many subsets as desired, e.g., `bins=[0.1, 0.2, 0.3, 0.7, 0.9]`.

    After the execution of this task an EOPatch will have a new (FeatureType, new_name) feature where each pixel will
    have a value representing the train, test and/or validation set.
    """

    def __init__(self, feature, bins, split_type=TrainTestSplitType.PER_PIXEL, ignore_values=None):
        """
        :param feature: The input feature out of which to generate the train mask.
        :type feature: (FeatureType, feature_name, new_name)
        :param bins: Cumulative probabilities of all value classes or a single float, representing a fraction.
        :type bins: a float or list of floats
        :param split_type: Valye split type, either 'per_pixel', 'per_class' or 'per_value'.
        :type split_type: str
        :param ignore_values: A list of values to ignore and not assign them to any subsets.
        :type ignore_values: a list of integers
        """
        allowed_types = [FeatureType.MASK_TIMELESS]
        self.feature = next(self._parse_features(feature, new_names=True, allowed_feature_types=allowed_types)())

        if np.isscalar(bins):
            bins = [bins]

        if not isinstance(bins, list) or not all(isinstance(bi, float) for bi in bins) \
                or np.any(np.diff(bins) <= 0) or bins[0] <= 0 or bins[-1] >= 1:
            raise ValueError('bins argument should be a list of ascending floats inside an open interval (0, 1)')

        self.ignore_values = set() if ignore_values is None else set(ignore_values)
        self.bins = bins
        self.split_type = TrainTestSplitType(split_type)

    def execute(self, eopatch, *, seed=None):
        """
        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param seed: An argument to be passed to numpy.random.seed function.
        :type seed: numpy.int64
        :return: Input EOPatch with the train set mask.
        :rtype: EOPatch
        """
        if self.split_type in [TrainTestSplitType.PER_CLASS, TrainTestSplitType.PER_PIXEL]:
            np.random.seed(seed)

        ftype, fname, new_name = self.feature
        data = eopatch[(ftype, fname)]

        if self.split_type == TrainTestSplitType.PER_PIXEL:
            rands = np.random.rand(*data.shape)
            output_mask = np.digitize(rands, self.bins) + 1
        else:
            classes = set(np.unique(data)) - self.ignore_values
            output_mask = np.zeros_like(data)

            for class_id in classes:
                if self.split_type == TrainTestSplitType.PER_VALUE:
                    np.random.seed(class_id)

                fold = np.digitize(np.random.rand(), self.bins) + 1
                output_mask[data == class_id] = fold

        eopatch[ftype][new_name] = output_mask

        return eopatch
