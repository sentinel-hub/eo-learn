"""
Tasks used for train set preparation.
"""
from enum import Enum
import numpy as np

from eolearn.core import EOTask


class TrainTestSplitType(Enum):
    """ An enum defining TrainTestSplitTask's methods of splitting the data into subsets
    """
    PER_PIXEL = 'per_pixel'
    PER_CLASS = 'per_class'
    PER_VALUE = 'per_value'


class TrainTestSplitTask(EOTask):
    """ Randomly assign each or group of pixels to a train, test, and/or validation sets.

    The group of pixels is defined by an input feature (i.e. MASK_TIMELESS with polygon ids, or connected component id,
    or similar that groups pixels with similar properties). By default pixels get assigned to a subset
    :attr:`PER_CLASS<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_CLASS>`, where pixels of the same class
    will be assigned to the same subset. On the other hand if split_type is set to
    :attr:`PER_PIXEL<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_PIXEL>`, each pixel gets randomly
    assigned to a subset regardless of its class. The third option is splitting
    :attr:`PER_VALUE<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_VALUE>`, where each class is assigned to
    a certain subset consistently across eopatches. In other words, if a polyogn lies on multiple eopatches, it will
    still be assigned to the same subset in all eopatches. In this case, the *seed* argument of the *execute* method
    gets ignored.

    Classes are defined by a list of cumulative probabilities, passed as the *bins* argument, the same way as the *bins*
    argument in `numpy.digitize <https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html>`_. Valid
    classes are enumerated from 1 onward and if no_data_value is provided, all values equal to it get assigned to class
    0.

    To get a train/test split as 80/20, bins argument should be provided as `bins=[0.8]`.

    To get a train/val/test split as 60/20/20, bins argument should be provided as `bins=[0.6, 0.8]`.

    Splits can also be made into as many subsets as desired, e.g.: `bins=[0.1, 0.2, 0.3, 0.7, 0.9]`

    After the execution of this task an EOPatch will have a new (FeatureType, new_name) feature where each pixel will
    have a value representing the train, test and/or validation set.
    """

    def __init__(self, feature, bins, split_type=TrainTestSplitType.PER_CLASS, no_data_value=None):
        """
        :param feature: The input feature out of which to generate the train mask.
        :type feature: (FeatureType, feature_name, new_name)
        :param bins: Cumulative probabilities of all value classes or a single float, representing a fraction.
        :type bins: a float or list of floats
        :param split_type: Valye split type, either 'per_class' or 'per_pixel'.
        :type split_type: str
        :param no_data_value: A value to ignore and not assign it to any subsets.
        :type no_data_value: any numpy.dtype
        """
        self.feature = next(self._parse_features(feature, new_names=True)())

        if np.isscalar(bins):
            bins = [bins]

        if not isinstance(bins, list) or not all(isinstance(bi, float) for bi in bins) \
                or np.any(np.diff(bins) <= 0) or bins[0] <= 0 or bins[-1] >= 1:
            raise ValueError('bins argument should be a list of ascending floats inside an open interval (0, 1)')

        self.bins = bins
        self.no_data_value = no_data_value
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
            classes = set(np.unique(data)) - {self.no_data_value}
            output_mask = np.zeros_like(data)

            for class_id in classes:
                if self.split_type == TrainTestSplitType.PER_VALUE:
                    np.random.seed(class_id)

                fold = np.digitize(np.random.rand(), self.bins) + 1
                output_mask[data == class_id] = fold

        eopatch[ftype][new_name] = output_mask

        return eopatch
