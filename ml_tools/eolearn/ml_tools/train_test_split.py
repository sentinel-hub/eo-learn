"""
Tasks used for train set preparation.
"""
import numpy as np

from eolearn.core import EOTask


class TrainTestSplitTask(EOTask):
    """ Randomly assign each or group of pixels to a train, test, and/or validation sets.

    The group of pixels is defined by an input feature (i.e. MASK_TIMELESS with polygon ids, or connected component id,
    or similar that groups pixels with similar properties). By default pixels get assigned to a subset 'per_class'. If
    split_type is set to 'random', then pixels get randomly assigned to a subset, regardless of their class.

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

    def __init__(self, feature, bins, split_type='per_class', no_data_value=None):
        """
        :param feature: The input feature out of which to generate the train mask.
        :type feature: (FeatureType, feature_name, new_name)
        :param bins: Cumulative probabilities of all value classes or a single float, representing a fraction.
        :type bins: a float or list of floats
        :param split_type: Valye split type, either 'per_class' or 'random'.
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

        if split_type not in ['per_class', 'random']:
            raise ValueError('Invalid split type.')

        self.bins = bins
        self.no_data_value = no_data_value
        self.split_type = split_type

    def execute(self, eopatch, *, seed=None):
        """
        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param seed: An argument to be passed to numpy.random.seed function.
        :type seed: numpy.int64
        :return: Input EOPatch with the train set mask.
        :rtype: EOPatch
        """
        np.random.seed(seed)

        ftype, fname, new_name = self.feature
        data = eopatch[(ftype, fname)]

        if self.split_type == 'per_class':
            classes = set(np.unique(data)) - {self.no_data_value}

            class_masks = (data == class_mask for class_mask in classes)
            if self.no_data_value:
                class_masks = (class_mask & (data != self.no_data_value) for class_mask in class_masks)

            rands = np.random.rand(len(classes))
            split = np.digitize(rands, self.bins) + 1
            output_mask = np.zeros_like(data)

            for class_mask, split_class in zip(class_masks, split):
                output_mask[class_mask] = split_class

        elif self.split_type == 'random':
            rands = np.random.rand(*data.shape)
            output_mask = np.digitize(rands, self.bins) + 1

        eopatch[ftype][new_name] = output_mask

        return eopatch
