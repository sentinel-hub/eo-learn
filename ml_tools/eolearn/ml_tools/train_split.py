"""
Tasks used for train set preparation.
"""
import warnings
import numpy as np

from eolearn.core import EOTask


class TrainSplitTask(EOTask):
    """ Randomly assigns each value to a class of values by generating a class mask. Classes are defined by a list of
    cumulative probabilities, passed as the *bins* argument, the same way as the *bins* argument in
    `numpy.digitize <https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html>`_.
    """

    def __init__(self, feature, bins, no_data_value=None):
        """
        :param feature: The input feature out of which to generate the train mask.
        :type feature: (FeatureType, feature_name, new_name)
        :param bins: Cumulative probabilities of all value classes or a single float, representing a fraction.
        :type bins: a float or list of floats
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
        data = np.copy(eopatch[(ftype, fname)])

        classes = set(np.unique(data)) - {self.no_data_value}
        rands = np.random.rand(len(classes))
        split = np.digitize(rands, self.bins)

        if self.no_data_value in split:
            warnings.warn('TrainSplitTask: no_data_value found in split_classes.')

        for value_class, split_class in zip(classes, split):
            data[data == value_class] = split_class

        eopatch[ftype][new_name] = data

        return eopatch
