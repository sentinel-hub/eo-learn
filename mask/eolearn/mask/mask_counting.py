"""
Module for generating count masks
"""

import numpy as np

from eolearn.core import MapFeatureTask


class CountValidTask(MapFeatureTask):
    """ Counts valid (non-zero) data through the temporal dimension.
    """
    def __init__(self, input_feature, output_feature):
        """
        :param input_feature: A source feature from which to read the values.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature to which to write the counts.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        super().__init__(input_feature, output_feature)

    def map_method(self, feature):
        """ Map method being applied to the feature that counts the valid data.
        """
        return np.count_nonzero(feature, axis=0)


class ClassFrequencyTask(MapFeatureTask):
    """ Calculates frequencies of each provided class through the temporal dimension.
    """
    def __init__(self, input_feature, output_feature, classes):
        """
        :param input_feature: A source feature from which to read the values.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature to which to write the frequencies.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param classes: Classes of which frequencies to calculate.
        :type classes: a list of integers
        """
        super().__init__(input_feature, output_feature)

        if not isinstance(classes, list) or not all((isinstance(x, int) for x in classes)):
            raise ValueError('classes argument should be a list of integers.')

        self.classes = classes

    def map_method(self, feature):
        """ Map method being applied to the feature that calculates the frequencies.
        """
        count_valid = np.count_nonzero(feature, axis=0)

        class_counts = (np.count_nonzero(feature == scl, axis=0) for scl in self.classes)

        with np.errstate(invalid='ignore'):
            class_counts = [np.divide(count, count_valid, dtype=np.float32) for count in class_counts]

        class_counts = np.concatenate(class_counts, axis=-1)

        class_counts[class_counts == np.inf] = np.nan

        return class_counts
