"""
A collection of bands extraction EOTasks
"""

import numpy as np

from eolearn.core import EOTask

class EuclideanNormTask(EOTask):
    """ The task calculates the Euclidean Norm:

        :math:`\sqrt{\sum_{i} B_i^2}`,

    where :math:`B_i` are the individual bands within a user-specified feature array.
    """

    def __init__(self, input_feature, output_feature, bands=None):
        """
        :param input_feature: A source feature from which to take the subset of bands.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature to which to write the euclidean norm.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param bands: A list of bands from which to extract the euclidean norm. If None, all bands are taken.
        :type bands: list
        """
        self.input_feature = next(self._parse_features(input_feature)())
        self.output_feature = next(self._parse_features(output_feature)())
        self.bands = bands

    def execute(self, eopatch):
        """
        :param eopatch: An eopatch on which to calculate the euclidean norm.
        :type eopatch: EOPatch
        """
        feature_data = eopatch[self.input_feature]
        feature_data = feature_data if not self.bands else feature_data[..., self.bands]

        norm = np.sqrt(np.sum(feature_data**2, axis=-1))
        eopatch[self.output_feature] = norm[..., np.newaxis]

        return eopatch
