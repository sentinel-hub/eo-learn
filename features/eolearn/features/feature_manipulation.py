"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.
"""

from eolearn.core import EOTask, FeatureType
import numpy as np

import logging
LOGGER = logging.getLogger(__name__)


class RemoveFeature(EOTask):
    """
    Removes a feature from existing EOPatch.

    :param feature_type: Type of the feature to be removed.
    :type feature_type: FeatureType
    :param feature_name: Name (key) of the feature to be removed.
    :type feature_name: str
    """

    def __init__(self, feature_type, feature_name):
        self.feature_type = feature_type
        self.feature_name = feature_name

    def execute(self, eopatch):
        """ Removes the feature and returns the EOPatch.

        :param eopatch: input EOPatch
        :type eopatch: eolearn.core.EOPatch
        :return: input EOPatch without the specified feature
        :rtype: eolearn.core.EOPatch
        """
        eopatch.remove_feature(self.feature_type, self.feature_name)

        return eopatch


class SimpleFilterTask(EOTask):
    """
    Transforms an eopatch of shape [n, w, h, d] into [m, w, h, d] for m <= n. It removes all slices which don't
    conform to the predicate.

    A predicate is a callable which takes an numpy array and returns a bool.
    """
    def __init__(self, predicate, attr_type, field):
        """
        :param predicate: A callable that takes an eopatch instance and evaluates to bool.
        :param attr_type: A constant from FeatureType
        :param field: The name of the field in the dictionary corresponding to attr_type
        """
        self.predicate = predicate
        self.attr_type = attr_type
        self.field = field

    def execute(self, eopatch):
        """
        :param eopatch: Input eopatch.
        :type eopatch: EOPatch
        :return: Transformed eo patch
        :rtype: EOPatch
        """
        return self.do_transform(eopatch, self.attr_type.value, self.field)

    def do_transform(self, eopatch, attr_name, field):
        good_idxs = [idx for idx, img in enumerate(eopatch[attr_name][field]) if self.predicate(img)]

        LOGGER.debug("good_idxs: %s", good_idxs)

        time_dependent = [FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL]

        eopatch.timestamp = [eopatch.timestamp[idx] for idx in good_idxs]

        for attr_type in time_dependent:
            attr = getattr(eopatch, attr_type.value)
            assert isinstance(attr, dict)
            for target_field in attr:
                value = attr[target_field]
                eopatch.add_feature(attr_type=attr_type, field=target_field,
                                    value=np.asarray([value[idx] for idx in good_idxs]))

        return eopatch
