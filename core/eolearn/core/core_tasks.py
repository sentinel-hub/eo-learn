"""
A collection of most basic EOTasks
"""

from .eotask import EOTask, FeatureTask


class CopyTask(EOTask):
    """ A task that makes a copy of given EOPatch. This is not a deep copy, therefore it will make a copy of feature
    type dictionaries and not the data itself.

    :param feature_list: A list of features or feature types that will be copied into new EOPatch. If None, all
    features will be copied.

    Example: feature_list=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

    :type feature_list: list((FeatureType, str)) or None
    """
    def __init__(self, feature_list=None):
        self.feature_list = feature_list

    def execute(self, eopatch):
        return eopatch.__copy__(self.feature_list)


class DeepCopyTask(CopyTask):
    """ A task that makes a deep copy of given EOPatch.

    :param feature_list: A list of features or feature types that will be copied into new EOPatch. If None, all
    features will be copied.

    Example: feature_list=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

    :type feature_list: list((FeatureType, str)) or None
    """
    def __init__(self, **kwargs):
        super(DeepCopyTask, self).__init__(**kwargs)

    def execute(self, eopatch):
        return eopatch.__deepcopy__(self.feature_list)


class AddFeature(FeatureTask):
    """
    Adds a feature to given EOPatch.

    :param feature_type: Type of the feature to be added.
    :type feature_type: FeatureType
    :param feature_name: Name of the feature to be added.
    :type feature_name: str
    """
    def __init__(self, feature_type, feature_name):
        super(AddFeature, self).__init__(feature_type, feature_name)

    def execute(self, eopatch, data):
        """Adds the feature and returns the EOPatch.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param data: data to be added to the feature
        :type data: object
        :return: input EOPatch with the specified feature
        :rtype: EOPatch
        """
        eopatch[self.feature_type][self.feature_name] = data

        return eopatch


class RemoveFeature(FeatureTask):
    """
    Removes a feature from given EOPatch.

    :param feature_type: Type of the feature to be removed.
    :type feature_type: FeatureType
    :param feature_name: Name of the feature to be removed.
    :type feature_name: str
    """
    def __init__(self, feature_type, feature_name):
        super(RemoveFeature, self).__init__(feature_type, feature_name)

    def execute(self, eopatch):
        """ Removes the feature and returns the EOPatch.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: input EOPatch without the specified feature
        :rtype: EOPatch
        """
        eopatch.remove_feature(self.feature_type, self.feature_name)

        return eopatch
