"""
A collection of most basic EOTasks
"""

from .eotask import EOTask


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
        super(CopyTask, self).__init__(**kwargs)

    def execute(self, eopatch):
        return eopatch.__deepcopy__(self.feature_list)
