"""
A collection of most basic EOTasks
"""

import os.path

from .eodata import EOPatch
from .eotask import EOTask


class CopyTask(EOTask):
    """ A task that makes a copy of given EOPatch. This is not a deep copy, therefore it will make a copy of feature
    type dictionaries and not the data itself.

    :param features: A collection of features or feature types that will be copied into new EOPatch. By default all
    features will be copied.

    Example:
        features={
            FeatureType.Data: {'TRUE-COLOR'},
            FeatureType.MASK: {'CLOUD-MASK'},
            FeatureType.LABEL: ...
        }
        or
        features=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

    :type features: dict(FeatureType: set(str)) or list((FeatureType, str) or FeatureType) or ...
    """
    def __init__(self, features=...):
        self.features = features

    def execute(self, eopatch):
        return eopatch.__copy__(self.features)


class DeepCopyTask(CopyTask):
    """ A task that makes a deep copy of given EOPatch.

    :param features: A collection of features or feature types that will be copied into new EOPatch. By default all
    features will be copied.

    Example:
        features={
            FeatureType.Data: {'TRUE-COLOR'},
            FeatureType.MASK: {'CLOUD-MASK'},
            FeatureType.LABEL: ...
        }
        or
        features=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

    :type features: dict(FeatureType: set(str)) or list((FeatureType, str) or FeatureType) or ...
    """

    def execute(self, eopatch):
        return eopatch.__deepcopy__(self.features)


class SaveToDisk(EOTask):
    """ Saves EOPatch to disk.

    :param folder: root directory where all EOPatches are saved
    :type folder: str
    """

    def __init__(self, folder, *args, **kwargs):
        self.folder = folder
        self.args = args
        self.kwargs = kwargs

    def execute(self, eopatch, *, eopatch_folder):
        """ Saves the EOPatch to disk: `folder/eopatch_folder`.

        :param eopatch: EOPatch which will be saved
        :type eopatch: EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        :return: The same EOPatch
        :rtype: EOPatch
        """
        eopatch.save(os.path.join(self.folder, eopatch_folder), *self.args, **self.kwargs)
        return eopatch


class LoadFromDisk(EOTask):
    """ Loads EOPatch from disk.

    :param folder: root directory where all EOPatches are saved
    :type folder: str
    """
    def __init__(self, folder, *args, **kwargs):
        self.folder = folder
        self.args = args
        self.kwargs = kwargs

    def execute(self, *, eopatch_folder):
        """ Loads the EOPatch from disk: `folder/eopatch_folder`.

        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        :return: EOPatch loaded from disk
        :rtype: EOPatch
        """
        eopatch = EOPatch.load(os.path.join(self.folder, eopatch_folder), *self.args, **self.kwargs)
        return eopatch


class AddFeature(EOTask):
    """
    Adds a feature to given EOPatch.

    :param feature: Feature to be added
    :type feature: (FeatureType, feature_name) or FeatureType
    """
    def __init__(self, feature):
        self.feature_type, self.feature_name = list(self._parse_features(feature))[0]

    def execute(self, eopatch, data):
        """Adds the feature and returns the EOPatch.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param data: data to be added to the feature
        :type data: object
        :return: input EOPatch with the specified feature
        :rtype: EOPatch
        """
        if self.feature_name is None:
            eopatch[self.feature_type] = data
        else:
            eopatch[self.feature_type][self.feature_name] = data

        return eopatch


class RemoveFeature(EOTask):
    """
    Removes one or multiple features from given EOPatch.

    :param features: A collection of features to be removed
    :type features: dict(FeatureType: set(str)) or list((FeatureType, str)) or (FeatureType, str)
    """
    def __init__(self, features):
        self.feature_gen = self._parse_features(features)

    def execute(self, eopatch):
        """ Removes feature and returns the EOPatch.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: input EOPatch without the specified feature
        :rtype: EOPatch
        """
        for feature_type, feature_name in self.feature_gen(eopatch):
            if feature_name is ...:
                eopatch.reset_feature_type(feature_type)
            else:
                del eopatch[feature_type][feature_name]

        return eopatch


class RenameFeature(EOTask):
    """
    Renames one or multiple features from given EOPatch.

    :param features: A collection of features to be renamed

    Example:
        features=(FeatureType.DATA, 'name', 'new_name')
    or
        features={
            FeatureType.DATA: {
                'name1': 'new_name1',
                'name2': 'new_name2',
            },
            FeatureType.MASK: {
                'name1': 'new_name1',
                'name2': 'new_name2',
                'name3': 'new_name3',
            },
        }

    :type features: dict(FeatureType: set(str)) or list((FeatureType, str)) or (FeatureType, str)
    """
    def __init__(self, features):
        self.feature_gen = self._parse_features(features, new_names=True)

    def execute(self, eopatch):
        """ Renames features and returns the EOPatch.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: input EOPatch without the specified feature
        :rtype: EOPatch
        """
        for feature_type, feature_name, new_feature_name in self.feature_gen(eopatch):
            eopatch[feature_type][new_feature_name] = eopatch[feature_type][feature_name]
            del eopatch[feature_type][feature_name]

        return eopatch
