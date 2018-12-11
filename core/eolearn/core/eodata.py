"""
The eodata module provides core objects for handling remotely sensing multi-temporal data (such as satellite imagery).
"""

import os
import logging
import pickle
import numpy as np
import gzip
import shutil
import warnings
import attr
import datetime
import dateutil.parser

from copy import copy, deepcopy
from geopandas import GeoDataFrame, GeoSeries

from sentinelhub import BBox

from .constants import FeatureType, FileFormat, OverwritePermission
from .utilities import deep_eq, FeatureParser


LOGGER = logging.getLogger(__name__)

MAX_DATA_REPR_LEN = 100


@attr.s(repr=False, cmp=False, kw_only=True)
class EOPatch:
    """The basic data object for multi-temporal remotely sensed data, such as satellite imagery and its derivatives.

    The EOPatch contains multi-temporal remotely sensed data of a single patch of earth's surface defined by the
    bounding box in specific coordinate reference system. The patch can be a rectangle, polygon, or pixel in space.
    The EOPatch object can also be used to store derived quantities, such as for example means, standard deviations,
    etc., of a patch. In this case the 'space' dimension is equivalent to a pixel.

    Primary goal of EOPatch is to store remotely sensed data, usually of a shape n_time x height x width x n_features
    images, where height and width are the numbers of pixels in y and x, n_features is the number of features
    (i.e. bands/channels, cloud probability, etc.), and n_time is the number of time-slices (the number of times this
    patch was recorded by the satellite; can also be a single image)

    In addition to that other auxiliary information is also needed and can be stored in additional attributes of the
    EOPatch (thus extending the functionality of numpy ndarray). These attributes are listed in the FeatureType enum.

    Currently the EOPatch object doesn't enforce that the length of timestamp be equal to n_times dimensions of numpy
    arrays in other attributes.
    """
    data = attr.ib(factory=dict)
    mask = attr.ib(factory=dict)
    scalar = attr.ib(factory=dict)
    label = attr.ib(factory=dict)
    vector = attr.ib(factory=dict)
    data_timeless = attr.ib(factory=dict)
    mask_timeless = attr.ib(factory=dict)
    scalar_timeless = attr.ib(factory=dict)
    label_timeless = attr.ib(factory=dict)
    vector_timeless = attr.ib(factory=dict)
    meta_info = attr.ib(factory=dict)
    bbox = attr.ib(default=None)
    timestamp = attr.ib(factory=list)

    def __setattr__(self, key, value):
        """Raises TypeError if feature type attributes are not of correct type.

        In case they are a dictionary they are cast to _FeatureDict class
        """
        if FeatureType.has_value(key) and not isinstance(value, _FileLoader):
            feature_type = FeatureType(key)
            value = self._parse_feature_type_value(feature_type, value)

        super().__setattr__(key, value)

    @staticmethod
    def _parse_feature_type_value(feature_type, value):
        """ Checks or parses value which will be assigned to a feature type attribute of `EOPatch`. If the value
        cannot be parsed correctly it raises an error.

        :raises: TypeError, ValueError
        """
        if feature_type.has_dict() and isinstance(value, dict):
            return value if isinstance(value, _FeatureDict) else _FeatureDict(value, feature_type)

        if feature_type is FeatureType.BBOX:
            if value is None or isinstance(value, BBox):
                return value
            if isinstance(value, (tuple, list)) and len(value) == 5:
                return BBox(value[:4], crs=value[4])

        if feature_type is FeatureType.TIMESTAMP:
            if isinstance(value, (tuple, list)):
                return [timestamp if isinstance(timestamp, datetime.date) else dateutil.parser.parse(timestamp)
                        for timestamp in value]

        raise TypeError('Attribute {} requires value of type {} - '
                        'failed to parse given value'.format(feature_type, feature_type.type()))

    def __getattribute__(self, key, load=True):
        """ Handles lazy loading
        """
        value = super().__getattribute__(key)

        if isinstance(value, _FileLoader) and load:
            value = value.load()
            setattr(self, key, value)
            return getattr(self, key)

        return value

    def __getitem__(self, feature_type):
        """Provides features of requested feature type.

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str
        :return: Dictionary of features
        """
        return getattr(self, FeatureType(feature_type).value)

    def __setitem__(self, feature_type, value):
        """Sets a new dictionary / list to the given FeatureType.

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str
        :param value: New dictionary or list
        :type value: dict or list
        :return: Dictionary of features
        """
        return setattr(self, FeatureType(feature_type).value, value)

    def __eq__(self, other):
        """True if FeatureType attributes, bbox, and timestamps of both EOPatches are equal by value."""
        if not isinstance(self, type(other)):
            return False

        for feature_type in FeatureType:
            if not deep_eq(self[feature_type], other[feature_type]):
                return False
        return True

    def __add__(self, other):
        """Concatenates two EOPatches into a new EOPatch."""
        return EOPatch.concatenate(self, other)

    def __repr__(self):
        feature_repr_list = ['{}('.format(self.__class__.__name__)]
        for feature_type in FeatureType:
            content = self[feature_type]

            if isinstance(content, dict) and content:
                content_str = '\n    '.join(['{'] + ['{}: {}'.format(label, self._repr_value(value)) for label, value in
                                                     sorted(content.items())]) + '\n  }'
            else:
                content_str = self._repr_value(content)
            feature_repr_list.append('{}: {}'.format(feature_type.value, content_str))

        return '\n  '.join(feature_repr_list) + '\n)'

    @staticmethod
    def _repr_value(value):
        """Creates a representation string for different types of data.

        :param value: data in any type
        :return: representation string
        :rtype: str
        """
        if isinstance(value, np.ndarray):
            return '{}(shape={}, dtype={})'.format(EOPatch._repr_value_class(value), value.shape, value.dtype)
        if isinstance(value, GeoDataFrame):
            return '{}(columns={}, length={}, crs={})'.format(EOPatch._repr_value_class(value), list(value),
                                                              len(value), value.crs['init'])
        if isinstance(value, (list, tuple, dict)) and value:
            repr_str = str(value)
            if len(repr_str) <= MAX_DATA_REPR_LEN:
                return repr_str

            bracket_str = '[{}]' if isinstance(value, list) else '({})'
            if isinstance(value, (list, tuple)) and len(value) > 2:
                repr_str = bracket_str.format('{}, ..., {}'.format(repr(value[0]), repr(value[-1])))

            if len(repr_str) > MAX_DATA_REPR_LEN and isinstance(value, (list, tuple)) and len(value) > 1:
                repr_str = bracket_str.format('{}, ...'.format(repr(value[0])))

            if len(repr_str) > MAX_DATA_REPR_LEN:
                repr_str = str(type(value))

            return '{}, length={}'.format(repr_str, len(value))

        return repr(value)

    @staticmethod
    def _repr_value_class(value):
        """ A representation of a class of a given value
        """
        cls = value.__class__
        return '.'.join([cls.__module__.split('.')[0], cls.__name__])

    def __copy__(self, features=...):
        """Returns a new EOPatch with shallow copies of given features.

        :param features: A collection of features or feature types that will be copied into new EOPatch.
        :type features: object supported by eolearn.core.utilities.FeatureParser class
        """
        if not features:  # For some reason deepcopy and copy pass {} by default
            features = ...

        new_eopatch = EOPatch()
        for feature_type, feature_name in FeatureParser(features)(self):
            if feature_name is ...:
                new_eopatch[feature_type] = copy(self[feature_type])
            else:
                new_eopatch[feature_type][feature_name] = self[feature_type][feature_name]
        return new_eopatch

    def __deepcopy__(self, memo=None, features=...):
        """Returns a new EOPatch with deep copies of given features.

        :param memo: built-in parameter for memoization
        :type memo: dict
        :param features: A collection of features or feature types that will be copied into new EOPatch.
        :type features: object supported by eolearn.core.utilities.FeatureParser class
        """
        if not features:  # For some reason deepcopy and copy pass {} by default
            features = ...

        new_eopatch = self.__copy__(features=features)
        for feature_type in FeatureType:
            new_eopatch[feature_type] = deepcopy(new_eopatch[feature_type], memo)

        return new_eopatch

    def remove_feature(self, feature_type, feature_name):
        """Removes the feature ``feature_name`` from dictionary of ``feature_type``.

        :param feature_type: Enum of the attribute we're about to modify
        :type feature_type: FeatureType
        :param feature_name: Name of the feature of the attribute
        :type feature_name: str
        """
        LOGGER.debug("Removing feature '%s' from attribute '%s'", feature_name, feature_type.value)

        self._check_if_dict(feature_type)
        if feature_name in self[feature_type]:
            del self[feature_type][feature_name]

    def add_feature(self, feature_type, feature_name, value):
        """Sets EOPatch[feature_type][feature_name] to the given value.

        :param feature_type: Type of feature
        :type feature_type: FeatureType
        :param feature_name: Name of the feature
        :type feature_name: str
        :param value: New value of the feature
        :type value: object
        """
        self._check_if_dict(feature_type)
        self[feature_type][feature_name] = value

    @staticmethod
    def _check_if_dict(feature_type):
        """Checks if the given feature type contains a dictionary and raises an error if it doesn't.

        :param feature_type: Type of feature
        :type feature_type: FeatureType
        :raise: TypeError
        """
        feature_type = FeatureType(feature_type)
        if feature_type.type() is not dict:
            raise TypeError('{} does not contain a dictionary of features'.format(feature_type))

    def reset_feature_type(self, feature_type):
        """Resets the values of the given feature type.

        :param feature_type: Type of a feature
        :type feature_type: FeatureType
        """
        feature_type = FeatureType(feature_type)
        if feature_type.has_dict():
            self[feature_type] = {}
        elif feature_type is FeatureType.BBOX:
            self[feature_type] = None
        else:
            self[feature_type] = []

    def set_bbox(self, new_bbox):
        self.bbox = new_bbox

    def set_timestamp(self, new_timestamp):
        """
        :param new_timestamp: list of dates
        :type new_timestamp: list(str)
        """
        self.timestamp = new_timestamp

    def get_feature(self, feature_type, feature_name=None):
        """Returns the array of corresponding feature.

        :param feature_type: Enum of the attribute
        :type feature_type: FeatureType
        :param feature_name: Name of the feature
        :type feature_name: str
        """
        if feature_name is None:
            return self[feature_type]
        return self[feature_type][feature_name]

    def get_features(self):
        """Returns a dictionary of all non-empty features of EOPatch.

        The elements are either sets of feature names or a boolean `True` in case feature type has no dictionary of
        feature names.

        :return: A dictionary of features
        :rtype: dict(FeatureType: str or True)
        """
        feature_dict = {}
        for feature_type in FeatureType:
            if self[feature_type]:
                feature_dict[feature_type] = set(self[feature_type]) if feature_type.has_dict() else True

        return feature_dict

    def get_spatial_dimension(self, feature_type, feature_name):
        """
        Returns a tuple of spatial dimension (height, width) of a feature.

        The feature has to be spatial or time dependent.

        :param feature_type: Enum of the attribute
        :type feature_type: FeatureType
        :param feature_name: Name of the feature
        :type feature_name: str
        """
        if feature_type.is_time_dependent() or feature_type.is_spatial():
            shape = self[feature_type][feature_name].shape
            return shape[1:3] if feature_type.is_time_dependent() else shape[0:2]

        raise ValueError('FeatureType used to determine the width and height of raster must be'
                         ' time dependent or spatial.')

    def get_feature_list(self):
        """Returns a list of all non-empty features of EOPatch.

        The elements are either only FeatureType or a pair of FeatureType and feature name.

        :return: list of features
        :rtype: list(FeatureType or (FeatureType, str))
        """
        feature_list = []
        for feature_type in FeatureType:
            if feature_type.has_dict():
                for feature_name in self[feature_type]:
                    feature_list.append((feature_type, feature_name))
            elif self[feature_type]:
                feature_list.append(feature_type)
        return feature_list

    @staticmethod
    def concatenate(eopatch1, eopatch2):
        """Joins all data from two EOPatches and returns a new EOPatch.

        If timestamps don't match it will try to join all time-dependent features with the same name.

        Note: In general the data won't be deep copied. Deep copy will only happen when merging time-dependent features
        along time

        :param eopatch1: First EOPatch
        :type eopatch1: EOPatch
        :param eopatch2: First EOPatch
        :type eopatch2: EOPatch
        :return: Joined EOPatch
        :rtype: EOPatch
        """
        eopatch_content = {}

        timestamps_exist = eopatch1.timestamp and eopatch2.timestamp
        timestamps_match = timestamps_exist and deep_eq(eopatch1.timestamp, eopatch2.timestamp)

        # if not timestamps_match and timestamps_exist and eopatch1.timestamp[-1] >= eopatch2.timestamp[0]:
        #     raise ValueError('Could not merge timestamps because any timestamp of the first EOPatch must be before '
        #                      'any timestamp of the second EOPatch')

        for feature_type in FeatureType:
            if feature_type.has_dict():
                eopatch_content[feature_type.value] = {**eopatch1[feature_type], **eopatch2[feature_type]}

                for feature_name in eopatch1[feature_type].keys() & eopatch2[feature_type].keys():
                    data1 = eopatch1[feature_type][feature_name]
                    data2 = eopatch2[feature_type][feature_name]

                    if feature_type.is_time_dependent() and not timestamps_match:
                        eopatch_content[feature_type.value][feature_name] = EOPatch.concatenate_data(data1, data2)
                    elif not deep_eq(data1, data2):
                        raise ValueError('Could not merge ({}, {}) feature because values differ'.format(feature_type,
                                                                                                         feature_name))

            elif feature_type is FeatureType.TIMESTAMP and timestamps_exist and not timestamps_match:
                eopatch_content[feature_type.value] = eopatch1[feature_type] + eopatch2[feature_type]
            else:
                if not eopatch1[feature_type] or deep_eq(eopatch1[feature_type], eopatch2[feature_type]):
                    eopatch_content[feature_type.value] = copy(eopatch2[feature_type])
                elif not eopatch2[feature_type]:
                    eopatch_content[feature_type.value] = copy(eopatch1[feature_type])
                else:
                    raise ValueError('Could not merge {} feature because values differ'.format(feature_type))

        return EOPatch(**eopatch_content)

    @staticmethod
    def concatenate_data(data1, data2):
        """A method that concatenates two numpy array along first axis.

        :param data1: Numpy array of shape (times1, height, width, n_features)
        :type data1: numpy.ndarray
        :param data2: Numpy array of shape (times2, height, width, n_features)
        :type data1: numpy.ndarray
        :return: Numpy array of shape (times1 + times2, height, width, n_features)
        :rtype: numpy.ndarray
        """
        if data1.shape[1:] != data2.shape[1:]:
            raise ValueError('Could not concatenate data because non-temporal dimensions do not match')
        return np.concatenate((data1, data2), axis=0)

    def save(self, path, features=..., file_format=FileFormat.NPY,
             overwrite_permission=OverwritePermission.ADD_ONLY, compress_level=0):
        """Saves EOPatch to disk.

        :param path: Location on the disk
        :type path: str
        :param features: A collection of features types specifying features of which type will be saved. By default
        all features will be saved.
        :type features: list(FeatureType) or list((FeatureType, str)) or ...
        :param file_format: File format
        :type file_format: FileFormat or str
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified with an integer from 0 (no compression)
            to 9 (highest compression).
        :type compress_level: int
        """
        if os.path.isfile(path):
            raise NotADirectoryError("A file exists at the given path, expected a directory")

        file_format = FileFormat(file_format)
        if file_format is FileFormat.GZIP:
            raise ValueError('file_format cannot be {}, compression is specified with compression_level '
                             'parameter'.format(FileFormat.GZIP))

        overwrite_permission = OverwritePermission(overwrite_permission)

        tmp_path = '{}_tmp_{}'.format(path, datetime.datetime.now().timestamp())
        if os.path.exists(tmp_path):  # Basically impossible case
            raise OSError('Path {} already exists, try again'.format(tmp_path))

        save_file_list = self._get_save_file_list(path, tmp_path, features, file_format, compress_level)

        self._check_forbidden_characters(save_file_list)

        existing_content = self._get_eopatch_content(path) if \
            os.path.exists(path) and overwrite_permission is not OverwritePermission.OVERWRITE_PATCH else {}

        self._check_feature_case_matching(save_file_list, existing_content)

        if overwrite_permission is OverwritePermission.ADD_ONLY and os.path.exists(path):
            self._check_feature_uniqueness(save_file_list, existing_content)

        try:
            for file_saver in save_file_list:
                file_saver.save(self)

            if os.path.exists(path):
                if overwrite_permission is OverwritePermission.OVERWRITE_PATCH:
                    shutil.rmtree(path)
                    os.renames(tmp_path, path)
                else:
                    for file_saver in save_file_list:
                        existing_features = existing_content.get(file_saver.feature_type.value, {})
                        if file_saver.feature_name is None and isinstance(existing_features, _FileLoader):
                            os.remove(existing_features.get_file_path())
                        elif isinstance(existing_features, dict) and file_saver.feature_name in existing_features:
                            os.remove(existing_features[file_saver.feature_name].get_file_path())
                        os.renames(file_saver.tmp_filename, file_saver.final_filename)
                    if os.path.exists(tmp_path):
                        shutil.rmtree(tmp_path)
            else:
                os.renames(tmp_path, path)

        except BaseException as ex:
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            raise ex

    def _get_save_file_list(self, path, tmp_path, features, file_format, compress_level):
        """ Creates a list of _FileSaver classes for each feature which will have to be saved
        """
        save_file_list = []
        saved_feature_types = set()
        for feature_type, feature_name in FeatureParser(features)(self):
            if not self[feature_type]:
                continue
            if not feature_type.is_meta() or feature_type not in saved_feature_types:
                save_file_list.append(_FileSaver(path, tmp_path, feature_type,
                                                 None if feature_type.is_meta() else feature_name,
                                                 file_format if feature_type.contains_ndarrays() else FileFormat.PICKLE,
                                                 compress_level))
            saved_feature_types.add(feature_type)
        return save_file_list

    @staticmethod
    def _check_forbidden_characters(save_file_list):
        """ Checks if feature names have properties which might cause problems during saving or loading

        :param save_file_list: List of features which will be saved
        :type save_file_list: list(_FileSaver)
        :raises: ValueError
        """
        for file_saver in save_file_list:
            if file_saver.feature_name is None:
                continue
            for char in ['.', '/', '\\', '|', ';', ':', '\n', '\t']:
                if char in file_saver.feature_name:
                    raise ValueError("Cannot save feature ({}, {}) because feature name contains an illegal character "
                                     "'{}'. Please change the feature name".format(file_saver.feature_type,
                                                                                   file_saver.feature_name, char))
            if file_saver.feature_name == '':
                raise ValueError("Cannot save feature with empty string for a name. Please change the feature name")

    @staticmethod
    def _check_feature_case_matching(save_file_list, existing_content):
        """ This is required for Windows OS where file names cannot differ only in case size

        :raises: OSError
        """
        feature_collection = {feature_type: set() for feature_type in FeatureType}

        for feature_type_str, content in existing_content.items():
            feature_type = FeatureType(feature_type_str)
            if isinstance(content, dict):
                for feature_name in content:
                    feature_collection[feature_type].add(feature_name)

        for file_saver in save_file_list:
            if file_saver.feature_name is not None:
                feature_collection[file_saver.feature_type].add(file_saver.feature_name)

        for features in feature_collection.values():
            lowercase_features = {}
            for feature_name in features:
                lowercase_feature_name = feature_name.lower()

                if lowercase_feature_name in lowercase_features:
                    raise OSError("Features '{}' and '{}' differ only in casing and cannot be saved into separate "
                                  "files".format(feature_name, lowercase_features[lowercase_feature_name]))

                lowercase_features[lowercase_feature_name] = feature_name

    @staticmethod
    def _check_feature_uniqueness(save_file_list, existing_content):
        """ Check if any feature already exists in saved EOPatch

        :raises: ValueError
        """
        for file_saver in save_file_list:
            if file_saver.feature_type.value not in existing_content:
                continue
            content = existing_content[file_saver.feature_type.value]
            if file_saver.feature_name in content:
                file_path = content[file_saver.feature_name].get_file_path()
                alternative_permissions = tuple(op for op in OverwritePermission if
                                                op is not OverwritePermission.ADD_ONLY)
                raise ValueError("Feature ({}, {}) already exists in {}\n"
                                 "In order to overwrite it set 'overwrite_permission' parameter to one of the "
                                 "options {}".format(file_saver.feature_type, file_saver.feature_name, file_path,
                                                     alternative_permissions))

    @staticmethod
    def load(path, features=..., lazy_loading=False, mmap=False):
        """Loads EOPatch from disk.

        :param path: Location on the disk
        :type path: str
        :param features: A collection of features to be loaded. By default all features will be loaded.
        :type features: object
        :param lazy_loading: If `True` features will be lazy loaded.
        :type lazy_loading: bool
        :param mmap: If True, then memory-map the file. Works only on uncompressed npy files
        :type mmap: bool
        :return: Loaded EOPatch
        :rtype: EOPatch
        """
        if not os.path.exists(path):
            raise ValueError('Specified path {} does not exist'.format(path))

        entire_content = EOPatch._get_eopatch_content(path, mmap=mmap)
        requested_content = {}
        for feature_type, feature_name in FeatureParser(features):
            feature_type_str = feature_type.value
            if feature_type_str not in entire_content:
                continue

            content = entire_content[feature_type_str]
            if isinstance(content, _FileLoader) or (isinstance(content, dict) and feature_name is ...):
                requested_content[feature_type_str] = content
            else:
                if feature_type_str not in requested_content:
                    requested_content[feature_type_str] = {}
                requested_content[feature_type_str][feature_name] = content[feature_name]

        if not lazy_loading:
            for feature_type, content in requested_content.items():
                if isinstance(content, _FileLoader):
                    requested_content[feature_type] = content.load()
                elif isinstance(content, dict):
                    for feature_name, loader in content.items():
                        content[feature_name] = loader.load()

        return EOPatch(**requested_content)

    @staticmethod
    def _get_eopatch_content(path, mmap=False):
        """ Checks the content of saved EOPatch and creates a dictionary with _FileLoader classes

        :param path: Location on the disk
        :type path: str
        :param mmap: If True, then memory-map the file. Works only on uncompressed npy files
        :type mmap: bool
        :return: A dictionary describing content of existing EOPatch
        """
        eopatch_content = {}

        for feature_type_name in os.listdir(path):
            feature_type_path = os.path.join(path, feature_type_name)

            if os.path.isdir(feature_type_path):
                if not FeatureType.has_value(feature_type_name) or FeatureType(feature_type_name).is_meta():
                    warnings.warn('Folder {} is not recognized in EOPatch folder structure, will be skipped'.format(
                        feature_type_path))
                    continue
                if feature_type_name in eopatch_content:
                    warnings.warn('There are multiple files containing data about {}'.format(FeatureType(
                        feature_type_name)))
                    if not isinstance(eopatch_content[feature_type_name], dict):
                        eopatch_content[feature_type_name] = {}
                else:
                    eopatch_content[feature_type_name] = {}

                for feature in os.listdir(feature_type_path):
                    feature_path = os.path.join(feature_type_path, feature)
                    if os.path.isdir(feature_path):
                        warnings.warn(
                            'Folder {} is not recognized in EOPatch folder structure, will be skipped'.format(
                                feature_path))
                        continue
                    feature_name = FileFormat.split_by_extensions(feature)[0]
                    if feature_name in eopatch_content[feature_type_name]:
                        warnings.warn('There are multiple files containing data about ({}, {})'.format(
                            FeatureType(feature_type_name), feature_name))
                        continue

                    eopatch_content[feature_type_name][feature_name] = \
                        _FileLoader(path, os.path.join(feature_type_name, feature))
            else:
                feature_type_str = FileFormat.split_by_extensions(feature_type_name)[0]
                if not FeatureType.has_value(feature_type_str):
                    warnings.warn('File {} is not recognized in EOPatch folder structure, will be skipped'.format(
                        feature_type_path))
                elif feature_type_str in eopatch_content:
                    warnings.warn('There are multiple files containing data about {}'.format(
                        FeatureType(feature_type_str)))
                elif os.path.getsize(feature_type_path):
                    eopatch_content[feature_type_str] = _FileLoader(path, feature_type_name, mmap)

        return eopatch_content

    def time_series(self, ref_date=None, scale_time=1):
        """Returns a numpy array with seconds passed between the reference date and the timestamp of each image.

        An array is constructed as time_series[i] = (timestamp[i] - ref_date).total_seconds().
        If reference date is None the first date in the EOPatch's timestamp is taken.
        If EOPatch timestamp attribute is empty the method returns None.

        :param ref_date: reference date relative to which the time is measured
        :type ref_date: datetime object
        :param scale_time: scale seconds by factor. If `60`, time will be in minutes, if `3600` hours
        :type scale_time: int
        """

        if not self.timestamp:
            return None

        if ref_date is None:
            ref_date = self.timestamp[0]

        return np.asarray([round((timestamp - ref_date).total_seconds() / scale_time) for timestamp in self.timestamp],
                          dtype=np.int64)

    def consolidate_timestamps(self, timestamps):
        """Removes all frames from the EOPatch with a date not found in the provided timestamps list.

        :param timestamps: keep frames with date found in this list
        :type timestamps: list of datetime objects
        :return: set of removed frames' dates
        :rtype: set of datetime objects
        """
        remove_from_patch = set(self.timestamp).difference(timestamps)
        remove_from_patch_idxs = [self.timestamp.index(rm_date) for rm_date in remove_from_patch]
        good_timestamp_idxs = [idx for idx, _ in enumerate(self.timestamp) if idx not in remove_from_patch_idxs]
        good_timestamps = [date for idx, date in enumerate(self.timestamp) if idx not in remove_from_patch_idxs]

        for feature_type in [feature_type for feature_type in FeatureType if (feature_type.is_time_dependent() and
                                                                              feature_type.has_dict())]:

            for feature_name, value in self[feature_type].items():
                if isinstance(value, np.ndarray):
                    self[feature_type][feature_name] = value[good_timestamp_idxs, ...]
                if isinstance(value, list):
                    self[feature_type][feature_name] = [value[idx] for idx in good_timestamp_idxs]

        self.timestamp = good_timestamps
        return remove_from_patch


class _FeatureDict(dict):
    """A dictionary structure that holds features of certain feature type.

    It checks that features have a correct and dimension. It also supports lazy loading by accepting a function as a
    feature value, which is then called when the feature is accessed.

    :param feature_dict: A dictionary of feature names and values
    :type feature_dict: dict(str: object)
    :param feature_type: Type of features
    :type feature_type: FeatureType
    """
    def __init__(self, feature_dict, feature_type):
        super().__init__()

        self.feature_type = feature_type
        self.ndim = self.feature_type.ndim()
        self.is_vector = self.feature_type.is_vector()

        for feature_name, value in feature_dict.items():
            self[feature_name] = value

    def __setitem__(self, feature_name, value):
        """ Before setting value to the dictionary it checks that value is of correct type and dimension and tries to
        transform value in correct form.
        """
        value = self._parse_feature_value(value)
        super().__setitem__(feature_name, value)

    def __getitem__(self, feature_name, load=True):
        """Implements lazy loading."""
        value = super().__getitem__(feature_name)

        if isinstance(value, _FileLoader) and load:
            value = value.load()
            self[feature_name] = value

        return value

    def get_dict(self):
        """Returns a Python dictionary of features and value."""
        return dict(self)

    def _parse_feature_value(self, value):
        """ Checks if value fits the feature type. If not it tries to fix it or raise an error

        :raises: ValueError
        """
        if isinstance(value, _FileLoader):
            return value
        if not hasattr(self, 'ndim'):  # Because of serialization/deserialization during multiprocessing
            return value

        if self.ndim:
            if not isinstance(value, np.ndarray):
                raise ValueError('{} feature has to be a numpy array'.format(self.feature_type))
            if value.ndim != self.ndim:
                raise ValueError('Numpy array of {} feature has to have {} '
                                 'dimension{}'.format(self.feature_type, self.ndim, 's' if self.ndim > 1 else ''))

            if self.feature_type.is_discrete():
                if not issubclass(value.dtype.type, (np.integer, np.bool, np.bool_, np.bool8)):
                    msg = '{} is a discrete feature type therefore dtype of data should be a subtype of ' \
                          'numpy.integer or numpy.bool, found type {}. In the future an error will be raised because' \
                          'of this'.format(self.feature_type, value.dtype.type)
                    warnings.warn(msg, DeprecationWarning, stacklevel=3)

                    # raise ValueError('{} is a discrete feature type therefore dtype of data has to be a subtype of '
                    #                  'numpy.integer or numpy.bool, found type {}'.format(self.feature_type,
                    #                                                                      value.dtype.type))
            # This checking is disabled for now
            # else:
            #     if not issubclass(value.dtype.type, (np.floating, np.float)):
            #         raise ValueError('{} is a floating feature type therefore dtype of data has to be a subtype of '
            #                          'numpy.floating or numpy.float, found type {}'.format(self.feature_type,
            #                                                                                value.dtype.type))
            return value

        if self.is_vector:
            if isinstance(value, GeoSeries):
                value = GeoDataFrame(dict(geometry=value), crs=value.crs)

            if isinstance(value, GeoDataFrame):
                if self.feature_type is FeatureType.VECTOR:
                    if FeatureType.TIMESTAMP.value.upper() not in value:
                        raise ValueError("{} feature has to contain a column 'TIMESTAMP' with "
                                         "timestamps".format(self.feature_type))

                return value

            raise ValueError('{} feature works with data of type {}, parsing data type {} is not supported'
                             'given'.format(self.feature_type, GeoDataFrame.__name__, type(value)))

        return value


class _FileLoader:
    """ Class taking care for loading objects from disk. Its purpose is to support lazy loading
    """
    def __init__(self, patch_path, filename, mmap=False):
        """
        :param patch_path: Location of EOPatch on disk
        :type patch_path: str
        :param filename: Location of file inside the EOPatch, extension should be included
        :type filename: str
        :param mmap: In case of npy files the tile can be loaded as memory map
        :type mmap: bool
        """
        self.patch_path = patch_path
        self.filename = filename
        self.mmap = mmap

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.get_file_path())

    def set_new_patch_path(self, new_patch_path):
        """ Sets new patch location on disk
        """
        self.patch_path = new_patch_path

    def get_file_path(self):
        """ Returns file path from where feature will be loaded
        """
        return os.path.join(self.patch_path, self.filename)

    def load(self):
        """ Method which loads data from the file
        """
        if not os.path.isdir(self.patch_path):
            raise OSError('EOPatch does not exist in path {} anymore'.format(self.patch_path))

        path = self.get_file_path()
        if not os.path.exists(path):
            raise OSError('Feature in path {} does not exist anymore'.format(path))

        file_formats = FileFormat.split_by_extensions(path)[1:]

        if not file_formats or file_formats[-1] is FileFormat.PICKLE:
            with open(path, "rb") as infile:
                return pickle.load(infile)

        if file_formats[-1] is FileFormat.NPY:
            if self.mmap:
                return np.load(path, mmap_mode='r')
            return np.load(path)

        if file_formats[-1] is FileFormat.GZIP:
            if file_formats[-2] is FileFormat.NPY:
                return np.load(gzip.open(path))

            if len(file_formats) == 1 or file_formats[-2] is FileFormat.PICKLE:
                return pickle.load(gzip.open(path))

        raise ValueError('Could not load data from unsupported file format {}'.format(file_formats[-1]))


class _FileSaver:
    """ Class taking care for saving feature to disk
    """
    def __init__(self, path, tmp_path, feature_type, feature_name, file_format, compress_level):
        self.feature_type = feature_type
        self.feature_name = feature_name
        self.file_format = file_format
        self.compress_level = compress_level

        self.final_filename = self.get_file_path(path)
        self.tmp_filename = self.get_file_path(tmp_path)

    def get_file_path(self, path):
        """ Creates a filename with file path
        """
        feature_filename = self._get_filename_path(path)

        feature_filename += self.file_format.extension()
        if self.compress_level:
            feature_filename += FileFormat.GZIP.extension()

        return feature_filename

    def _get_filename_path(self, path):
        """ Helper function for creating filename without file extension
        """
        feature_filename = os.path.join(path, self.feature_type.value)

        if self.feature_name is not None:
            feature_filename = os.path.join(feature_filename, self.feature_name)

        return feature_filename

    def save(self, eopatch, use_tmp=True):
        """ Method which does the saving

        :param eopatch: EOPatch containing the data which will be saved
        :type eopatch: EOPatch
        :param use_tmp: If `True` data will be saved to temporary file, otherwise it will be saved to intended
        (i.e. final) location
        :type use_tmp: bool
        """
        filename = self.tmp_filename if use_tmp else self.final_filename

        if self.feature_name is None:
            data = eopatch[self.feature_type]
            if self.feature_type.has_dict():
                data = data.get_dict()

            if self.feature_type is FeatureType.BBOX:
                data = tuple(data) + (int(data.crs.value),)
        else:
            data = eopatch[self.feature_type][self.feature_name]

        file_dir = os.path.dirname(filename)
        os.makedirs(file_dir, exist_ok=True)

        if self.compress_level:
            file_handle = gzip.GzipFile(filename, 'w', self.compress_level)
        else:
            file_handle = open(filename, 'wb')

        with file_handle as outfile:
            LOGGER.debug("Saving (%s, %s) to %s", str(self.feature_type), str(self.feature_name), filename)

            if self.file_format is FileFormat.NPY:
                np.save(outfile, data)
            elif self.file_format is FileFormat.PICKLE:
                pickle.dump(data, outfile)
            else:
                ValueError('File {} was not saved because saving in file format {} is currently not '
                           'supported'.format(filename, self.file_format))
