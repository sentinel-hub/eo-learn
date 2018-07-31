"""
The eodata module provides core objects for handling remotely sensing multi-temporal data (such as satellite imagery).
"""

import os
import logging
import pickle
import numpy as np
import gzip

from enum import Enum
from copy import copy, deepcopy

from .utilities import deep_eq


LOGGER = logging.getLogger(__name__)


class FeatureType(Enum):
    """
    The Enum class of all possible feature types that can be included in EOPatch:
     - DATA with shape t x n x m x d: time- and position-dependent remote sensing data (e.g. bands) of type float
     - MASK with shape t x n x m x d': time- and position-dependent mask (e.g. ground truth, cloud/shadow mask,
       super pixel identifier) of type int
     - SCALAR with shape t x s: time-dependent and position-independent remote sensing data (e.g. weather data,) of type
       float
     - LABEL with shape t x s': time-dependent and position-independent label (e.g. ground truth) of type int
     - VECTOR: a list of time-dependent vector shapes in shapely.geometry classes
     - DATA_TIMELESS with shape n x m x d'': time-independent and position-dependent remote sensing data (e.g.
       elevation model) of type float
     - MASK_TIMELESS with shape n x m x d''': time-independent and position-dependent mask (e.g. ground truth,
       region of interest mask) of type int
     - SCALAR_TIMELESS with shape s'':  time-independent and position-independent remote sensing data of type float
     - LABEL_TIMELESS with shape s''': time-independent and position-independent label of type int
     - VECTOR_TIMELESS: time-independent vector shapes in shapely.geometry classes
     - META_INFO: dictionary of additional info (e.g. resolution, time difference)
     - BBOX: bounding box of the patch which is an instance of sentinelhub.BBox
     - TIMESTAMP: list of dates which are instances of datetime.datetime
    """
    # IMPORTANT: these feature names must exactly match those in EOPatch constructor
    DATA = 'data'
    MASK = 'mask'
    SCALAR = 'scalar'
    LABEL = 'label'
    VECTOR = 'vector'
    DATA_TIMELESS = 'data_timeless'
    MASK_TIMELESS = 'mask_timeless'
    SCALAR_TIMELESS = 'scalar_timeless'
    LABEL_TIMELESS = 'label_timeless'
    VECTOR_TIMELESS = 'vector_timeless'
    META_INFO = 'meta_info'
    BBOX = 'bbox'
    TIMESTAMP = 'timestamp'

    @classmethod
    def has_value(cls, value):
        """ Checks if value is in FeatureType values
        """
        return any(value == item.value for item in cls)

    def is_spatial(self):
        """Tells if FeatureType has a spatial component

        :param self: A feature type
        :type self: FeatureType
        :return: `True` if feature type has a spatial component and `False` otherwise.
        :rtype: bool
        """
        return self in frozenset([FeatureType.DATA, FeatureType.MASK, FeatureType.VECTOR, FeatureType.DATA_TIMELESS,
                                  FeatureType.MASK_TIMELESS, FeatureType.VECTOR_TIMELESS])

    def is_time_dependent(self):
        """Tells if FeatureType has a time component

        :param self: A feature type
        :type self: FeatureType
        :return: `True` if feature type has a time component and `False` otherwise.
        :rtype: bool
        """
        return self in frozenset([FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL,
                                  FeatureType.VECTOR])

    def is_discrete(self):
        """Tells if FeatureType should have discrete (integer) values

        :param self: A feature type
        :type self: FeatureType
        :return: `True` if feature type should have discrete values and `False` otherwise.
        :rtype: bool
        """
        return self in frozenset([FeatureType.MASK, FeatureType.MASK_TIMELESS, FeatureType.LABEL,
                                  FeatureType.LABEL_TIMELESS])

    def is_vector(self):
        """Tells if FeatureType is vector feature type

        :param self: A feature type
        :type self: FeatureType
        :return: `True` if feature type vector feature type and `False` otherwise.
        :rtype: bool
        """
        return self in frozenset([FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS])

    def has_dict(self):
        """Tells if FeatureType stores a dictionary

        :param self: A feature type
        :type self: FeatureType
        :return: `True` if feature type stores a dictionary and `False` otherwise.
        :rtype: bool
        """
        return self not in frozenset([FeatureType.TIMESTAMP, FeatureType.BBOX])

    def contains_ndarrays(self):
        """Tells if FeatureType stores a dictionary of numpy.ndarrays

        :param self: A feature type
        :type self: FeatureType
        :return: `True` if feature type stores a dictionary of numpy.ndarrays and `False` otherwise.
        :rtype: bool
        """
        return self in frozenset([FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL,
                                  FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS, FeatureType.SCALAR_TIMELESS,
                                  FeatureType.LABEL_TIMELESS])

    def ndim(self):
        """If given FeatureType stores a dictionary of numpy.ndarrays it returns dimensions of such arrays

        :param self: A feature type
        :type self: FeatureType
        :return: Number of dimensions of numpy.ndarrays or None if FeatureType doesn't store numpy.ndarrays
        :rtype: int or None
        """
        if self.contains_ndarrays():
            return {
                FeatureType.DATA: 4,
                FeatureType.MASK: 4,
                FeatureType.SCALAR: 2,
                FeatureType.LABEL: 2,
                FeatureType.DATA_TIMELESS: 3,
                FeatureType.MASK_TIMELESS: 3,
                FeatureType.SCALAR_TIMELESS: 1,
                FeatureType.LABEL_TIMELESS: 1
            }[self]
        return None

    def type(self):
        """Provides type of the data for the given FeatureType

        :return: A type of data
        """
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return object
        return dict


class FeatureDict(dict):
    """A dictionary structure that holds features of certain feature type. It also check that features have a correct
    dimension

    :param feature_dict: A dictionary of feature names and values
    :type feature_dict: dict(str: object)
    :param feature_type: Type of features
    :type feature_type: FeatureType
    """
    def __init__(self, feature_dict, feature_type):
        super(FeatureDict, self).__init__()

        self.feature_type = feature_type
        self.ndim = self.feature_type.ndim()

        for feature_name, value in feature_dict.items():
            self[feature_name] = value

    def __setitem__(self, feature_name, value):
        """ Before setting value to the dictionary it checks that value is of correct type and dimension
        """
        if self.ndim and (not isinstance(value, np.ndarray) or value.ndim != self.ndim):
            raise ValueError('{} feature has to be {} of dimension {}'.format(self.feature_type, np.ndarray, self.ndim))
        super(FeatureDict, self).__setitem__(feature_name, value)

    def get_dict(self):
        """ Returns a normal dictionary of features and value

        :return: A normal dictionary class
        :rtype: dict(str: object)
        """
        return dict(self)


class FileFormat(Enum):
    PICKLE = 'pickle'
    NPY = 'npy'


NPY_SUPPORTED_FTYPES = set([
    FeatureType.DATA,
    FeatureType.MASK,
    FeatureType.SCALAR,
    FeatureType.LABEL,
    FeatureType.VECTOR,
    FeatureType.DATA_TIMELESS,
    FeatureType.MASK_TIMELESS,
    FeatureType.SCALAR_TIMELESS,
    FeatureType.LABEL_TIMELESS,
    FeatureType.VECTOR_TIMELESS])


class EOPatch:
    """
    This is the basic data object for multi-temporal remotely sensed data, such as satellite imagery and
    its derivatives, mainly for development, training, and testing ML algorithms.

    The EOPatch contains multi-temporal remotely sensed data of a single patch of earth's surface defined by the
    bounding box in specific coordinate reference system. The patch can be a rectangle, polygon, or pixel in space.
    The EOPatch object can also be used to store derived quantities, such as for example means, standard deviations,
    etc ..., of a patch. In this case the 'space' dimension is equivalent to a pixel.

    Primary goal of EOPatch is to store remotely sensed data:
        - usually of shape n_time x height x width x n_features images, where height and width are the numbers of
          pixels in y and x, n_features is the number of features (i.e. bands/channels, cloud probability, ...),
          and n_time is the number of time-slices (the number of times this patch was recorded by the satellite
          -- can also be a single image)

    In addition to that other auxiliary information is also needed and can be stored in additional attributes of the
    EOPatch (thus extending the functionality of numpy ndarray).

    These attributes are:
        - data: A dictionary of FeatureType.DATA features

        - mask: A dictionary of FeatureType.MASK features

        - scalar: A dictionary of scalar features, each of shape n_times x d, d >= 1

        - label: A dictionary of labels, each of shape n_times x d, d >= 1

        - vector: A dictionary of lists of time-dependent vector shapes

        - data_timeless: A dictionary containing time-independent data (e.g. DEM of the bbox)

        - mask_timeless: A dictionary containing time-independent masks (e.g. cloud mask), each mask is a numpy array.

        - scalar_timeless: A dictionary of time-independent scalar features (e.g. standard deviation of heights of the
          terrain)

        - label_timeless: A dictionary of time-independent label features

        - vector_timeless: A dictionary of time-independent vector shapes

        - meta_info: A dictionary of meta information

        - bounding box: (bbox, crs) where bbox is an array of 4 floats and crs is the epsg code

        - timestamp: list of dimension 1 and length n_time, where each element represents the time (datetime object) at
          which the individual image was taken.

    Currently the EOPatch object doesn't enforce that the length of timestamp be equal to n_times dimensions of numpy
    arrays in other attributes.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, *, data=None, mask=None, scalar=None, label=None, vector=None, data_timeless=None,
                 mask_timeless=None, scalar_timeless=None, label_timeless=None, vector_timeless=None,
                 meta_info=None, bbox=None, timestamp=None):

        self.data = data if data is not None else {}
        self.mask = mask if mask is not None else {}
        self.scalar = scalar if scalar is not None else {}
        self.label = label if label is not None else {}
        self.vector = vector if vector is not None else {}
        self.data_timeless = data_timeless if data_timeless is not None else {}
        self.mask_timeless = mask_timeless if mask_timeless is not None else {}
        self.scalar_timeless = scalar_timeless if scalar_timeless is not None else {}
        self.label_timeless = label_timeless if label_timeless is not None else {}
        self.vector_timeless = vector_timeless if vector_timeless is not None else {}
        self.meta_info = meta_info if meta_info is not None else {}
        self.bbox = bbox
        self.timestamp = timestamp if timestamp is not None else []

    def __setattr__(self, key, value):
        """Before attribute is set it is checked that feature type attributes are of correct type and in case they
        are a dictionary they are cast to FeatureDict class
        """
        if FeatureType.has_value(key):
            feature_type = FeatureType(key)
            value_type = feature_type.type()
            if not isinstance(value, value_type):
                raise TypeError('Attribute {} only takes items of type {}'.format(feature_type, value_type))
            if feature_type.has_dict() and not isinstance(value, FeatureDict):
                value = FeatureDict(value, feature_type)

        super(EOPatch, self).__setattr__(key, value)

    def __getitem__(self, feature_type):
        """Provides features of requested feature type from EOPatch

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str
        :return: Dictionary of features
        """
        return getattr(self, FeatureType(feature_type).value)

    def __setitem__(self, feature_type, value):
        """Sets new dictionary / list to the given FeatureType

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str
        :param value: New dictionary or list
        :type value: dict or list
        :return: Dictionary of features
        """
        return setattr(self, FeatureType(feature_type).value, value)

    def __eq__(self, other):
        """ EO patches are defined equal if all FeatureType attributes, bbox, and timestamp are (deeply) equal.
        """
        if not isinstance(self, type(other)):
            return False

        for feature_type in FeatureType:
            if not deep_eq(self[feature_type], other[feature_type]):
                return False
        return True

    def __add__(self, other):
        """ Adding two EOPatches will result into concatenation and a new EOPatch will be produced
        """
        return EOPatch.concatenate(self, other)

    def __repr__(self):
        """ Representation of EOPatch object

        :return: representation
        :rtype: str
        """
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

    def __copy__(self, feature_list=None):
        """ Overwrites copy method

        :param feature_list: A list of features or feature types that will be copied into new EOPatch. If None, all
        features will be copied.

        Example: feature_list=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

        :type feature_list: list((FeatureType, str) or FeatureType) or None
        :return: Copied EOPatch
        :rtype: EOPatch
        """
        new_eopatch = EOPatch()
        for feature_type, features in self._parse_to_dict(feature_list).items():
            if features is True:
                new_eopatch[feature_type] = copy(self[feature_type])
            else:
                for feature_name in features:
                    new_eopatch[feature_type][feature_name] = self[feature_type][feature_name]
        return new_eopatch

    def __deepcopy__(self, feature_list=None):
        """ Overwrites deepcopy method

        :param feature_list: A list of features or feature types that will be copied into new EOPatch. If None, all
        features will be copied.

        Example: feature_list=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

        :type feature_list: list((FeatureType, str)) or None
        :return: Deep copied EOPatch
        :rtype: EOPatch
        """
        new_eopatch = self.__copy__(feature_list=feature_list)
        for feature_type in FeatureType:
            new_eopatch[feature_type] = deepcopy(new_eopatch[feature_type])

        return new_eopatch

    @staticmethod
    def _repr_value(value):
        """ Creates representation string for different types of data

        :param value: data in any type
        :return: representation string
        :rtype: str
        """
        if isinstance(value, np.ndarray):
            return '{}, shape={}, dtype={}'.format(type(value), value.shape, value.dtype)
        if isinstance(value, (list, tuple, dict)):
            return '{}, length={}'.format(type(value), len(value))
        return repr(value)

    @staticmethod
    def _parse_to_dict(feature_list):
        """ Parses list of features to dictionary of sets

        :param feature_list: A list of features or feature types that will be copied into new EOPatch. If None, all
        features will be copied.

        Example: feature_list=[(FeatureType.DATA, 'TRUE-COLOR'), (FeatureType.MASK, 'CLOUD-MASK'), FeatureType.LABEL]

        :type feature_list: list((FeatureType, str) or FeatureType) or None
        :return: dictionary of sets
        :rtype: dict(FeatureType: set(str))
        """
        if not feature_list:
            feature_list = FeatureType

        feature_dict = {}
        for feature_item in feature_list:
            if isinstance(feature_item, FeatureType):
                feature_dict[feature_item] = True
            elif isinstance(feature_item, (tuple, list)):
                feature_type = feature_item[0]
                if not isinstance(feature_type, FeatureType):
                    raise ValueError('First element of {} must be of type {}'.format(feature_item, FeatureType))
                if feature_type in [FeatureType.TIMESTAMP, FeatureType.BBOX]:
                    raise ValueError('{} cannot be in a tuple'.format(FeatureType.TIMESTAMP))

                feature_dict[feature_type] = feature_dict.get(feature_type, set())
                if feature_dict[feature_type] is not True:
                    for feature_name in feature_item[1:]:
                        feature_dict[feature_type].add(feature_name)
            else:
                raise ValueError('Item {} in feature_list must be of type {} or {}'.format(feature_item, tuple,
                                                                                           FeatureType))
        return feature_dict

    def remove_feature(self, feature_type, feature_name):
        """ Removes the feature ``feature_name`` from dictionary of ``feature_type``

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
        """ Sets EOPatch[feature_type][feature_name] to the given value

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
        """ Checks if given FeatureType contains a dictionary and raises an error if it doesn't

        :param feature_type: Type of feature
        :type feature_type: FeatureType
        :raise: TypeError
        """
        feature_type = FeatureType(feature_type)
        if feature_type.type() is not dict:
            raise TypeError('{} does not contain a dictionary of features'.format(feature_type))

    def set_bbox(self, new_bbox):
        """ Method for setting a new bounding box
        :param new_bbox: Bounding box of any type
        """
        self.bbox = new_bbox

    def set_timestamp(self, new_timestamp):
        """ Method for setting a new list of dates
        :param new_timestamp: list of dates
        :type new_timestamp: list(str)
        """
        self.timestamp = new_timestamp

    def get_feature(self, feature_type, feature_name=None):
        """
        Returns the array of corresponding feature.

        :param feature_type: Enum of the attribute
        :type feature_type: FeatureType
        :param feature_name: Name of the feature
        :type feature_name: str
        """
        if feature_name is None:
            return self[feature_type]
        return self[feature_type][feature_name]

    def get_features(self):
        """ Returns a dictionary of all non-empty features of EOPatch. The elements are either sets of feature names or
        a boolean `True` in case feature type has no dictionary of feature names

        :return: A dictionary of features
        :rtype: dict(FeatureType: str or True)
        """
        feature_dict = {}
        for feature_type in FeatureType:
            if self[feature_type]:
                feature_dict[feature_type] = set(self[feature_type]) if feature_type.has_dict() else True

        return feature_dict

    def get_feature_list(self):
        """ Returns a list of all non-empty features of EOPatch. The elements are either only FeatureType or a pair of
        FeatureType and feature name.

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
        """ Joins all data from two EOPatches and returns a new EOPatch. If timestamps don't match it will try to join
        all time-dependent features with the same name.

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

        timestamps_match = eopatch1.timestamp and eopatch2.timestamp and deep_eq(eopatch1.timestamp, eopatch2.timestamp)

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

            elif feature_type is FeatureType.TIMESTAMP and timestamps_match:
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
        """ A method that concatenates two numpy array along first axis

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

    def save(self, path, feature_list=None, file_format='pickle', compress=False):
        """ Saves EOPatch to disk.

        :param path: Location on the disk
        :type path: str
        :param feature_list: List of features types specifying features of which type will be saved. If set to `None`
        all features will be saved.
        :type feature_list: list(FeatureType) or None
        :param file_format: File format
        :type file_format: str or FileFormat
        :param compress: Compress features. Can only be used with npy file_format
        :type compress: bool
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            LOGGER.warning('Overwriting data in %s', path)

        file_format = FileFormat(file_format)
        ftype_paths = self._get_file_paths(path, NPY_SUPPORTED_FTYPES)

        if file_format == FileFormat.PICKLE and \
                any([os.path.isdir(p) for p in ftype_paths]):
            raise Exception("Cannot save in a different format")
        elif file_format == FileFormat.NPY and \
                any([os.path.isfile(p) for p in ftype_paths]):
            raise Exception("Cannot save in a different format")

        if feature_list is None:
            feature_list = list(FeatureType)

        for feature_type in feature_list:
            if not self[feature_type]:
                LOGGER.debug("Attribute '%s' is None, nothing to serialize", str(feature_type))
                continue

            if file_format == FileFormat.PICKLE \
                    or feature_type not in NPY_SUPPORTED_FTYPES:
                file_path = os.path.join(path, FeatureType(feature_type).value)

                with open(file_path, 'wb') as outfile:
                    LOGGER.debug("Saving %s to %s", str(feature_type), file_path)

                    pickle.dump(self[feature_type].get_dict()
                                if feature_type.has_dict()
                                else self[feature_type], outfile)

            elif file_format == FileFormat.NPY:
                dir_path = os.path.join(path, FeatureType(feature_type).value)
                os.makedirs(dir_path, exist_ok=True)

                for feature_name, feature in self[feature_type].items():
                    file_path = os.path.join(dir_path, feature_name)

                    if compress:
                        file_handle = gzip.GzipFile(file_path + '.npy.gz', 'w')
                    else:
                        file_handle = open(file_path + '.npy', 'wb')

                    LOGGER.debug("Saving %s to %s", str(feature_type), file_path)
                    np.save(file_handle, feature)
                    file_handle.close()

    @staticmethod
    def load(path, feature_list=None):
        """ Loads EOPatch from disk.

        :param path: Location on the disk
        :type path: str
        :param feature_list: List of features to be loaded. If set to None all features will be loaded.
        :type feature_list: list(FeatureType) or None
        :return: Loaded EOPatch
        :rtype: EOPatch
        """
        if not os.path.exists(path):
            raise ValueError('Specified path {} does not exist'.format(path))

        file_format = EOPatch._get_file_format(path)

        if feature_list is None:
            feature_list = [feature_type for feature_type in FeatureType]

        eopatch_content = {}
        for feature_type in FeatureType:
            ftype_path = os.path.join(path, FeatureType(feature_type).value)

            if not os.path.exists(ftype_path):
                continue

            if file_format == FileFormat.PICKLE or \
                    feature_type not in NPY_SUPPORTED_FTYPES:

                if os.path.getsize(ftype_path):
                    with open(ftype_path, "rb") as infile:
                        eopatch_content[feature_type.value] = pickle.load(infile)
            else:
                data = {}
                for file_name in os.listdir(ftype_path):
                    file_path = os.path.join(ftype_path, file_name)

                    if file_name.endswith('.npy'):
                        feature_name = file_name[:-4]
                        file_handle = open(file_path, 'rb')
                    elif file_name.endswith('.npy.gz'):
                        feature_name = file_name[:-7]
                        file_handle = gzip.open(file_path)
                    else:
                        continue

                    data[feature_name] = np.load(file_handle)
                    file_handle.close()

                eopatch_content[feature_type.value] = data

        return EOPatch(**eopatch_content)

    @staticmethod
    def _get_file_format(path):
        file_format = None

        file_paths = EOPatch._get_file_paths(path, NPY_SUPPORTED_FTYPES)

        for file_path in file_paths:
            if os.path.isfile(file_path):
                ftype_file_format = FileFormat.PICKLE
            elif os.path.isdir(file_path):
                ftype_file_format = FileFormat.NPY
            else:
                continue

            if file_format is None:
                file_format = ftype_file_format
            elif file_format != ftype_file_format:
                raise Exception("Found multiple formats")

        return file_format

    @staticmethod
    def _get_file_paths(path, feature_list):
        """ Returns a list of file paths on disk for each FeatureType in list of features

        :param path: Location on the disk
        :type path: str
        :param feature_list: List of features types
        :type feature_list: list(FeatureType)
        :return: A list of file paths
        :rtype: list(str)
        """
        return [os.path.join(path, FeatureType(feature).value) for feature in feature_list]

    def time_series(self, ref_date=None, scale_time=1):
        """
        Returns a numpy array with seconds passed between reference date and the timestamp of each image:

        time_series[i] = (timestamp[i] - ref_date).total_seconds()

        If reference date is none the first date in the EOPatch's timestamp is taken.

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
        """
        Removes all frames from the EOPatch with a date not found in the provided timestamps list.

        :param timestamps: keep frames with date found in this list
        :type timestamps: list of datetime objects
        :return: set of removed frames' dates
        :rtype: set of datetime objects
        """
        remove_from_patch = set(self.timestamp).difference(timestamps)
        remove_from_patch_idxs = [self.timestamp.index(rm_date) for rm_date in remove_from_patch]
        good_timestamp_idxs = [idx for idx, _ in enumerate(self.timestamp) if idx not in remove_from_patch_idxs]
        good_timestamps = [date for idx, date in enumerate(self.timestamp) if idx not in remove_from_patch_idxs]

        for feature_type in [feature_type for feature_type in FeatureType if feature_type.is_time_dependent()]:

            for feature_name, value in self[feature_type].items():
                if isinstance(value, np.ndarray):
                    self[feature_type][feature_name] = value[good_timestamp_idxs, ...]
                if isinstance(value, list):
                    self[feature_type][feature_name] = [value[idx] for idx in good_timestamp_idxs]

        self.timestamp = good_timestamps
        return remove_from_patch
