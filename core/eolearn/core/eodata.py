"""
The eodata module provides core objects for handling remotely sensing multi-temporal data (such as satellite imagery).
"""

import os
import logging
import pickle
import collections

from enum import Enum

import numpy as np

from .utilities import deep_eq


LOGGER = logging.getLogger(__name__)


class FeatureType(Enum):
    """
    The Enum class of all possible feature types that can be included in EOPatch:
     - DATA with shape t x n x m x d: time- and position-dependent remote sensing data (e.g. bands) of type float
     - MASK with shape t x n x m x d': time- and position-dependent mask (e.g. ground truth, cloud/shadow mask,
       super pixel identifier) of type int
     - DATA_TIMELESS with shape n x m x d'': time-independent and position-dependent remote sensing data (e.g.
       elevation model) of type float
     - MASK_TIMELESS with shape n x m x d''': time-independent and position-dependent mask (e.g. ground truth,
       region of interest mask) of type int
     - SCALAR with shape t x s: time-dependent and position-independent remote sensing data (e.g. weather data,) of type
       float
     - LABEL with shape t x s': time-dependent and position-independent label (e.g. ground truth) of type int
     - SCALAR_TIMELESS with shape s'':  time-independent and position-independent remote sensing data of type float
     - LABEL_TIMELESS with shape s''': time-independent and position-independent label of type int
     - META_INFO: dictionary of additional info (e.g. resolution, time difference)
     - BBOX: bounding box of the patch which is an instance of sentinelhub.BBox
     - TIMESTAMP: list of dates which are instances of datetime.datetime
    """
    # IMPORTANT: these feature names must exactly match those in EOPatch constructor
    DATA = 'data'
    MASK = 'mask'
    SCALAR = 'scalar'
    LABEL = 'label'
    DATA_TIMELESS = 'data_timeless'
    MASK_TIMELESS = 'mask_timeless'
    SCALAR_TIMELESS = 'scalar_timeless'
    LABEL_TIMELESS = 'label_timeless'
    META_INFO = 'meta_info'
    BBOX = 'bbox'
    TIMESTAMP = 'timestamp'


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
        - features: dictionary of feature names (length n_features) and their array indices

        - scalar: array of scalar features (aggregates over single image in a time series); shape n_time x n_scalar,
          where n_scalar is the number of all scalar features

        - bounding box: (bbox, crs) where bbox is an array of 4 floats and crs is the epsg code

        - data_timeless: A dictionary containing time-independent data (e.g. DEM of the bbox)

        - mask_timeless: A dictionary containing time-independent masks (e.g. cloud mask), each mask is a numpy array.

        - scalar: dictionary of scalar features, each of shape n_times x d, d >= 1

        - label: dictionary of labels, each of shape n_times x d, d >= 1

        - scalar_timeless: Dictionary of time-independent scalar features (e.g. standard deviation of heights of the
          terrain)

        - label_timeless: Dictionary of time-independent label features

        - timestamp: list of dimension 1 and length n_time, where each element represents the time (datetime object) at
          which the individual image was taken.

        - meta_info: dictionary of meta information

    Currently the EOPatch object doesn't enforce that the length of timestamp be equal to n_times dimensions of numpy
    arrays in other attributes.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, *, bbox=None, timestamp=None, data=None, mask=None,
                 scalar=None, label=None, data_timeless=None, mask_timeless=None,
                 scalar_timeless=None, label_timeless=None, meta_info=None):

        self.bbox = bbox

        self.timestamp = timestamp if timestamp is not None else []

        self.data = data if data is not None else {}
        self.mask = mask if mask is not None else {}
        self.scalar = scalar if scalar is not None else {}
        self.label = label if label is not None else {}
        self.data_timeless = data_timeless if data_timeless is not None else {}
        self.mask_timeless = mask_timeless if mask_timeless is not None else {}
        self.scalar_timeless = scalar_timeless if scalar_timeless is not None else {}
        self.label_timeless = label_timeless if label_timeless is not None else {}
        self.meta_info = meta_info if meta_info is not None else {}
        self.ndims = {'data': 4,
                      'mask': 4,
                      'data_timeless': 3,
                      'mask_timeless': 3,
                      'scalar': 2,
                      'label': 2,
                      'scalar_timeless': 1,
                      'label_timeless': 1}
        self.features = collections.defaultdict(dict)
        self._check_dimensions()
        self._initialize_features()

    def _check_dimensions(self):
        """ Check if dimensions of arrays are in line with requirements
        """
        for attr_type in FeatureType:
            if attr_type in [FeatureType.META_INFO, FeatureType.BBOX, FeatureType.TIMESTAMP]:
                continue
            attr = getattr(self, attr_type.value)
            for field, value in attr.items():
                if isinstance(value, np.ndarray) and (not value.ndim == self.ndims[attr_type.value]):
                    raise ValueError("Error in dimensionality of {0:s}.{1:s},"
                                     " has to be {2:d}D array".format(attr_type.value, field,
                                                                      self.ndims[attr_type.value]))

    def _initialize_features(self):
        for attr_type in FeatureType:
            if attr_type in [FeatureType.META_INFO, FeatureType.BBOX, FeatureType.TIMESTAMP]:
                continue
            attr = getattr(self, attr_type.value)
            for field, value in attr.items():
                if isinstance(value, np.ndarray):
                    self.features[attr_type][field] = value.shape
                else:
                    self.features[attr_type][field] = type(value)

    def __getitem__(self, attr_name):
        LOGGER.debug("Accessing attribute '%s'", attr_name)
        return getattr(self, attr_name)

    def __eq__(self, other):
        """
        EO patches are defined equal if all FeatureType attributes, bbox, and timestamp are (deeply) equal.
        """
        if not isinstance(self, type(other)):
            return False

        for ftr_type in FeatureType:
            if not deep_eq(getattr(self, ftr_type.value), getattr(other, ftr_type.value)):
                return False

        return self.bbox == other.bbox and self.timestamp == other.timestamp

    def add_meta_info(self, meta_info):
        """
        Adds meta information to existing meta info dictionary.

        :param meta_info: dictionary of meta information to be added
        :type meta_info: dictionary
        """
        self.meta_info = {**self.meta_info, **meta_info}

    def remove_feature(self, attr_type, field):
        """
        Removes the feature ``field`` from ``attr_type``
        :param attr_type: Enum of the attribute we're about to modify
        :type attr_type: FeatureType
        :param field: Name of the field of the attribute
        :type field: str
        """
        if not isinstance(attr_type, FeatureType):
            raise TypeError('Expected FeatureType instance for attribute type')

        LOGGER.debug("Removing feature '%s' from attribute '%s'", field, attr_type.value)

        attr = getattr(self, attr_type.value)

        if field in attr.keys():
            del attr[field]
            del self.features[attr_type][field]

    def add_feature(self, attr_type, field, value):
        """
        Sets the appropriate attribute's ``field`` to ``value``
        :param attr_type: Enum of the attribute we're about to modify
        :type attr_type: FeatureType
        :param field: Name of the field of the attribute
        :type field: str
        :param value: Value to store in the field of the attribute
        :type value: object
        """
        if not isinstance(attr_type, FeatureType):
            raise TypeError('Expected FeatureType instance for attribute type')

        LOGGER.debug("Accessing attribute '%s'", attr_type.value)

        attr = getattr(self, attr_type.value)
        attr[field] = value
        self._check_dimensions()
        self.features[attr_type][field] = value.shape

    def get_feature(self, attr_type, field):
        """
        Returns the array of corresponding feature.

        :param attr_type: Enum of the attribute
        :type attr_type: FeatureType
        :param field: Name of the field of the attribute
        :type field: str
        """
        if not isinstance(attr_type, FeatureType):
            raise TypeError('Expected FeatureType instance for attribute type')

        LOGGER.debug("Accessing attribute '%s'", attr_type.value)

        attr = getattr(self, attr_type.value)

        return attr[field] if field in attr.keys() else None

    def feature_exists(self, attr_type, field):
        """
        Checks if the corresponding feature exists.

        :param attr_type: Enum of the attribute
        :type attr_type: FeatureType
        :param field: Name of the field of the attribute
        :type field: str
        """
        if not isinstance(attr_type, FeatureType):
            raise TypeError('Expected FeatureType instance for attribute type')

        LOGGER.debug("Accessing attribute '%s'", attr_type.value)

        attr = getattr(self, attr_type.value)

        return field in attr.keys()

    def get_features(self):
        """ Returns all features of EOPatch
        :return: dictionary of features
        :rtype: dict(FeatureType)
        """
        return self.features

    @staticmethod
    def concatenate(eopatch1, eopatch2):
        """
        Combines all data from two EOPatches and returns the new EOPatch.

        For time-independent attribute ``a`` a key ``k`` is retrained if and only if we have
        ``eopatch1.a[k]==eopatch2.a[k]``.
        """

        if eopatch1.bbox != eopatch2.bbox:
            raise ValueError('Cannot concatenate two EOpatches with different BBoxes')

        def merge_dicts(fst_dict, snd_dict, concatenator=EOPatch._concatenate):
            if not fst_dict or not snd_dict:
                return {}

            if fst_dict.keys() != snd_dict.keys():
                raise ValueError('Key mismatch')

            return {field: concatenator(fst_dict[field], snd_dict[field]) for field in fst_dict}

        data = merge_dicts(eopatch1.data, eopatch2.data)

        timestamp = eopatch1.timestamp + eopatch2.timestamp

        bbox = eopatch1.bbox
        meta_info = {**eopatch2.meta_info, **eopatch1.meta_info}

        mask = merge_dicts(eopatch1.mask, eopatch2.mask)
        scalar = merge_dicts(eopatch1.scalar, eopatch2.scalar)
        label = merge_dicts(eopatch1.label, eopatch2.label)

        def merge_time_independent_dicts(fst_dict, snd_dict):
            merged_dict = {}
            if not fst_dict or not snd_dict:
                return merged_dict

            for field in fst_dict.keys() & snd_dict.keys():
                if isinstance(fst_dict[field], np.ndarray) and isinstance(snd_dict[field], np.ndarray):
                    if np.array_equal(fst_dict[field], snd_dict[field]):
                        merged_dict[field] = snd_dict[field]
                    else:
                        LOGGER.debug("Field %s skipped due to value mismatch", field)
                        continue
                elif fst_dict[field] == snd_dict[field]:
                    merged_dict[field] = fst_dict[field]
                else:
                    LOGGER.debug("Field %s skipped due to value mismatch", field)
            return merged_dict

        data_timeless = merge_time_independent_dicts(eopatch1.data_timeless, eopatch2.data_timeless)
        mask_timeless = merge_time_independent_dicts(eopatch1.mask_timeless, eopatch2.mask_timeless)
        scalar_timeless = merge_time_independent_dicts(eopatch1.scalar_timeless, eopatch2.scalar_timeless)
        label_timeless = merge_time_independent_dicts(eopatch1.label_timeless, eopatch2.label_timeless)

        return EOPatch(data=data, timestamp=timestamp, bbox=bbox, mask=mask, data_timeless=data_timeless,
                       mask_timeless=mask_timeless, scalar=scalar, label=label, scalar_timeless=scalar_timeless,
                       label_timeless=label_timeless, meta_info=meta_info)

    @staticmethod
    def _concatenate(data1, data2):
        """
        Private method to concatenate data nparrays.

        :param data1: array, shape (n_times1, height, width, n_features)
        :param data2: array, shape (n_times2, height, width, n_features)
        """

        data1_shape, data2_shape = data1.shape[1:], data2.shape[1:]

        if data1_shape == data2_shape:
            return np.concatenate((data1, data2), axis=0)

        raise TypeError('Add data failed. Entries are not of correct shape.\n'
                        'Expected {}, but got {}'.format(data1_shape, data2_shape))

    @staticmethod
    def _get_filenames(path):
        """
        Returns dictionary of filenames and locations on disk.
        """
        return {feature.value: os.path.join(path, feature.value) for feature in FeatureType}

    def save(self, path, feature_list=None):
        """
        Saves EOPatch to disk.

        :param path: Location on the disk
        :type path: str
        :param feature_list: List of features to be saved. If set to None all features will be saved.
        :type feature_list: list(FeatureType) or None
        """
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            LOGGER.warning('Overwriting data in %s', path)

        LOGGER.debug('Saving to %s', path)

        filenames = EOPatch._get_filenames(path)

        if feature_list is None:
            feature_list = FeatureType
        else:
            for feature in feature_list:
                if not isinstance(feature, FeatureType):
                    raise ValueError("Parameter feature_list must get a list with elements of type FeatureType")

        for feature in feature_list:
            attribute = feature.value
            path = filenames[attribute]

            LOGGER.debug("Saving %s to %s", attribute, path)

            with open(path, 'wb') as outfile:
                if not hasattr(self, attribute):
                    raise AttributeError(
                        "The object doesn't have attribute '{}' and hence cannot serialize it.".format(attribute))

                value = getattr(self, attribute)
                if value:
                    pickle.dump(value, outfile)
                else:
                    LOGGER.debug("Attribute '%s' is None, nothing to serialize", attribute)

    @staticmethod
    def load(path, feature_list=None):
        """
        Loads EOPatch from disk.

        :param path: Location on the disk
        :type path: str
        :param feature_list: List of features to be loaded. If set to None all features will be loaded.
        :type feature_list: list(FeatureType) or None
        :return: Loaded EOPatch
        :rtype: EOPatch
        """
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            LOGGER.warning('Specified path does not exist: %s', path)

        filenames = EOPatch._get_filenames(path)

        eopatch_features = {feature.value: None for feature in FeatureType}

        if feature_list is None:
            feature_list = FeatureType

        for feature in feature_list:
            feature_filename = filenames[feature.value]
            if os.path.exists(feature_filename) and os.path.getsize(feature_filename) > 0:
                with open(feature_filename, "rb") as feature_file:
                    eopatch_features[feature.value] = pickle.load(feature_file)

        return EOPatch(**eopatch_features)

    def time_series(self, ref_date=None):
        """
        Returns a numpy array with seconds passed between reference date and the timestamp of each image:

        time_series[i] = (timestamp[i] - ref_date).total_seconds()

        If reference date is none the first date in the EOPatch's timestamp is taken.

        If EOPatch timestamp atribute is empty the method returns None.

        :param ref_date: reference date relative to which the time is measured
        :type ref_date: datetime object
        """

        if not self.timestamp:
            return None

        if ref_date is None:
            ref_date = self.timestamp[0]

        return np.asarray([(timestamp - ref_date).total_seconds() for timestamp in self.timestamp], dtype=np.int64)
