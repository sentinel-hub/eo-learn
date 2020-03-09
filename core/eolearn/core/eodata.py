"""
The eodata module provides core objects for handling remote sensing multi-temporal data (such as satellite imagery).

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import logging
import warnings
import copy
import datetime

import attr
import dateutil.parser
import numpy as np
import geopandas as gpd

from sentinelhub import BBox

from .constants import FeatureType, OverwritePermission
from .eodata_io import save_eopatch, load_eopatch, FeatureIO
from .fs_utils import get_filesystem
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

    def __setattr__(self, key, value, feature_name=None):
        """Raises TypeError if feature type attributes are not of correct type.

        In case they are a dictionary they are cast to _FeatureDict class
        """
        if feature_name not in (None, Ellipsis) and FeatureType.has_value(key):
            self[key][feature_name] = value
            return

        if FeatureType.has_value(key) and not isinstance(value, FeatureIO):
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
                        'failed to parse given value {}'.format(feature_type, feature_type.type(), value))

    def __getattribute__(self, key, load=True, feature_name=None):
        """ Handles lazy loading and it can even provide a single feature from _FeatureDict
        """
        value = super().__getattribute__(key)

        if isinstance(value, FeatureIO) and load:
            value = value.load()
            setattr(self, key, value)
            value = getattr(self, key)

        if feature_name not in (None, Ellipsis) and isinstance(value, _FeatureDict):
            return value[feature_name]

        return value

    def __getitem__(self, feature_type):
        """ Provides features of requested feature type. It can also accept a tuple of (feature_type, feature_name)

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str or (FeatureType, str)
        :return: Dictionary of features
        """
        feature_name = None
        if isinstance(feature_type, tuple):
            self._check_tuple_key(feature_type)
            feature_type, feature_name = feature_type

        return self.__getattribute__(FeatureType(feature_type).value, feature_name=feature_name)

    def __setitem__(self, feature_type, value):
        """Sets a new dictionary / list to the given FeatureType. As a key it can also accept a tuple of
        (feature_type, feature_name)

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str or (FeatureType, str)
        :param value: New dictionary or list
        :type value: dict or list
        :return: Dictionary of features
        """
        feature_name = None
        if isinstance(feature_type, tuple):
            self._check_tuple_key(feature_type)
            feature_type, feature_name = feature_type

        return self.__setattr__(FeatureType(feature_type).value, value, feature_name=feature_name)

    @staticmethod
    def _check_tuple_key(key):
        """ A helper function that checks a tuple, which should hold (feature_type, feature_name)
        """
        if len(key) != 2:
            raise ValueError('Given element should be a feature_type or a tuple of (feature_type, feature_name),'
                             'but {} found'.format(key))

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
        if isinstance(value, gpd.GeoDataFrame):
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
                new_eopatch[feature_type] = copy.copy(self[feature_type])
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
            new_eopatch[feature_type] = copy.deepcopy(new_eopatch[feature_type], memo)

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

    def rename_feature(self, feature_type, feature_name, new_feature_name):
        """Renames the feature ``feature_name`` to ``new_feature_name`` from dictionary of ``feature_type``.

        :param feature_type: Enum of the attribute we're about to rename
        :type feature_type: FeatureType
        :param feature_name: Name of the feature of the attribute
        :type feature_name: str
        :param new_feature_name : New Name of the feature of the attribute
        :type feature_name: str
        """

        self._check_if_dict(feature_type)
        if feature_name != new_feature_name:
            if feature_name in self[feature_type]:
                LOGGER.debug("Renaming feature '%s' from attribute '%s' to '%s'",
                             feature_name, feature_type.value, new_feature_name)
                self[feature_type][new_feature_name] = self[feature_type][feature_name]
                del self[feature_type][feature_name]
            else:
                raise ValueError("Feature {} from attribute {} does not exist!".format(
                    feature_name, feature_type.value))
        else:
            LOGGER.debug("Feature '%s' was not renamed because new name is identical.", feature_name)

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
        """
        :param new_bbox: new bbox
        :type: new_bbox: BBox
        """
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
                    eopatch_content[feature_type.value] = copy.copy(eopatch2[feature_type])
                elif not eopatch2[feature_type]:
                    eopatch_content[feature_type.value] = copy.copy(eopatch1[feature_type])
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

    def save(self, path, features=..., overwrite_permission=OverwritePermission.ADD_ONLY, compress_level=0,
             filesystem=None):
        """ Method to save an EOPatch from memory to a storage

        :param path: A location where to save EOPatch. It can be either a local path or a remote URL path.
        :type path: str
        :param features: A collection of features types specifying features of which type will be saved. By default
        all features will be saved.
        :type features: list(FeatureType) or list((FeatureType, str)) or ...
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified with an integer from 0 (no compression)
            to 9 (highest compression).
        :type compress_level: int
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the `path`
            parameter.
        :type filesystem: fs.FS or None
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=True)
            path = '/'

        save_eopatch(self, filesystem, path, features=features, compress_level=compress_level,
                     overwrite_permission=OverwritePermission(overwrite_permission))

    @staticmethod
    def load(path, features=..., lazy_loading=False, filesystem=None):
        """ Method to load an EOPatch from a storage into memory

        :param path: A location from where to load EOPatch. It can be either a local path or a remote URL path.
        :type path: str
        :param features: A collection of features to be loaded. By default all features will be loaded.
        :type features: object
        :param lazy_loading: If `True` features will be lazy loaded.
        :type lazy_loading: bool
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the `path`
            parameter.
        :type filesystem: fs.FS or None
        :return: Loaded EOPatch
        :rtype: EOPatch
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=False)
            path = '/'

        return load_eopatch(EOPatch(), filesystem, path, features=features, lazy_loading=lazy_loading)

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

    def plot(self, feature, rgb=None, rgb_factor=3.5, vdims=None, timestamp_column='TIMESTAMP',
             geometry_column='geometry', pixel=False, mask=None):
        """ Plots eopatch features

        :param feature: feature of eopatch
        :type feature: (FeatureType, str)
        :param rgb: indexes of bands to create rgb image from
        :type rgb: [int, int, int]
        :param rgb_factor: factor for rgb bands multiplication
        :type rgb_factor: float
        :param vdims: value dimension for vector data
        :type vdims: str
        :param timestamp_column: name of the timestamp column, valid for vector data
        :type timestamp_column: str
        :param geometry_column: name of the geometry column, valid for vector data
        :type geometry_column: str
        :param pixel: plot values through time for one pixel
        :type pixel: bool
        :param mask: where eopatch[FeatureType.MASK] == False, value = 0
        :type mask: str
        :return: plot
        :rtype: holovies/bokeh
        """
        try:
            # pylint: disable=C0415
            from eolearn.visualization import EOPatchVisualization
        except ImportError:
            raise RuntimeError('Subpackage eo-learn-visualization has to be installed with an option [FULL] in order '
                               'to use plot method')

        vis = EOPatchVisualization(self, feature=feature, rgb=rgb, rgb_factor=rgb_factor, vdims=vdims,
                                   timestamp_column=timestamp_column, geometry_column=geometry_column,
                                   pixel=pixel, mask=mask)
        return vis.plot()


class _FeatureDict(dict):
    """A dictionary structure that holds features of certain feature type.

    It checks that features have a correct and dimension. It also supports lazy loading by accepting a function as a
    feature value, which is then called when the feature is accessed.
    """
    FORBIDDEN_CHARS = {'.', '/', '\\', '|', ';', ':', '\n', '\t'}

    def __init__(self, feature_dict, feature_type):
        """
        :param feature_dict: A dictionary of feature names and values
        :type feature_dict: dict(str: object)
        :param feature_type: Type of features
        :type feature_type: FeatureType
        """
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
        self._check_feature_name(feature_name)
        super().__setitem__(feature_name, value)

    def _check_feature_name(self, feature_name):
        if not isinstance(feature_name, str):
            error_msg = "Feature name must be a string but an object of type {} was given."
            raise ValueError(error_msg.format(type(feature_name)))

        for char in feature_name:
            if char in self.FORBIDDEN_CHARS:
                error_msg = "The name of feature ({}, {}) contains an illegal character '{}'."
                raise ValueError(error_msg.format(self.feature_type, feature_name, char))

        if feature_name == '':
            raise ValueError("Feature name cannot be an empty string.")

    def __getitem__(self, feature_name, load=True):
        """Implements lazy loading."""
        value = super().__getitem__(feature_name)

        if isinstance(value, FeatureIO) and load:
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
        if isinstance(value, FeatureIO):
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
            if isinstance(value, gpd.GeoSeries):
                value = gpd.GeoDataFrame(dict(geometry=value), crs=value.crs)

            if isinstance(value, gpd.GeoDataFrame):
                if self.feature_type is FeatureType.VECTOR:
                    if FeatureType.TIMESTAMP.value.upper() not in value:
                        raise ValueError("{} feature has to contain a column 'TIMESTAMP' with "
                                         "timestamps".format(self.feature_type))

                return value

            raise ValueError('{} feature works with data of type {}, parsing data type {} is not supported'
                             'given'.format(self.feature_type, gpd.GeoDataFrame.__name__, type(value)))

        return value
