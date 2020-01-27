"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
from collections import OrderedDict
from logging import Filter

import numpy as np
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from .constants import FeatureType

LOGGER = logging.getLogger(__name__)


class LogFileFilter(Filter):
    """ Filters log messages passed to log file
    """

    def __init__(self, thread_name, *args, **kwargs):
        self.thread_name = thread_name
        super().__init__(*args, **kwargs)

    def filter(self, record):
        """ Shows everything from the thread that it was initialized in.
        """
        return record.threadName == self.thread_name


class FeatureParser:
    """ Takes a collection of features structured in a various ways and parses them into one way. It can parse features
    straight away or it can parse them only if they exist in a given `EOPatch`. If input format is not recognized or
    feature don't exist in a given `EOPatch` it raises an error. The class is a generator therefore parsed features
    can be obtained by iterating over an instance of the class. An `EOPatch` is given as a parameter of the generator.

    General guidelines:
        - Almost all `EOTask`s have take as a parameter some information about features. The purpose of this class is
        to unite and generalize parsing of such parameter over entire eo-learn package
        - The idea for this class is that it should support more or less any logical way how to describe a collection
        of features.
        - Parameter `...` is used as a contextual clue. In the supported formats it is used to describe the most obvious
        way how to specify certain parts of feature collection.
        - Supports formats defined with lists, tuples, sets and dictionaries.

    Supported input formats:
        - `...` - Anything that exists in a given `EOPatch`
        - A feature type describing all features of that type. E.g. `FeatureType.DATA` or `FeatureType.BBOX`
        - A single feature as a tuple. E.g. (FeatureType.DATA, 'BANDS')
        - A single feature as a tuple with new name. E.g. (FeatureType.DATA, 'BANDS', 'NEW_BANDS')
        - A list of features (new names or not).
        E.g. [(FeatureType.DATA, 'BANDS'), (FeatureType.MASK, 'CLOUD_MASK', 'NEW_CLOUD_MASK')]
        - A dictionary with feature types as keys and lists, sets, single feature or `...` of feature names as values.
        E.g. {
            FeatureType.DATA: ['S2-BANDS', 'L8-BANDS'],
            FeatureType.MASK: {'IS_VALID', 'IS_DATA'},
            FeatureType.MASK_TIMELESS: 'LULC',
            FeatureType.TIMESTAMP: ...
        }
        - A dictionary with feature types as keys and dictionaries, where feature names are mapped into new names, as
          values.
        E.g. {
            FeatureType.DATA: {
                'S2-BANDS': 'INTERPOLATED_S2_BANDS',
                'L8-BANDS': 'INTERPOLATED_L8_BANDS',
                'NDVI': ...
            },
        }

    Note: Therese are most general input formats, but even more are supported or might be supported in the future.

    Outputs of the generator:
        - tuples in form of (feature type, feature name) if parameter `new_names=False`
        - tuples in form of (feature type, feature name, new feature name) if parameter `new_names=True`
    """
    def __init__(self, features, new_names=False, rename_function=None, default_feature_type=None,
                 allowed_feature_types=None):
        """
        :param features: A collection of features in one of the supported formats
        :type features: object
        :param new_names: If `False` the generator will only return tuples with in form of
            (feature type, feature name). If `True` it will return tuples
            (feature type, feature name, new feature name) which can be used for renaming
            features or creating new features out of old ones.
        :type new_names: bool
        :param rename_function: A function which transforms feature name into a new feature name, default is identity
            function. This parameter is only applied if `new_names` is set to `True`.
        :type rename_function: function or None
        :param default_feature_type: If feature type of any given feature is not set, this will be used. By default this
            is set to `None`. In this case if feature type of any feature is not given the following will happen:
                - if iterated over `EOPatch` - It will try to find a feature with matching name in EOPatch. If such
                    features exist, it will return any of them. Otherwise it will raise an error.
                - if iterated without `EOPatch` - It will return `...` instead of a feature type.
        :type default_feature_type: FeatureType or None
        :param allowed_feature_types: Makes sure that only features of these feature types will be returned, otherwise
            an error is raised
        :type: set(FeatureType) or None
        :raises: ValueError
        """
        self.feature_collection = self._parse_features(features, new_names)
        self.new_names = new_names
        self.rename_function = rename_function
        self.default_feature_type = default_feature_type
        self.allowed_feature_types = FeatureType if allowed_feature_types is None else set(allowed_feature_types)

        if rename_function is None:
            self.rename_function = self._identity_rename_function  # <- didn't use lambda function - it can't be pickled

        if allowed_feature_types is not None:
            self._check_feature_types()

    def __call__(self, eopatch=None):
        return self._get_features(eopatch)

    def __iter__(self):
        return self._get_features()

    @staticmethod
    def _parse_features(features, new_names):
        """Takes a collection of features structured in a various ways and parses them into one way.

        If input format is not recognized it raises an error.

        :return: A collection of features
        :rtype: collections.OrderedDict(FeatureType: collections.OrderedDict(str: str or Ellipsis) or Ellipsis)
        :raises: ValueError
        """
        if isinstance(features, dict):
            return FeatureParser._parse_dict(features, new_names)

        if isinstance(features, list):
            return FeatureParser._parse_list(features, new_names)

        if isinstance(features, tuple):
            return FeatureParser._parse_tuple(features, new_names)

        if features is ...:
            return OrderedDict([(feature_type, ...) for feature_type in FeatureType])

        if isinstance(features, FeatureType):
            return OrderedDict([(features, ...)])

        if isinstance(features, str):
            return OrderedDict([(None, OrderedDict([(features, ...)]))])

        raise ValueError('Unknown format of input features: {}'.format(features))

    @staticmethod
    def _parse_dict(features, new_names):
        """Helping function of `_parse_features` that parses a list."""
        feature_collection = OrderedDict()
        for feature_type, feature_names in features.items():
            try:
                feature_type = FeatureType(feature_type)
            except ValueError:
                ValueError('Failed to parse {}, keys of the dictionary have to be instances '
                           'of {}'.format(features, FeatureType.__name__))

            feature_collection[feature_type] = feature_collection.get(feature_type, OrderedDict())

            if feature_names is ...:
                feature_collection[feature_type] = ...

            if feature_type.has_dict() and feature_collection[feature_type] is not ...:
                feature_collection[feature_type].update(FeatureParser._parse_feature_names(feature_names, new_names))

        return feature_collection

    @staticmethod
    def _parse_list(features, new_names):
        """Helping function of `_parse_features` that parses a list."""
        feature_collection = OrderedDict()
        for feature in features:
            if isinstance(feature, FeatureType):
                feature_collection[feature] = ...

            elif isinstance(feature, (tuple, list)):
                for feature_type, feature_dict in FeatureParser._parse_tuple(feature, new_names).items():
                    feature_collection[feature_type] = feature_collection.get(feature_type, OrderedDict())

                    if feature_dict is ...:
                        feature_collection[feature_type] = ...

                    if feature_collection[feature_type] is not ...:
                        feature_collection[feature_type].update(feature_dict)
            else:
                raise ValueError('Failed to parse {}, expected a tuple'.format(feature))
        return feature_collection

    @staticmethod
    def _parse_tuple(features, new_names):
        """Helping function of `_parse_features` that parses a tuple."""
        name_idx = 1
        try:
            feature_type = FeatureType(features[0])
        except ValueError:
            feature_type = None
            name_idx = 0

        if feature_type and not feature_type.has_dict():
            return OrderedDict([(feature_type, ...)])
        return OrderedDict([(feature_type, FeatureParser._parse_names_tuple(features[name_idx:], new_names))])

    @staticmethod
    def _parse_feature_names(feature_names, new_names):
        """Helping function of `_parse_features` that parses a collection of feature names."""
        if isinstance(feature_names, set):
            return FeatureParser._parse_names_set(feature_names)

        if isinstance(feature_names, dict):
            return FeatureParser._parse_names_dict(feature_names)

        if isinstance(feature_names, (tuple, list)):
            return FeatureParser._parse_names_tuple(feature_names, new_names)

        raise ValueError('Failed to parse {}, expected dictionary, set or tuple'.format(feature_names))

    @staticmethod
    def _parse_names_set(feature_names):
        """Helping function of `_parse_feature_names` that parses a set of feature names."""
        feature_collection = OrderedDict()
        for feature_name in feature_names:
            if isinstance(feature_name, str):
                feature_collection[feature_name] = ...
            else:
                raise ValueError('Failed to parse {}, expected string'.format(feature_name))
        return feature_collection

    @staticmethod
    def _parse_names_dict(feature_names):
        """Helping function of `_parse_feature_names` that parses a dictionary of feature names."""
        feature_collection = OrderedDict()
        for feature_name, new_feature_name in feature_names.items():
            if isinstance(feature_name, str) and (isinstance(new_feature_name, str) or
                                                  new_feature_name is ...):
                feature_collection[feature_name] = new_feature_name
            else:
                if not isinstance(feature_name, str):
                    raise ValueError('Failed to parse {}, expected string'.format(feature_name))
                raise ValueError('Failed to parse {}, expected string or Ellipsis'.format(new_feature_name))
        return feature_collection

    @staticmethod
    def _parse_names_tuple(feature_names, new_names):
        """Helping function of `_parse_feature_names` that parses a tuple or a list of feature names."""
        for feature in feature_names:
            if not isinstance(feature, str) and feature is not ...:
                raise ValueError('Failed to parse {}, expected a string'.format(feature))

        if feature_names[0] is ...:
            return ...

        if new_names:
            if len(feature_names) == 1:
                return OrderedDict([(feature_names[0], ...)])
            if len(feature_names) == 2:
                return OrderedDict([(feature_names[0], feature_names[1])])
            raise ValueError("Failed to parse {}, it should contain at most two strings".format(feature_names))

        if ... in feature_names:
            return ...
        return OrderedDict([(feature_name, ...) for feature_name in feature_names])

    def _check_feature_types(self):
        """ Checks that feature types are a subset of allowed feature types. (`None` is handled

        :raises: ValueError
        """
        if self.default_feature_type is not None and self.default_feature_type not in self.allowed_feature_types:
            raise ValueError('Default feature type parameter must be one of the allowed feature types')

        for feature_type in self.feature_collection:
            if feature_type is not None and feature_type not in self.allowed_feature_types:
                raise ValueError('Feature type has to be one of {}, but {} found'.format(self.allowed_feature_types,
                                                                                         feature_type))

    def _get_features(self, eopatch=None):
        """A generator of parsed features.

        :param eopatch: A given EOPatch
        :type eopatch: EOPatch or None
        :return: One by one feature
        :rtype: tuple(FeatureType, str) or tuple(FeatureType, str, str)
        """
        for feature_type, feature_dict in self.feature_collection.items():
            if feature_type is None and self.default_feature_type is not None:
                feature_type = self.default_feature_type

            if feature_type is None:
                for feature_name, new_feature_name in feature_dict.items():
                    if eopatch is None:
                        yield self._return_feature(..., feature_name, new_feature_name)
                    else:
                        found_feature_type = self._find_feature_type(feature_name, eopatch)
                        if found_feature_type:
                            yield self._return_feature(found_feature_type, feature_name, new_feature_name)
                        else:
                            raise ValueError("Feature with name '{}' does not exist among features of allowed feature"
                                             " types in given EOPatch. Allowed feature types are "
                                             "{}".format(feature_name, self.allowed_feature_types))
            elif feature_dict is ...:
                if not feature_type.has_dict() or eopatch is None:
                    yield self._return_feature(feature_type, ...)
                else:
                    for feature_name in eopatch[feature_type]:
                        yield self._return_feature(feature_type, feature_name)
            else:
                for feature_name, new_feature_name in feature_dict.items():
                    if eopatch is not None and feature_name not in eopatch[feature_type]:
                        raise ValueError('Feature {} of type {} was not found in EOPatch'.format(feature_name,
                                                                                                 feature_type))
                    yield self._return_feature(feature_type, feature_name, new_feature_name)

    def _find_feature_type(self, feature_name, eopatch):
        """ Iterates over allowed feature types of given EOPatch and tries to find a feature type for which there
        exists a feature with given name

        :return: A feature type or `None` if such feature type does not exist
        :rtype: FeatureType or None
        """
        for feature_type in self.allowed_feature_types:
            if feature_type.has_dict() and feature_name in eopatch[feature_type]:
                return feature_type
        return None

    def _return_feature(self, feature_type, feature_name, new_feature_name=...):
        """ Helping function of `get_features`
        """
        if self.new_names:
            return feature_type, feature_name, (self.rename_function(feature_name) if new_feature_name is ... else
                                                new_feature_name)
        return feature_type, feature_name

    @staticmethod
    def _identity_rename_function(name):
        return name


def get_common_timestamps(source, target):
    """Return indices of timestamps from source that are also found in target.

    :param source: timestamps from source
    :type source: list of datetime objects
    :param target: timestamps from target
    :type target: list of datetime objects
    :return: indices of timestamps from source that are also found in target
    :rtype: list of ints
    """
    remove_from_source = set(source).difference(target)
    remove_from_source_idxs = [source.index(rm_date) for rm_date in remove_from_source]
    return [idx for idx, _ in enumerate(source) if idx not in remove_from_source_idxs]


def deep_eq(fst_obj, snd_obj):
    """Compares whether fst_obj and snd_obj are deeply equal.

    In case when both fst_obj and snd_obj are of type np.ndarray or either np.memmap, they are compared using
    np.array_equal(fst_obj, snd_obj). Otherwise, when they are lists or tuples, they are compared for length and then
    deep_eq is applied component-wise. When they are dict, they are compared for key set equality, and then deep_eq is
    applied value-wise. For all other data types that are not list, tuple, dict, or np.ndarray, the method falls back
    to the __eq__ method.

    Because np.ndarray is not a hashable object, it is impossible to form a set of numpy arrays, hence deep_eq works
    correctly.

    :param fst_obj: First object compared
    :param snd_obj: Second object compared
    :return: `True` if objects are deeply equal, `False` otherwise
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(fst_obj, type(snd_obj)):
        return False

    if isinstance(fst_obj, np.ndarray):
        if fst_obj.dtype != snd_obj.dtype:
            return False
        fst_nan_mask = np.isnan(fst_obj)
        snd_nan_mask = np.isnan(snd_obj)
        return np.array_equal(fst_obj[~fst_nan_mask], snd_obj[~snd_nan_mask]) and \
            np.array_equal(fst_nan_mask, snd_nan_mask)

    if isinstance(fst_obj, GeoDataFrame):
        try:
            assert_geodataframe_equal(fst_obj, snd_obj)
            return True
        except AssertionError:
            return False

    if isinstance(fst_obj, (list, tuple)):
        if len(fst_obj) != len(snd_obj):
            return False

        for element_fst, element_snd in zip(fst_obj, snd_obj):
            if not deep_eq(element_fst, element_snd):
                return False
        return True

    if isinstance(fst_obj, dict):
        if fst_obj.keys() != snd_obj.keys():
            return False

        for key in fst_obj:
            if not deep_eq(fst_obj[key], snd_obj[key]):
                return False
        return True

    return fst_obj == snd_obj


def negate_mask(mask):
    """Returns the negated mask.

    If elements of input mask have 0 and non-zero values, then the returned matrix will have all elements 0 (1) where
    the original one has non-zero (0).

    :param mask: Input mask
    :type mask: np.array
    :return: array of same shape and dtype=int8 as input array
    :rtype: np.array
    """
    res = np.ones(mask.shape, dtype=np.int8)
    res[mask > 0] = 0

    return res


def constant_pad(X, multiple_of, up_down_rule='even', left_right_rule='even', pad_value=0):
    """Function pads an image of shape (rows, columns, channels) with zeros.

    It pads an image so that the shape becomes (rows + padded_rows, columns + padded_columns, channels), where
    padded_rows = (int(rows/multiple_of[0]) + 1) * multiple_of[0] - rows

    Same rule is applied to columns.

    :type X: array of shape (rows, columns, channels) or (rows, columns)
    :param multiple_of: make X' rows and columns multiple of this tuple
    :type multiple_of: tuple (rows, columns)
    :param up_down_rule: Add padded rows evenly to the top/bottom of the image, or up (top) / down (bottom) only
    :type up_down_rule: up_down_rule: string, (even, up, down)
    :param up_down_rule: Add padded columns evenly to the left/right of the image, or left / right only
    :type up_down_rule: up_down_rule: string, (even, left, right)
    :param pad_value: Value to be assigned to padded rows and columns
    :type pad_value: int
    """
    # pylint: disable=invalid-name
    shape = X.shape

    row_padding, col_padding = 0, 0

    if shape[0] % multiple_of[0]:
        row_padding = (int(shape[0] / multiple_of[0]) + 1) * multiple_of[0] - shape[0]

    if shape[1] % multiple_of[1]:
        col_padding = (int(shape[1] / multiple_of[1]) + 1) * multiple_of[1] - shape[1]

    row_padding_up, row_padding_down, col_padding_left, col_padding_right = 0, 0, 0, 0

    if row_padding > 0:
        if up_down_rule == 'up':
            row_padding_up = row_padding
        elif up_down_rule == 'down':
            row_padding_down = row_padding
        elif up_down_rule == 'even':
            row_padding_up = int(row_padding / 2)
            row_padding_down = row_padding_up + (row_padding % 2)
        else:
            raise ValueError('Padding rule for rows not supported. Choose beteen even, down or up!')

    if col_padding > 0:
        if left_right_rule == 'left':
            col_padding_left = col_padding
        elif left_right_rule == 'right':
            col_padding_right = col_padding
        elif left_right_rule == 'even':
            col_padding_left = int(col_padding / 2)
            col_padding_right = col_padding_left + (col_padding % 2)
        else:
            raise ValueError('Padding rule for columns not supported. Choose beteen even, left or right!')

    return np.lib.pad(X, ((row_padding_up, row_padding_down), (col_padding_left, col_padding_right)),
                      'constant', constant_values=((pad_value, pad_value), (pad_value, pad_value)))


def bgr_to_rgb(bgr):
    """Converts Blue, Green, Red to Red, Green, Blue."""
    return bgr[..., [2, 1, 0]]
