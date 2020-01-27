"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
import datetime as dt

import numpy as np

from sentinelhub.time_utils import iso_to_datetime
from eolearn.core import EOTask, FeatureType


LOGGER = logging.getLogger(__name__)


class SimpleFilterTask(EOTask):
    """
    Transforms an eopatch of shape [n, w, h, d] into [m, w, h, d] for m <= n. It removes all slices which don't
    conform to the filter_func.

    A filter_func is a callable which takes an numpy array and returns a bool.
    """
    def __init__(self, feature, filter_func, filter_features=...):
        """
        :param feature: Feature in the EOPatch , e.g. feature=(FeatureType.DATA, 'bands')
        :type feature: (FeatureType, str)
        :param filter_func: A callable that takes a numpy evaluates to bool.
        :type filter_func: object
        :param filter_features: A collection of features which will be filtered
        :type filter_features: dict(FeatureType: set(str))
        """
        self.feature = self._parse_features(feature)
        self.filter_func = filter_func
        self.filter_features = self._parse_features(filter_features)

    def _get_filtered_indices(self, feature_data):
        return [idx for idx, img in enumerate(feature_data) if self.filter_func(img)]

    def _update_other_data(self, eopatch):
        pass

    def execute(self, eopatch):
        """
        :param eopatch: Input EOPatch.
        :type eopatch: EOPatch
        :return: Transformed eo patch
        :rtype: EOPatch
        """
        feature_type, feature_name = next(self.feature(eopatch))

        good_idxs = self._get_filtered_indices(eopatch[feature_type][feature_name] if feature_name is not ... else
                                               eopatch[feature_type])
        if not good_idxs:
            raise RuntimeError('EOPatch has no good indices after filtering with given filter function')

        for feature_type, feature_name in self.filter_features(eopatch):
            if feature_type.is_time_dependent():
                if feature_type.has_dict():
                    if feature_type.contains_ndarrays():
                        eopatch[feature_type][feature_name] = np.asarray([eopatch[feature_type][feature_name][idx] for
                                                                          idx in good_idxs])
                    # else:
                    #     NotImplemented
                else:
                    eopatch[feature_type] = [eopatch[feature_type][idx] for idx in good_idxs]

        self._update_other_data(eopatch)

        return eopatch


class FilterTimeSeries(SimpleFilterTask):
    """
    Removes all frames in the time-series with dates outside the user specified time interval.
    """
    def __init__(self, start_date, end_date, filter_features=...):
        """
        :param start_date: Start date. All frames within the time-series taken after this date will be kept.
        :type start_date: datetime.datetime
        :param end_date: End date. All frames within the time-series taken before this date will be kept.
        :type end_date: datetime.datetime
        :param filter_features: A collection of features which will be filtered
        :type filter_features: dict(FeatureType: set(str))
        """
        self.start_date = start_date
        self.end_date = end_date

        if not isinstance(start_date, dt.datetime):
            raise ValueError('Start date is not of correct type. Please provide the start_date as datetime.datetime.')

        if not isinstance(end_date, dt.datetime):
            raise ValueError('End date is not of correct type. Please provide the end_date as datetime.datetime.')

        super().__init__(FeatureType.TIMESTAMP, lambda date: start_date <= date <= end_date, filter_features)

    def _update_other_data(self, eopatch):

        if 'time_interval' in eopatch.meta_info:

            start_time, end_time = [iso_to_datetime(x) if isinstance(x, str)
                                    else x for x in eopatch.meta_info['time_interval']]
            eopatch.meta_info['time_interval'] = (max(start_time, self.start_date),
                                                  min(end_time, self.end_date))

        return eopatch


class ValueFilloutTask(EOTask):
    """ Overwrites occurrences of a desired value with their neighbor values in either forward, backward direction or
    both, along an axis.

    Possible fillout operations are 'f' (forward), 'b' (backward) or both, 'fb' or 'bf':

        'f': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> nan, nan, nan, 8, 5, 5, 1, 0, 0, 0

        'b': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 1, 1, 0, nan, nan

        'fb': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 5, 1, 0, 0, 0

        'bf': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 1, 1, 0, 0, 0
    """

    def __init__(self, feature, operations='fb', value=np.nan, axis=0):
        """
        :param feature: A feature that must be value-filled.
        :type feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param operations: Fill directions, which should be one of ['f', 'b', 'fb', 'bf'].
        :type operations: str
        :param value: Which value to fill by it's neighbors.
        :type value: any numpy dtype
        :param axis: An axis along which to fill values.
        :type axis: int
        """
        if operations not in ['f', 'b', 'fb', 'bf']:
            raise ValueError("'operations' parameter should be one of the following options: f, b, fb, bf.")

        self.feature = next(self._parse_features(feature)())
        self.operations = operations
        self.value = value
        self.axis = axis

    @staticmethod
    def fill(data, value=np.nan, operation='f'):
        """ Fills occurrences of a desired value in a 2d array with their neighbors in either forward or backward
        direction.

        :param data: A 2d numpy array.
        :type data: numpy.ndarray
        :param value: Which value to fill by it's neighbors.
        :type value: any numpy dtype
        :param operation: Fill directions, which should be either 'f' or 'b'.
        :type operation: str
        :return: Value-filled numpy array.
        :rtype: numpy.ndarray
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError('Wrong data input')

        if operation not in ['f', 'b']:
            raise ValueError("'operation' parameter should either be 'f' (forward) or 'b' (backward)!")

        value_match_mask = np.isnan(data) if np.isnan(value) else (data == value)
        value_match_true = 0 if operation == 'f' else (value_match_mask.shape[0] + 1)
        accumulate_function = np.maximum.accumulate if operation == 'f' else np.minimum.accumulate

        idx = np.where(value_match_mask, value_match_true, np.arange(value_match_mask.shape[1]))

        idx = idx if operation == 'f' else idx[:, ::-1]
        idx = accumulate_function(idx, axis=1)
        idx = idx if operation == 'f' else idx[:, ::-1]

        cannot_fill = np.all(idx == value_match_true, axis=1)
        idx[cannot_fill] = 0

        return data[np.arange(idx.shape[0])[:, np.newaxis], idx]

    def execute(self, eopatch):
        """
        :param eopatch: Source EOPatch from which to read the feature data.
        :type eopatch: EOPatch
        :return: An eopatch with the value-filled feature.
        :rtype: EOPatch
        """
        data = eopatch[self.feature]

        value_mask = np.isnan(data) if np.isnan(self.value) else (data == self.value)

        if not value_mask.any():
            return eopatch

        data = np.swapaxes(data, self.axis, -1)
        original_shape = data.shape
        data = data.reshape(np.prod(original_shape[:-1]), original_shape[-1])

        for operation in self.operations:
            data = self.fill(data, value=self.value, operation=operation)

        data = data.reshape(*original_shape)
        data = np.swapaxes(data, self.axis, -1)

        eopatch[self.feature] = data

        return eopatch
