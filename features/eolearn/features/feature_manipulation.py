"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.
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
