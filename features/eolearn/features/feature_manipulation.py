"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.
"""

from eolearn.core import EOTask, FeatureType
import numpy as np

from datetime import datetime

import logging
LOGGER = logging.getLogger(__name__)


class SimpleFilterTask(EOTask):
    """
    Transforms an eopatch of shape [n, w, h, d] into [m, w, h, d] for m <= n. It removes all slices which don't
    conform to the predicate.

    A predicate is a callable which takes an numpy array and returns a bool.
    """
    def __init__(self, predicate, attr_type, field):
        """
        :param predicate: A callable that takes an eopatch instance and evaluates to bool.
        :param attr_type: A constant from FeatureType
        :param field: The name of the field in the dictionary corresponding to attr_type
        """
        self.predicate = predicate
        self.attr_type = attr_type
        self.field = field

    def execute(self, eopatch):
        """
        :param eopatch: Input eopatch.
        :type eopatch: EOPatch
        :return: Transformed eo patch
        :rtype: EOPatch
        """
        return self.do_transform(eopatch, self.attr_type.value, self.field)

    def do_transform(self, eopatch, attr_name, field):
        good_idxs = [idx for idx, img in enumerate(eopatch[attr_name][field]) if self.predicate(img)]

        LOGGER.debug("good_idxs: %s", good_idxs)

        time_dependent = [FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL]

        eopatch.timestamp = [eopatch.timestamp[idx] for idx in good_idxs]

        for attr_type in time_dependent:
            attr = getattr(eopatch, attr_type.value)
            assert isinstance(attr, dict)
            for target_field in attr:
                value = attr[target_field]
                eopatch.add_feature(attr_type=attr_type, field=target_field,
                                    value=np.asarray([value[idx] for idx in good_idxs]))

        return eopatch


class FilterTimeSeries(EOTask):
    """
    Removes all frames in the time-series with dates outside the user specified time interval.
    """
    def __init__(self, start_date, end_date):
        """
        :param start_date: Start date. All frames within the time-series taken after this date will be kept.
        :type start_date: datetime.datetime
        :param end_date: End date. All frames within the time-series taken before this date will be kept.
        :type end_date: datetime.datetime
        """
        self.start_date = start_date
        self.end_date = end_date

        if not isinstance(start_date, datetime):
            raise ValueError('Start date is not of correct type. Please provide the start_date as datetime.datetime.')

        if not isinstance(end_date, datetime):
            raise ValueError('End date is not of correct type. Please provide the end_date as datetime.datetime.')

    def execute(self, eopatch):
        good_idxs = []
        for idx, date in enumerate(eopatch.timestamp):
            diff_to_begin = (date - self.start_date)
            diff_to_end = (date - self.end_date)

            if (diff_to_begin.total_seconds() > 0) and (diff_to_end.total_seconds() < 0):
                good_idxs.append(idx)

        LOGGER.debug("good_idxs: %s", good_idxs)

        time_dependent = [FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL]

        eopatch.timestamp = [eopatch.timestamp[idx] for idx in good_idxs]

        for attr_type in time_dependent:
            attr = getattr(eopatch, attr_type.value)
            assert isinstance(attr, dict)
            for target_field in attr:
                value = attr[target_field]
                eopatch.add_feature(attr_type=attr_type, field=target_field,
                                    value=np.asarray([value[idx] for idx in good_idxs]))

        return eopatch
