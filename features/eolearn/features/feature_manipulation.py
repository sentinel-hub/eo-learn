"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import datetime as dt
import logging
from typing import Optional

import numpy as np

from eolearn.core import EOTask, FeatureType, MapFeatureTask

LOGGER = logging.getLogger(__name__)


class SimpleFilterTask(EOTask):
    """
    Transforms an eopatch of shape [n, w, h, d] into [m, w, h, d] for m <= n. It removes all slices which don't
    conform to the filter_func.

    A filter_func is a callable which takes a numpy array and returns a bool.
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
        self.feature = self.parse_feature(feature)
        self.filter_func = filter_func
        self.filter_features_parser = self.get_feature_parser(filter_features)

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
        good_idxs = self._get_filtered_indices(eopatch[self.feature])
        if not good_idxs:
            raise RuntimeError("EOPatch has no good indices after filtering with given filter function")

        for feature_type, feature_name in self.filter_features_parser.get_features(eopatch):
            if feature_type.is_temporal():
                if feature_type.has_dict():
                    if feature_type.contains_ndarrays():
                        eopatch[feature_type][feature_name] = np.asarray(
                            [eopatch[feature_type][feature_name][idx] for idx in good_idxs]
                        )
                    # else:
                    #     NotImplemented
                else:
                    eopatch[feature_type] = [eopatch[feature_type][idx] for idx in good_idxs]

        self._update_other_data(eopatch)

        return eopatch


class FilterTimeSeriesTask(SimpleFilterTask):
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
            raise ValueError("Start date is not of correct type. Please provide the start_date as datetime.datetime.")

        if not isinstance(end_date, dt.datetime):
            raise ValueError("End date is not of correct type. Please provide the end_date as datetime.datetime.")

        super().__init__(FeatureType.TIMESTAMP, lambda date: start_date <= date <= end_date, filter_features)


class ValueFilloutTask(EOTask):
    """Overwrites occurrences of a desired value with their neighbor values in either forward, backward direction or
    both, along an axis.

    Possible fillout operations are 'f' (forward), 'b' (backward) or both, 'fb' or 'bf':

        'f': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> nan, nan, nan, 8, 5, 5, 1, 0, 0, 0

        'b': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 1, 1, 0, nan, nan

        'fb': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 5, 1, 0, 0, 0

        'bf': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 1, 1, 0, 0, 0
    """

    def __init__(self, feature, operations="fb", value=np.nan, axis=0):
        """
        :param feature: A feature that must be value-filled.
        :type feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param operations: Fill directions, which should be one of ['f', 'b', 'fb', 'bf'].
        :type operations: str
        :param value: Which value to fill by its neighbors.
        :type value: any numpy dtype
        :param axis: An axis along which to fill values.
        :type axis: int
        """
        if operations not in ["f", "b", "fb", "bf"]:
            raise ValueError("'operations' parameter should be one of the following options: f, b, fb, bf.")

        self.feature = self.parse_feature(feature)
        self.operations = operations
        self.value = value
        self.axis = axis

    @staticmethod
    def fill(data, value=np.nan, operation="f"):
        """Fills occurrences of a desired value in a 2d array with their neighbors in either forward or backward
        direction.

        :param data: A 2d numpy array.
        :type data: numpy.ndarray
        :param value: Which value to fill by its neighbors.
        :type value: any numpy dtype
        :param operation: Fill directions, which should be either 'f' or 'b'.
        :type operation: str
        :return: Value-filled numpy array.
        :rtype: numpy.ndarray
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Wrong data input")

        if operation not in ["f", "b"]:
            raise ValueError("'operation' parameter should either be 'f' (forward) or 'b' (backward)!")

        n_rows, n_frames = data.shape

        value_mask = np.isnan(data) if np.isnan(value) else (data == value)
        init_index = 0 if operation == "f" else (n_frames - 1)

        idx = np.where(value_mask, init_index, np.arange(n_frames))

        if operation == "f":
            idx = np.maximum.accumulate(idx, axis=1)
        else:
            idx = idx[:, ::-1]
            idx = np.minimum.accumulate(idx, axis=1)
            idx = idx[:, ::-1]

        return data[np.arange(n_rows)[:, np.newaxis], idx]

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


class LinearFunctionTask(MapFeatureTask):
    """Applies a linear function to the values of input features.

    Each value in the feature is modified as `x -> x * slope + intercept`. The `dtype` of the result can be customized.
    """

    def __init__(
        self, input_features, output_features=None, slope: float = 1, intercept: float = 0, dtype: Optional[str] = None
    ):
        """
        :param input_features: Feature or features on which the function is used.
        :param output_features: Feature or features for saving the result. If not provided the input_features are
            overwritten.
        :param slope: Slope of the function i.e. the multiplication factor.
        :param intercept: Intercept of the function i.e. the value added.
        :param dtype: Numpy dtype of the output feature. If not provided the dtype is determined by Numpy, so it is
            recommended to set manually.
        """
        if output_features is None:
            output_features = input_features
        super().__init__(input_features, output_features, slope=slope, intercept=intercept, dtype=dtype)

    def map_method(self, feature: np.ndarray, slope: float, intercept: float, dtype: Optional[str]) -> np.ndarray:
        """A method where feature is multiplied by a slope"""
        rescaled_feature = feature * slope + intercept
        return rescaled_feature if dtype is None else rescaled_feature.astype(dtype)
