""" Module for interpolating, smoothing and re-sampling features in EOPatch """

import numpy as np

from dateutil import parser
from datetime import timedelta
from scipy import interpolate

from eolearn.core import EOTask, EOPatch, FeatureType


class InterpolationTask(EOTask):
    """
    Main EOTask class for interpolation and resampling of time-series.

    The task takes from EOPatch the specified data feature and timestamps. For each pixel in the spatial grid it
    creates an interpolation model using values that are not NaN or masked with `eopatch.mask['VALID_DATA']`. Then
    it replaces invalid values using interpolation model. If ``resample_range`` parameter is used the values in
    time series will be resampled to new timestamps.

    In the process the interpolated feature is overwritten and so are the timestamps. After the execution of the task
    the feature will contain interpolated and resampled values and corresponding new timestamps.

    :param feature: A feature to be interpolated with optional new feature name
    :type feature: (FeatureType, str) or (FeatureType, str, str)
    :param interpolation_object: Interpolation class which is initialized with
    :type interpolation_object: object
    :param resample_range: If None the data will be only interpolated over existing timestamps and NaN values will be
        replaced with interpolated values (if possible) in the existing EOPatch. Otherwise ``resample_range`` can be
        set to tuple in a form of (start_date, end_date, step_days), e.g. ('2018-01-01', '2018-06-01', 16). This will
        create a new EOPatch with resampled values for times start_date, start_date + step_days,
        start_date + 2 * step_days, ... . End date is excluded from timestamps.
    :type resample_range: (str, str, int) or None
    :param result_interval: Maximum and minimum of returned data
    :type result_interval: (float, float)
    :param mask_feature: Feature that contains binary masks of interpolated feature
    :type mask_feature: (FeatureType, str)
    :param unknown_value: Value which will be used for timestamps where interpolation cannot be calculated
    :type unknown_value: float or numpy.nan
    :param filling_factor: Multiplication factor used to create temporal gap between consecutive observations. Value
        has to be greater than 1. Default is `10`
    :type filling_factor: int
    :param scale_time: Factor used to scale the time difference in seconds between acquisitions. If `scale_time=60`,
        returned time is in minutes, if `scale_time=3600` in hours. Default is `3600`
    :type scale_time: int
    :param interpolation_parameters: Parameters which will be propagated to ``interpolation_object``
    """
    def __init__(self, feature, interpolation_object, *, resample_range=None, result_interval=None, mask_feature=None,
                 unknown_value=np.nan, filling_factor=10, scale_time=3600,
                 **interpolation_parameters):
        self.feature = self._parse_features(feature, new_names=True, default_feature_type=FeatureType.DATA)
        self.interpolation_object = interpolation_object

        self.resample_range = resample_range
        self.result_interval = result_interval

        self.mask_feature = None if mask_feature is None else \
            self._parse_features(mask_feature, default_feature_type=FeatureType.MASK)

        self.unknown_value = unknown_value
        self.interpolation_parameters = interpolation_parameters
        self.scale_time = scale_time
        self.filling_factor = filling_factor

        self._resampled_times = None

    @staticmethod
    def _get_start_end_nans(data):
        """ Find NaN values in data that either start or end the time-series

        Function to return a binary array of same size as data where `True` values correspond to NaN values present at
        beginning or end of time-series. NaNs internal to the time-series are not included in the binary mask.

        :param data: Array of observations of size TxNOBS
        :type data: numpy.array
        :return: Binary array of shape TxNOBS. `True` values indicate NaNs present at beginning or end of time-series
        :rtype: numpy.array
        """
        # find NaNs that start a time-series
        start_nan = np.isnan(data)
        for idx, row in enumerate(start_nan[:-1]):
            start_nan[idx+1] = np.logical_and(row, start_nan[idx+1])
        # find NaNs that end a time-series
        end_nan = np.isnan(data)
        for idx, row in enumerate(end_nan[-2::-1]):
            end_nan[-idx-2] = np.logical_and(row, end_nan[-idx-1])

        return np.logical_or(start_nan, end_nan)

    def interpolate_data(self, data, times, resampled_times):
        """ Interpolates data feature

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :type data: numpy.ndarray
        :param times: Array of reference times relative to the first timestamp
        :type times: numpy.array
        :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
        :type resampled_times: numpy.array
        :return: Array of interpolated values
        :rtype: numpy.ndarray
        """
        # get size of 2d array t x nobs
        ntimes, nobs = data.shape

        # mask representing overlap between reference and resampled times
        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))

        # define time values as linear mono-tonically increasing over the observations
        const = int(self.filling_factor * (np.max(times) - np.min(times)))
        temp_values = (times[:, np.newaxis] + const * np.arange(nobs)[np.newaxis, :]).astype(np.float64)
        res_temp_values = (resampled_times[:, np.newaxis] + const * np.arange(nobs)[np.newaxis, :]).astype(np.float64)

        # initialise array of interpolated values
        new_data = data if self.resample_range is None else np.full((len(resampled_times), nobs),
                                                                    np.nan, dtype=data.dtype)
        # array defining index correspondence between reference times and resampled times
        ori2res = np.arange(ntimes, dtype=np.int32) if self.resample_range is None else np.array(
            [np.abs(resampled_times - o).argmin()
             if np.min(resampled_times) <= o <= np.max(resampled_times) else None for o in times])

        # find NaNs that start or end a time-series
        row_nans, col_nans = np.where(self._get_start_end_nans(data))
        nan_row_res_indices = np.array([index for index in ori2res[row_nans] if index is not None], dtype=np.int32)
        nan_col_res_indices = np.array([True if index is not None else False for index in ori2res[row_nans]],
                                       dtype=np.bool)
        if nan_row_res_indices.size:
            # mask out from output values the starting/ending NaNs
            res_temp_values[nan_row_res_indices, col_nans[nan_col_res_indices]] = np.nan
        # if temporal values outside the reference dates are required (extrapolation) masked them to NaN
        res_temp_values[~time_mask, :] = np.nan

        # build 1d array for interpolation. Spline functions require monotonically increasing values of x, so .T is used
        input_x = temp_values.T[~np.isnan(data).T]
        input_y = data.T[~np.isnan(data).T]

        # build interpolation function
        interp_func = self.get_interpolation_function(input_x, input_y)

        # interpolate non-NaN values in resampled time values
        new_data[~np.isnan(res_temp_values)] = interp_func(res_temp_values[~np.isnan(res_temp_values)])

        # return interpolated values
        return new_data

    def get_interpolation_function(self, times, series):
        """ Initializes interpolation model

        :param times: Array of reference times in second relative to the first timestamp
        :type times: numpy.array
        :param series: One dimensional array of time series
        :type series: numpy.array
        :return: Initialized interpolation model class
        """
        return self.interpolation_object(times, series, **self.interpolation_parameters)

    def get_resampled_timestamp(self, timestamp):
        """ Takes a list of timestamps and generates new list of timestamps according to ``resample_range``

        :param timestamp: list of timestamps
        :type timestamp: list(datetime.datetime)
        :return: new list of timestamps
        :rtype: list(datetime.datetime)
        """
        if self.resample_range is None:
            return timestamp
        if not isinstance(self.resample_range, (tuple, list)):
            raise ValueError('Invalid resample_range {}, expected tuple'.format(self.resample_range))

        start_date = parser.parse(self.resample_range[0])
        end_date = parser.parse(self.resample_range[1])
        step = timedelta(days=self.resample_range[2])
        days = [start_date]
        while days[-1] + step < end_date:
            days.append(days[-1] + step)
        return days

    def execute(self, eopatch):
        """ Execute method that processes EOPatch and returns EOPatch
        """
        feature_type, feature_name, new_feature_name = next(self.feature(eopatch))

        # Make a copy not to change original numpy array
        feature_data = eopatch[feature_type][feature_name].copy()
        time_num, height, width, band_num = feature_data.shape

        # Prepare mask of valid data
        if self.mask_feature:
            mask_type, mask_name = next(self.mask_feature(eopatch))
            feature_data[~eopatch[mask_type][mask_name].squeeze(), :] = np.nan

        # Flatten array
        feature_data = np.reshape(feature_data, (time_num, height * width * band_num))

        # If resampling create new EOPatch
        new_eopatch = EOPatch() if self.resample_range else eopatch

        # Resample times
        times = eopatch.time_series(scale_time=self.scale_time)
        new_eopatch.timestamp = self.get_resampled_timestamp(eopatch.timestamp)
        total_diff = int((new_eopatch.timestamp[0].date() - eopatch.timestamp[0].date()).total_seconds())
        resampled_times = new_eopatch.time_series(scale_time=self.scale_time) + total_diff // self.scale_time

        # Add BBox to eopatch if it was created anew
        if new_eopatch.bbox is None:
            new_eopatch.bbox = eopatch.bbox

        # Interpolate
        feature_data = self.interpolate_data(feature_data, times, resampled_times)

        # Normalize
        if self.result_interval:
            min_val, max_val = self.result_interval
            valid_mask = ~np.isnan(feature_data)
            feature_data[valid_mask] = np.maximum(np.minimum(feature_data[valid_mask], max_val), min_val)

        # Replace unknown value
        if not np.isnan(self.unknown_value):
            feature_data[np.isnan(feature_data)] = self.unknown_value

        # Reshape back
        new_eopatch[feature_type][new_feature_name] = np.reshape(feature_data,
                                                                 (feature_data.shape[0], height, width, band_num))
        return new_eopatch


class LinearInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.interp1d(kind='linear')`
    """
    def __init__(self, feature, **kwargs):
        super().__init__(feature, interpolate.interp1d, kind='linear', **kwargs)


class CubicInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.interp1d(kind='cubic')`
    """
    def __init__(self, feature, **kwargs):
        super().__init__(feature, interpolate.interp1d, kind='cubic', **kwargs)


class SplineInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.UnivariateSpline`
    """
    def __init__(self, feature, *, spline_degree=3, smoothing_factor=0, **kwargs):
        super().__init__(feature, interpolate.UnivariateSpline, k=spline_degree, s=smoothing_factor, **kwargs)


class BSplineInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.BSpline`
    """
    def __init__(self, feature, *, spline_degree=3, **kwargs):
        super().__init__(feature, interpolate.make_interp_spline, k=spline_degree, **kwargs)


class AkimaInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.Akima1DInterpolator`
    """
    def __init__(self, feature, **kwargs):
        super().__init__(feature, interpolate.Akima1DInterpolator, **kwargs)


class ResamplingTask(InterpolationTask):
    """
    A subclass of InterpolationTask task that works only with data with no missing, masked or
    invalid values. It always resamples timeseries to different timestamps.
    """
    def __init__(self, feature, interpolation_object, resample_range, *, result_interval=None,
                 unknown_value=np.nan, **interpolation_parameters):
        if resample_range is None:
            raise ValueError("resample_range parameter must be in form ('start_date', 'end_date', step_days)")
        super().__init__(feature, interpolation_object, resample_range=resample_range,
                         result_interval=result_interval, unknown_value=unknown_value, **interpolation_parameters)

    def interpolate_data(self, data, times, resampled_times):
        """ Interpolates data feature

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :type data: numpy.ndarray
        :param times: Array of reference times in second relative to the first timestamp
        :type times: numpy.array
        :param resampled_times: Array of reference times in second relative to the first timestamp in initial timestamp
                                array.
        :type resampled_times: numpy.array
        :return: Array of interpolated values
        :rtype: numpy.ndarray
        """
        if True in np.unique(np.isnan(data)):
            raise ValueError('Data must not contain any masked/invalid pixels or NaN values')

        interp_func = self.get_interpolation_function(times, data)

        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))
        new_data = np.full((resampled_times.size,) + data.shape[1:], np.nan, dtype=data.dtype)
        new_data[time_mask] = interp_func(resampled_times[time_mask])
        return new_data

    def get_interpolation_function(self, times, data):
        """ Initializes interpolation model

        :param times: Array of reference times in second relative to the first timestamp
        :type times: numpy.array
        :param data: One dimensional array of time series
        :type data: numpy.array
        :return: Initialized interpolation model class
        """
        return self.interpolation_object(times, data, axis=0, **self.interpolation_parameters)


class NearestResampling(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='nearest')`
    """
    def __init__(self, feature, resample_range, **kwargs):
        super().__init__(feature, interpolate.interp1d, resample_range, kind='nearest', **kwargs)


class LinearResampling(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='linear')`
    """
    def __init__(self, feature, resample_range, **kwargs):
        super().__init__(feature, interpolate.interp1d, resample_range, kind='linear', **kwargs)


class CubicResampling(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='cubic')`
    """
    def __init__(self, feature, resample_range, **kwargs):
        super().__init__(feature, interpolate.interp1d, resample_range, kind='cubic', **kwargs)
