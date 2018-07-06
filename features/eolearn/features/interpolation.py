""" Module for interpolating, smoothing and re-sampling features in EOPatch """

import numpy as np

from dateutil import parser
from datetime import timedelta
from scipy import interpolate

from eolearn.core import EOTask, EOPatch


class InterpolationTask(EOTask):
    """
    Main EOTask class for interpolation and resampling of time-series.

    The task takes from EOPatch the specified data feature and timestamps. For each pixel in the spatial grid it
    creates an interpolation model using values that are not NaN or masked with `eopatch.mask['VALID_DATA']`. Then
    it replaces invalid values using interpolation model. If ``resample_range`` parameter is used the values in
    time series will be resampled to new timestamps.

    In the process the interpolated feature is overwritten and so are the timestamps. After the execution of the task
    the feature will contain interpolated and resampled values and corresponding new timestamps.

    :param feature_name: Name of the feature in FeatureType.DATA which will be processed
    :type feature_name: str
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
    :param unknown_value: Value which will be used for timestamps where interpolation cannot be calculated
    :type unknown_value: float or numpy.nan
    :param interpolation_parameters: Parameters which will be propagated to ``interpolation_object``
    """
    def __init__(self, feature_name, interpolation_object, *, resample_range=None, result_interval=None,
                 unknown_value=np.nan, **interpolation_parameters):
        self.feature_name = feature_name
        self.interpolation_object = interpolation_object

        self.resample_range = resample_range
        self.result_interval = result_interval
        self.unknown_value = unknown_value
        self.interpolation_parameters = interpolation_parameters

        self._resampled_times = None

    def interpolate_data(self, data, times, resampled_times):
        """ Interpolates data feature

        :param data: Array in a shape of t x h x w x n
        :type data: numpy.ndarray
        :param times: Array of reference times in second relative to the first timestamp
        :type times: numpy.array
        :param resampled_times: Array of reference times in second relative to the first timestamp in initial timestamp
                                array.
        :type resampled_times: numpy.array
        :return: Array of interpolated values
        :rtype: numpy.ndarray
        """
        return np.apply_along_axis(self.interpolate_series, axis=0, arr=data, times=times,
                                   resampled_times=resampled_times)

    def interpolate_series(self, series, times, resampled_times):
        """ Interpolates time series

        :param series: One dimensional array of time series
        :type series: numpy.array
        :param times: Array of reference times in second relative to the first timestamp
        :type times: numpy.array
        :param resampled_times: Array of reference times in second relative to the first timestamp in initial timestamp
                                array.
        :type resampled_times: numpy.array
        :return: Array of interpolated values
        :rtype: numpy.array
        """
        valid_mask = ~np.isnan(series)
        interp_func = self.get_interpolation_function(times[valid_mask], series[valid_mask])

        start_time = np.min(times[valid_mask])
        end_time = np.max(times[valid_mask])

        new_series = series if self.resample_range is None else \
            np.full(resampled_times.shape, np.nan, dtype=series.dtype)
        time_mask = (resampled_times >= start_time) & (resampled_times <= end_time)
        new_series[time_mask] = interp_func(resampled_times[time_mask])
        return new_series

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
        if self.feature_name not in eopatch.data:
            raise ValueError('Feature {} not found in EOPatch.data.'.format(self.feature_name))

        feature_data = eopatch.data[self.feature_name]
        time_num, height, width, band_num = eopatch.data[self.feature_name].shape

        # Prepare mask of valid data
        if 'VALID_DATA' in eopatch.mask:
            feature_data[~eopatch.mask['VALID_DATA']] = np.nan

        # Flatten array
        feature_data = np.reshape(feature_data, (time_num, height * width * band_num))

        # If resampling create new EOPatch
        new_eopatch = EOPatch() if self.resample_range else eopatch

        # Resample times
        times = eopatch.time_series()
        start_time = eopatch.timestamp[0]
        new_eopatch.timestamp = self.get_resampled_timestamp(eopatch.timestamp)
        total_diff = int((new_eopatch.timestamp[0].date() - start_time.date()).total_seconds())
        resampled_times = new_eopatch.time_series() + total_diff

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
        new_eopatch.data[self.feature_name] = np.reshape(feature_data, (feature_data.shape[0], height, width, band_num))
        return new_eopatch


class LinearInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.interp1d(kind='linear')`
    """
    def __init__(self, feature_name, **kwargs):
        super(LinearInterpolation, self).__init__(feature_name, interpolate.interp1d, kind='linear', **kwargs)


class CubicInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.interp1d(kind='cubic')`
    """
    def __init__(self, feature_name, **kwargs):
        super(CubicInterpolation, self).__init__(feature_name, interpolate.interp1d, kind='cubic', **kwargs)


class SplineInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.UnivariateSpline`
    """
    def __init__(self, feature_name, *, spline_degree=3, smoothing_factor=0, **kwargs):
        super(SplineInterpolation, self).__init__(feature_name, interpolate.UnivariateSpline, k=spline_degree,
                                                  s=smoothing_factor, **kwargs)


class BSplineInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.BSpline`
    """
    def __init__(self, feature_name, *, spline_degree=3, **kwargs):
        super(BSplineInterpolation, self).__init__(feature_name, interpolate.BSpline, k=spline_degree, **kwargs)


class AkimaInterpolation(InterpolationTask):
    """
    Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.Akima1DInterpolator`
    """
    def __init__(self, feature_name, **kwargs):
        super(AkimaInterpolation, self).__init__(feature_name, interpolate.Akima1DInterpolator, **kwargs)


class ResamplingTask(InterpolationTask):
    """
    A subclass of InterpolationTask task that works much faster, works only on data with no missing, masked or invalid
    values and resamples timeseries to different timestamps
    """
    def __init__(self, feature_name, interpolation_object, resample_range, *, result_interval=None,
                 unknown_value=np.nan, **interpolation_parameters):
        if resample_range is None:
            raise ValueError("resample_range parameter must be in form ('start_date', 'end_date', step_days)")
        super(ResamplingTask, self).__init__(feature_name, interpolation_object, resample_range=resample_range,
                                             result_interval=result_interval, unknown_value=unknown_value,
                                             **interpolation_parameters)

    def interpolate_data(self, data, times, resampled_times):
        if True in np.unique(np.isnan(data)):
            raise ValueError('Data must not contain any masked/invalid pixels or NaN values')

        interp_func = self.get_interpolation_function(times, data)

        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))
        new_data = np.full((resampled_times.size,) + data.shape[1:], np.nan, dtype=data.dtype)
        new_data[time_mask] = interp_func(resampled_times[time_mask])
        return new_data

    def get_interpolation_function(self, times, data):
        return self.interpolation_object(times, data, axis=0, **self.interpolation_parameters)


class LinearResampling(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='linear')`
    """
    def __init__(self, feature_name, resample_range, **kwargs):
        super(LinearResampling, self).__init__(feature_name, interpolate.interp1d, resample_range, kind='linear',
                                               **kwargs)


class CubicResampling(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='cubic')`
    """
    def __init__(self, feature_name, resample_range, **kwargs):
        super(CubicResampling, self).__init__(feature_name, interpolate.interp1d, resample_range, kind='cubic',
                                              **kwargs)
