"""
Module for interpolating, smoothing and re-sampling features in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime as dt
import inspect
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Iterable, List, Tuple, Union, cast

import dateutil
import numpy as np
import scipy.interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from typing_extensions import deprecated

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.exceptions import EODeprecationWarning, EOUserWarning
from eolearn.core.types import FeaturesSpecification, SingleFeatureSpec
from eolearn.core.utils.parsing import parse_renamed_feature

try:
    import numba
except ImportError as exception:
    warnings.warn(
        f"Failed to import numba with exception: '{exception}'. Some interpolation tasks won't work", EOUserWarning
    )

ResampleRangeType = Union[None, Tuple[str, str, int], List[str], List[dt.datetime]]


def base_interpolation_function(data: np.ndarray, times: np.ndarray, resampled_times: np.ndarray) -> np.ndarray:
    """Interpolates data feature

    :param data: Array in a shape of t x (h x w x n)
    :param times: Array of reference times relative to the first timestamp
    :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
    :return: Array of interpolated values
    """

    _, height_width_depth = data.shape
    new_bands = np.empty((len(resampled_times), height_width_depth))
    for n_feat in numba.prange(height_width_depth):  # pylint: disable=not-an-iterable
        mask1d = ~np.isnan(data[:, n_feat])
        if not mask1d.any():
            new_data = np.empty(len(resampled_times))
            new_data[:] = np.nan
        else:
            new_data = np.interp(
                resampled_times.astype(np.float64),
                times[mask1d].astype(np.float64),
                data[:, n_feat][mask1d].astype(np.float64),
            )

            true_index = np.where(mask1d)
            index_first, index_last = true_index[0][0], true_index[0][-1]
            min_time, max_time = times[index_first], times[index_last]
            first = np.where(resampled_times < min_time)[0]
            if first.size:
                new_data[: first[-1] + 1] = np.nan
            last = np.where(max_time < resampled_times)[0]
            if last.size:
                new_data[last[0] :] = np.nan

        new_bands[:, n_feat] = new_data.astype(data.dtype)

    return new_bands


try:
    # pylint: disable=invalid-name
    interpolation_function = numba.njit(base_interpolation_function)
    interpolation_function_parallel = numba.njit(base_interpolation_function, parallel=True)
except NameError:
    pass


class InterpolationTask(EOTask):
    """Main EOTask class for interpolation and resampling of time-series.

    The task takes from EOPatch the specified data feature and timestamps. For each pixel in the spatial grid it
    creates an interpolation model using values that are not NaN or masked with `eopatch.mask['VALID_DATA']`. Then
    it replaces invalid values using interpolation model. If ``resample_range`` parameter is used the values in
    time series will be resampled to new timestamps.

    In the process the interpolated feature is overwritten and so are the timestamps. After the execution of the task
    the feature will contain interpolated and resampled values and corresponding new timestamps.

    Examples of `interpolation_object:
    - `scipy.interpolate.interp1d`, supply the kind as a kwarg, e.g. `kind="cubic"`
    - `scipy.interpolate.UnivariateSpline`
    - `scipy.interpolate.make_interp_spline`
    - `scipy.interpolate.Akima1DInterpolator`

    :param feature: A feature to be interpolated with optional new feature name
    :param interpolation_object: Interpolation class which is initialized with `interpolation_parameters`
    :param resample_range: If None the data will be only interpolated over existing timestamps and NaN values will be
        replaced with interpolated values (if possible) in the existing EOPatch. Otherwise, `resample_range` can be
        set to tuple in a form of (start_date, end_date, step_days), e.g. ('2018-01-01', '2018-06-01', 16). This will
        create a new EOPatch with resampled values for times start_date, start_date + step_days,
        start_date + 2 * step_days, ... . End date is excluded from timestamps. Additionally, ``resample_range`` can
        be a list of dates or date-strings where the interpolation will be evaluated.
    :param result_interval: Maximum and minimum of returned data
    :param mask_feature: A mask feature which will be used to mask certain features
    :param copy_features: List of tuples of type (FeatureType, str) or (FeatureType, str, str) that are copied
        over into the new EOPatch. The first string is the feature name, and the second one (optional) is a new name
        to be used for the feature
    :param unknown_value: Value which will be used for timestamps where interpolation cannot be calculated
    :param filling_factor: Multiplication factor used to create temporal gap between consecutive observations. Value
        has to be greater than 1. Default is `10`
    :param scale_time: Factor used to scale the time difference in seconds between acquisitions. If `scale_time=60`,
        returned time is in minutes, if `scale_time=3600` in hours. Default is `3600`
    :param interpolate_pixel_wise: Flag to indicate pixel wise interpolation or fast interpolation that creates a single
        interpolation object for the whole image
    :param interpolation_parameters: Parameters which will be propagated to `interpolation_object`
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        interpolation_object: Callable,
        *,
        resample_range: ResampleRangeType = None,
        result_interval: tuple[float, float] | None = None,
        mask_feature: SingleFeatureSpec | None = None,
        copy_features: FeaturesSpecification | None = None,
        unknown_value: float = np.nan,
        filling_factor: int = 10,
        scale_time: int = 3600,
        interpolate_pixel_wise: bool = False,
        **interpolation_parameters: Any,
    ):
        self.renamed_feature = parse_renamed_feature(
            feature, allowed_feature_types=[FeatureType.MASK, FeatureType.DATA]
        )

        self.interpolation_object = interpolation_object
        self.resample_range = resample_range
        self.result_interval = result_interval

        self.mask_feature_parser = None
        if mask_feature is not None:
            self.mask_feature_parser = self.get_feature_parser(
                mask_feature, allowed_feature_types={FeatureType.MASK, FeatureType.MASK_TIMELESS, FeatureType.LABEL}
            )

        if resample_range is None and copy_features is not None:
            self.copy_features = None
            warnings.warn(
                "If `resample_range` is None the task is done in-place. Ignoring `copy_features`.", EOUserWarning
            )
        else:
            self.copy_features_parser = None if copy_features is None else self.get_feature_parser(copy_features)

        self.unknown_value = unknown_value
        self.interpolation_parameters = interpolation_parameters
        self.scale_time = scale_time
        self.filling_factor = filling_factor
        self.interpolate_pixel_wise = interpolate_pixel_wise

    @staticmethod
    def _mask_feature_data(feature_data: np.ndarray, mask: np.ndarray, mask_type: FeatureType) -> np.ndarray:
        """Masks values of data feature (in-place) with a given mask by assigning `numpy.nan` value to masked fields."""

        spatial_dim_wrong = mask_type.is_spatial() and feature_data.shape[1:3] != mask.shape[-3:-1]
        temporal_dim_wrong = mask_type.is_temporal() and feature_data.shape[0] != mask.shape[0]
        if spatial_dim_wrong or temporal_dim_wrong:
            raise ValueError(
                f"Dimensions of interpolation data {feature_data.shape} and mask {mask.shape} do not match."
            )

        # This allows masking each channel differently but causes some complications while masking with label
        if mask.shape[-1] != feature_data.shape[-1]:
            mask = mask[..., 0]

        if mask_type is FeatureType.MASK:
            feature_data[mask, ...] = np.nan

        elif mask_type is FeatureType.MASK_TIMELESS:
            feature_data[:, mask, ...] = np.nan

        elif mask_type is FeatureType.LABEL:
            np.swapaxes(feature_data, 1, 3)
            feature_data[mask, ..., :, :] = np.nan
            np.swapaxes(feature_data, 1, 3)

        return feature_data

    @staticmethod
    def _get_start_end_nans(data: np.ndarray) -> np.ndarray:
        """Find NaN values in data that either start or end the time-series

        Function returns a array of same size as data where `True` corresponds to NaN values present at
        beginning or end of time-series. NaNs internal to the time-series are not included in the binary mask.

        :param data: Array of observations of size t x num_obs
        :return: Array of shape t x num_obs. `True` values indicate NaNs present at beginning or end of time-series
        """
        # find NaNs that start a time-series
        start_nan = np.isnan(data)
        for idx, row in enumerate(start_nan[:-1]):
            start_nan[idx + 1] = np.logical_and(row, start_nan[idx + 1])
        # find NaNs that end a time-series
        end_nan = np.isnan(data)
        for idx, row in enumerate(end_nan[-2::-1]):
            end_nan[-idx - 2] = np.logical_and(row, end_nan[-idx - 1])

        return np.logical_or(start_nan, end_nan)

    @staticmethod
    def _get_unique_times(data: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace duplicate acquisitions which have same values on the chosen timescale with their average.
        The average is calculated with numpy.nanmean, meaning that NaN values are ignored when calculating the average.

        :param data: Array in a shape of t x num_obs, where num_obs = h x w x n
        :param times: Array of reference times relative to the first timestamp
        :return: cleaned versions of data input
        """
        time_groups = defaultdict(list)
        for idx, time in enumerate(times):
            time_groups[time].append(data[idx])

        clean_times = np.array(sorted(time_groups))
        clean_data = np.full((len(clean_times), *data.shape[1:]), np.nan)
        for idx, time in enumerate(clean_times):
            # np.nanmean complains about rows of full nans, so we have to use masking, makes more complicated
            data_for_time = np.array(time_groups[time])
            nan_mask = np.all(np.isnan(data_for_time), axis=0)
            clean_data[idx, ~nan_mask] = np.nanmean(data_for_time[:, ~nan_mask], axis=0)

        return clean_data, clean_times

    def _copy_old_features(self, new_eopatch: EOPatch, old_eopatch: EOPatch) -> EOPatch:
        """Copy features from old EOPatch into new_eopatch"""
        if self.copy_features_parser is not None:
            for ftype, fname, new_fname in self.copy_features_parser.get_renamed_features(old_eopatch):
                if (ftype, new_fname) in new_eopatch:
                    raise ValueError(f"Feature {new_fname} of {ftype} already exists in the new EOPatch!")

                new_eopatch[ftype, new_fname] = old_eopatch[ftype, fname]

        return new_eopatch

    def interpolate_data(self, data: np.ndarray, times: np.ndarray, resampled_times: np.ndarray) -> np.ndarray:
        """Interpolates data feature

        :param data: Array in a shape of t x num_obs, where num_obs = h x w x n
        :param times: Array of reference times relative to the first timestamp
        :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
        :return: Array of interpolated values
        """
        # pylint: disable=too-many-locals
        num_obs = data.shape[-1]
        new_data = np.full((len(resampled_times), num_obs), np.nan, dtype=data.dtype)

        if self.interpolate_pixel_wise:
            for idx in range(num_obs):
                tseries = data[:, idx]
                valid = ~np.isnan(tseries)
                obs_interpolating_func = self.get_interpolation_function(times[valid], tseries[valid])

                new_data[:, idx] = obs_interpolating_func(resampled_times[:, np.newaxis])

            return new_data

        # array defining index correspondence between reference times and resampled times
        min_time, max_time = np.min(resampled_times), np.max(resampled_times)
        ori2res = np.array([
            np.abs(resampled_times - orig_time).argmin() if min_time <= orig_time <= max_time else None
            for orig_time in times
        ])

        # find NaNs that start or end a time-series
        row_nans, col_nans = np.where(self._get_start_end_nans(data))
        nan_row_res_indices = np.array([index for index in ori2res[row_nans] if index is not None], dtype=np.int32)
        nan_col_res_indices = np.array([index is not None for index in ori2res[row_nans]], dtype=bool)

        # define time values as linear monotonically increasing over the observations
        const = int(self.filling_factor * (np.max(times) - np.min(times)))
        temp_values = times[:, np.newaxis] + const * np.arange(num_obs)[np.newaxis, :].astype(np.float64)
        res_temp_values = resampled_times[:, np.newaxis] + const * np.arange(num_obs)[np.newaxis, :].astype(np.float64)

        if nan_row_res_indices.size:
            # mask out from output values the starting/ending NaNs
            res_temp_values[nan_row_res_indices, col_nans[nan_col_res_indices]] = np.nan
        # if temporal values outside the reference dates are required (extrapolation) masked them to NaN
        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))
        res_temp_values[~time_mask, :] = np.nan

        # build 1d array for interpolation. Spline functions require monotonically increasing values of x, so .T is used
        input_x = temp_values.T[~np.isnan(data).T]
        input_y = data.T[~np.isnan(data).T]

        # build interpolation function
        if len(input_x) > 1:
            interp_func = self.get_interpolation_function(input_x, input_y)
            valid = ~np.isnan(res_temp_values)
            new_data[valid] = interp_func(res_temp_values[valid])

        return new_data

    def get_interpolation_function(self, times: np.ndarray, series: np.ndarray) -> Callable:
        """Initializes interpolation model

        :param times: Array of reference times in second relative to the first timestamp
        :param series: One dimensional array of time series
        :return: Initialized interpolation model class
        """
        if str(inspect.getmodule(self.interpolation_object))[9:14] == "numpy":
            return partial(self.interpolation_object, xp=times, fp=series, left=np.nan, right=np.nan)
        return self.interpolation_object(times, series, **self.interpolation_parameters)

    def get_resampled_timestamp(self, timestamps: list[dt.datetime]) -> list[dt.datetime]:
        """Takes a list of timestamps and generates new list of timestamps according to `resample_range`"""
        if self.resample_range is None:
            return timestamps

        if not isinstance(self.resample_range, (tuple, list)):
            raise ValueError(f"Invalid resample_range {self.resample_range}, expected tuple")

        if tuple(map(type, self.resample_range)) == (str, str, int):
            start_str, end_str, step_size = cast(Tuple[str, str, int], self.resample_range)
            start_date, end_date = dateutil.parser.parse(start_str), dateutil.parser.parse(end_str)
            step = dt.timedelta(days=step_size)
            days = [start_date]
            while days[-1] + step < end_date:
                days.append(days[-1] + step)

            return days

        if isinstance(self.resample_range, (list, tuple)):
            dates = cast(Iterable[Union[str, dt.datetime]], self.resample_range)
            return [dateutil.parser.parse(date) if isinstance(date, str) else date for date in dates]

        raise ValueError(f"Invalid format in {self.resample_range}, expected {ResampleRangeType}")

    @staticmethod
    def _get_eopatch_time_series(
        eopatch: EOPatch, ref_date: dt.datetime | None = None, scale_time: int = 1
    ) -> np.ndarray:
        """Returns a numpy array with seconds passed between the reference date and the timestamp of each image.

        An array is constructed as time_series[i] = (timestamp[i] - ref_date).total_seconds().
        If reference date is None the first date in the EOPatch's timestamp array is taken.
        If EOPatch `timestamps` attribute is empty the method returns None.

        :param eopatch: the EOPatch whose timestamps are used to construct the time series
        :param ref_date: reference date relative to which the time is measured
        :param scale_time: scale seconds by factor. If `60`, time will be in minutes, if `3600` hours
        """
        if not eopatch.timestamps:
            return np.zeros(0, dtype=np.int64)

        ref_date = ref_date or eopatch.timestamps[0]

        return np.asarray(
            [round((timestamp - ref_date).total_seconds() / scale_time) for timestamp in eopatch.timestamps],
            dtype=np.int64,
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute method that processes EOPatch and returns EOPatch"""
        # pylint: disable=too-many-locals
        feature_type, feature_name, new_feature_name = self.renamed_feature

        # Make a copy not to change original numpy array
        feature_data = eopatch[feature_type, feature_name].copy()
        time_num, height, width, band_num = feature_data.shape
        if time_num <= 1:
            raise ValueError(
                f"Feature {(feature_type, feature_name)} has temporal dimension {time_num}, required at least size 2"
            )

        # Apply a mask on data
        if self.mask_feature_parser is not None:
            for mask_type, mask_name in self.mask_feature_parser.get_features(eopatch):
                negated_mask = ~eopatch[mask_type, mask_name].astype(bool)
                feature_data = self._mask_feature_data(feature_data, negated_mask, mask_type)

        # If resampling create new EOPatch
        new_eopatch = EOPatch(bbox=eopatch.bbox) if self.resample_range else eopatch

        # Resample times
        times = self._get_eopatch_time_series(eopatch, scale_time=self.scale_time)
        new_eopatch.timestamps = self.get_resampled_timestamp(eopatch.get_timestamps())
        total_diff = int((new_eopatch.get_timestamps()[0].date() - eopatch.get_timestamps()[0].date()).total_seconds())
        resampled_times = (
            self._get_eopatch_time_series(new_eopatch, scale_time=self.scale_time) + total_diff // self.scale_time
        )

        # Flatten array
        feature_data = np.reshape(feature_data, (time_num, height * width * band_num))

        # Replace duplicate acquisitions which have same values on the chosen timescale with their average
        feature_data, times = self._get_unique_times(feature_data, times)

        # Interpolate
        feature_data = self.interpolate_data(feature_data, times, resampled_times)

        # Normalize and insert correct unknown value
        if self.result_interval:
            feature_data = np.clip(feature_data, *self.result_interval)
        feature_data[np.isnan(feature_data)] = self.unknown_value

        new_eopatch[feature_type, new_feature_name] = np.reshape(feature_data, (-1, height, width, band_num))

        return self._copy_old_features(new_eopatch, eopatch)


class LinearInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `numpy.interp` and `@numba.jit(nopython=True)`

    :param parallel: interpolation is calculated in parallel by numba.
    :param kwargs: parameters of InterpolationTask(EOTask)
    """

    def __init__(self, feature: SingleFeatureSpec, parallel: bool = False, **kwargs: Any):
        self.parallel = parallel
        super().__init__(feature, np.interp, **kwargs)

    def interpolate_data(self, data: np.ndarray, times: np.ndarray, resampled_times: np.ndarray) -> np.ndarray:
        """Interpolates data feature

        :param data: Array in a shape of t x num_obs, where num_obs = h x w x n
        :param times: Array of reference times in second relative to the first timestamp
        :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
        :return: Array of interpolated values
        """
        if self.parallel:
            return interpolation_function_parallel(data, times, resampled_times)
        return interpolation_function(data, times, resampled_times)


@deprecated(
    "The task `CubicInterpolationTask` has been deprecated. Use `InterpolationTask` with"
    " `interpolation_object=scipy.interpolate.interp1d` and `kind='cubic'`",
    category=EODeprecationWarning,
)
class CubicInterpolationTask(InterpolationTask):
    """
    [DEPRECATED] Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.interp1d(kind='cubic')`
    """

    def __init__(self, feature: SingleFeatureSpec, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, kind="cubic", **kwargs)


@deprecated(
    "The task `SplineInterpolationTask` has been deprecated. Use `InterpolationTask` with"
    " `interpolation_object=scipy.interpolate.UnivariateSpline`",
    category=EODeprecationWarning,
)
class SplineInterpolationTask(InterpolationTask):
    """[DEPRECATED] Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.UnivariateSpline`"""

    def __init__(
        self, feature: SingleFeatureSpec, *, spline_degree: int = 3, smoothing_factor: float = 0, **kwargs: Any
    ):
        super().__init__(feature, scipy.interpolate.UnivariateSpline, k=spline_degree, s=smoothing_factor, **kwargs)


@deprecated(
    "The task `BSplineInterpolationTask` has been deprecated. Use `InterpolationTask` with"
    " `interpolation_object=scipy.interpolate.make_interp_spline`",
    category=EODeprecationWarning,
)
class BSplineInterpolationTask(InterpolationTask):
    """[DEPRECATED] Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.BSpline`"""

    def __init__(self, feature: SingleFeatureSpec, *, spline_degree: int = 3, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.make_interp_spline, k=spline_degree, **kwargs)


@deprecated(
    "The task `AkimaInterpolationTask` has been deprecated. Use `InterpolationTask` with"
    " `interpolation_object=scipy.interpolate.Akima1DInterpolator`",
    category=EODeprecationWarning,
)
class AkimaInterpolationTask(InterpolationTask):
    """[DEPRECATED] Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.Akima1DInterpolator`"""

    def __init__(self, feature: SingleFeatureSpec, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.Akima1DInterpolator, **kwargs)


class KrigingObject:
    """Interpolation function like object for Kriging"""

    def __init__(self, times: np.ndarray, series: np.ndarray, **kwargs: Any):
        self.regressor = GaussianProcessRegressor(**kwargs)

        # Since most of the data is close to zero (relatively to time points), first get time data in [0,1] range
        # to ensure nonzero results

        # Should normalize by max in resample time to be totally consistent,
        # but this works fine (0.03% error in testing)
        self.normalizing_factor = max(times) - min(times)

        self.regressor.fit(times.reshape(-1, 1) / self.normalizing_factor, series)
        self.call_args = kwargs.get("call_args", {})

    def __call__(self, new_times: np.ndarray, **kwargs: Any) -> np.ndarray:
        call_args = self.call_args.copy()
        call_args.update(kwargs)
        return self.regressor.predict(new_times.reshape(-1, 1) / self.normalizing_factor, **call_args)


class KrigingInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `sklearn.gaussian_process.GaussianProcessRegressor`

    Gaussian processes (superset of kriging) are especially used in geological missing data estimation.
    Compared to spline interpolation, gaussian processes produce much more smoothed results (might be desirable).
    """

    def __init__(self, feature: SingleFeatureSpec, **kwargs: Any):
        super().__init__(feature, KrigingObject, interpolate_pixel_wise=True, **kwargs)


class ResamplingTask(InterpolationTask):
    """A subclass of InterpolationTask task that works only with data with no missing, masked or invalid values.

    It always resamples timeseries to different timestamps.

    For example, to perform interpolation with the 'nearest' resampling use the values
    `interpolation_object=scipy.interpolate.interp1d` with `kind="nearest"`.
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        interpolation_object: Callable,
        resample_range: ResampleRangeType,
        *,
        result_interval: tuple[float, float] | None = None,
        unknown_value: float = np.nan,
        **interpolation_parameters: Any,
    ):
        if resample_range is None:
            raise ValueError("resample_range parameter must be in form ('start_date', 'end_date', step_days)")
        super().__init__(
            feature,
            interpolation_object,
            resample_range=resample_range,
            result_interval=result_interval,
            unknown_value=unknown_value,
            **interpolation_parameters,
        )

    def interpolate_data(self, data: np.ndarray, times: np.ndarray, resampled_times: np.ndarray) -> np.ndarray:
        """Interpolates data feature

        :param data: Array in a shape of t x num_obs, where num_obs = h x w x n
        :param times: Array of reference times in second relative to the first timestamp
        :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
        :return: Array of interpolated values
        """
        if np.isnan(data).any():
            raise ValueError("Data must not contain any masked/invalid pixels or NaN values")

        interp_func = self.get_interpolation_function(times, data)

        time_mask = (np.min(times) <= resampled_times) & (resampled_times <= np.max(times))
        new_data = np.full((resampled_times.size, *data.shape[1:]), np.nan, dtype=data.dtype)
        new_data[time_mask] = interp_func(resampled_times[time_mask])
        return new_data

    def get_interpolation_function(self, times: np.ndarray, series: np.ndarray) -> Callable:
        """Initializes interpolation model

        :param times: Array of reference times in second relative to the first timestamp
        :param series: One dimensional array of time series
        :return: Initialized interpolation model class
        """
        return self.interpolation_object(times, series, axis=0, **self.interpolation_parameters)


@deprecated(
    "The task `NearestResamplingTask` has been deprecated. Use `ResamplingTask` with"
    " `interpolation_object=scipy.interpolate.interp1d` and `kind='nearest'`.",
    category=EODeprecationWarning,
)
class NearestResamplingTask(ResamplingTask):
    """
    [DEPRECATED] Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='nearest')`
    """

    def __init__(self, feature: SingleFeatureSpec, resample_range: ResampleRangeType, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, resample_range, kind="nearest", **kwargs)


@deprecated(
    "The task `LinearResamplingTask` has been deprecated. Use `ResamplingTask` with"
    " `interpolation_object=scipy.interpolate.interp1d` and `kind='linear'`.",
    category=EODeprecationWarning,
)
class LinearResamplingTask(ResamplingTask):
    """[DEPRECATED] Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='linear')`"""

    def __init__(self, feature: SingleFeatureSpec, resample_range: ResampleRangeType, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, resample_range, kind="linear", **kwargs)


@deprecated(
    "The task `CubicResamplingTask` has been deprecated. Use `ResamplingTask` with"
    " `interpolation_object=scipy.interpolate.interp1d` and `kind='cubic'`.",
    category=EODeprecationWarning,
)
class CubicResamplingTask(ResamplingTask):
    """[DEPRECATED] Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='cubic')`"""

    def __init__(self, feature: SingleFeatureSpec, resample_range: ResampleRangeType, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, resample_range, kind="cubic", **kwargs)
