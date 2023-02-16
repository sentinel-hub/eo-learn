"""
Module for interpolating, smoothing and re-sampling features in EOPatch

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)
Copyright (c) 2018-2019 Filip Koprivec (Jožef Stefan Institute)
Copyright (c) 2018-2019 William Ouellette

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import datetime as dt
import inspect
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Set, Tuple, Union, cast

import dateutil
import numpy as np
import scipy.interpolate
from sklearn.gaussian_process import GaussianProcessRegressor

from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet
from eolearn.core.exceptions import EOUserWarning
from eolearn.core.types import FeaturesSpecification, SingleFeatureSpec

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
        result_interval: Optional[Tuple[float, float]] = None,
        mask_feature: Optional[SingleFeatureSpec] = None,
        copy_features: Optional[FeaturesSpecification] = None,
        unknown_value: float = np.nan,
        filling_factor: int = 10,
        scale_time: int = 3600,
        interpolate_pixel_wise: bool = False,
        **interpolation_parameters: Any,
    ):
        self.renamed_feature = self.parse_renamed_feature(feature, allowed_feature_types=FeatureTypeSet.RASTER_TYPES_4D)

        self.interpolation_object = interpolation_object
        self.resample_range = resample_range
        self.result_interval = result_interval

        self.mask_feature_parser = (
            None
            if mask_feature is None
            else self.get_feature_parser(
                mask_feature, allowed_feature_types={FeatureType.MASK, FeatureType.MASK_TIMELESS, FeatureType.LABEL}
            )
        )

        if resample_range is None and copy_features is not None:
            self.copy_features = None
            warnings.warn(
                'Argument "copy_features" will be ignored if "resample_range" is None. Nothing to copy.', EOUserWarning
            )
        else:
            self.copy_features_parser = None if copy_features is None else self.get_feature_parser(copy_features)

        self.unknown_value = unknown_value
        self.interpolation_parameters = interpolation_parameters
        self.scale_time = scale_time
        self.filling_factor = filling_factor
        self.interpolate_pixel_wise = interpolate_pixel_wise

        self._resampled_times = None

    @staticmethod
    def _mask_feature_data(feature_data: np.ndarray, mask: np.ndarray, mask_type: FeatureType) -> np.ndarray:
        """Masks values of data feature with a given mask of given mask type. The masking is done by assigning
        `numpy.nan` value.

        :param feature_data: Data array which will be masked
        :param mask: Mask array
        :return: Masked data array
        """

        if mask_type.is_spatial() and feature_data.shape[1:3] != mask.shape[-3:-1]:
            raise ValueError(
                f"Spatial dimensions of interpolation and mask feature do not match: {feature_data.shape} {mask.shape}"
            )

        if mask_type.is_temporal() and feature_data.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Time dimension of interpolation and mask feature do not match: {feature_data.shape} {mask.shape}"
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

        Function to return a binary array of same size as data where `True` values correspond to NaN values present at
        beginning or end of time-series. NaNs internal to the time-series are not included in the binary mask.

        :param data: Array of observations of size TxNOBS
        :return: Binary array of shape TxNOBS. `True` values indicate NaNs present at beginning or end of time-series
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
    def _get_unique_times(data: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Replace duplicate acquisitions which have same values on the chosen timescale with their average.
        The average is calculated with numpy.nanmean, meaning that NaN values are ignored when calculating the average.

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :param times: Array of reference times relative to the first timestamp
        :return: cleaned versions of data input
        """
        seen = set()
        duplication_list = []
        for idx, item in enumerate(times):
            if item in seen:
                duplication_list.append(idx)
            else:
                seen.add(item)
        duplicated_indices = np.array(duplication_list, dtype=int)

        duplicated_times = np.unique(times[duplicated_indices])

        for time in duplicated_times:
            indices = np.where(times == time)[0]
            nan_mask = np.all(np.isnan(data[indices]), axis=0)
            data[indices[0], ~nan_mask] = np.nanmean(data[indices][:, ~nan_mask], axis=0)

        times = np.delete(times, duplicated_indices, axis=0)
        data = np.delete(data, duplicated_indices, axis=0)

        return data, times

    def _copy_old_features(self, new_eopatch: EOPatch, old_eopatch: EOPatch) -> EOPatch:
        """Copy features from old EOPatch

        :param new_eopatch: New EOPatch container where the old features will be copied to
        :param old_eopatch: Old EOPatch container where the old features are located
        """
        if self.copy_features_parser is not None:
            existing_features: Set[Tuple[FeatureType, Optional[str]]] = set(self.parse_features(..., new_eopatch))

            renamed_features = self.copy_features_parser.get_renamed_features(old_eopatch)
            for copy_feature_type, copy_feature_name, copy_new_feature_name in renamed_features:
                new_feature = copy_feature_type, copy_new_feature_name

                if new_feature in existing_features:
                    raise ValueError(
                        f"Feature {copy_new_feature_name} of {copy_feature_type} already exists in the "
                        "new EOPatch! Use a different name!"
                    )
                existing_features.add(new_feature)

                new_eopatch[copy_feature_type][copy_new_feature_name] = old_eopatch[copy_feature_type][
                    copy_feature_name
                ]

        return new_eopatch

    def interpolate_data(self, data: np.ndarray, times: np.ndarray, resampled_times: np.ndarray) -> np.ndarray:
        """Interpolates data feature

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :param times: Array of reference times relative to the first timestamp
        :param resampled_times: Array of reference times relative to the first timestamp in initial timestamp array.
        :return: Array of interpolated values
        """
        # pylint: disable=too-many-locals
        # get size of 2d array t x nobs
        nobs = data.shape[-1]
        if self.interpolate_pixel_wise:
            # initialise array of interpolated values
            new_data = (
                data if self.resample_range is None else np.full((len(resampled_times), nobs), np.nan, dtype=data.dtype)
            )

            # Interpolate for each pixel, could be easily parallelized
            for obs in range(nobs):
                valid = ~np.isnan(data[:, obs])

                obs_interpolating_func = self.get_interpolation_function(times[valid], data[valid, obs])

                new_data[:, obs] = obs_interpolating_func(resampled_times[:, np.newaxis])

            # return interpolated values
            return new_data

        # mask representing overlap between reference and resampled times
        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))

        # define time values as linear monotonically increasing over the observations
        const = int(self.filling_factor * (np.max(times) - np.min(times)))
        temp_values = times[:, np.newaxis] + const * np.arange(nobs)[np.newaxis, :].astype(np.float64)
        res_temp_values = resampled_times[:, np.newaxis] + const * np.arange(nobs)[np.newaxis, :].astype(np.float64)

        # initialise array of interpolated values
        new_data = np.full((len(resampled_times), nobs), np.nan, dtype=data.dtype)

        # array defining index correspondence between reference times and resampled times
        ori2res = np.array(
            [
                np.abs(resampled_times - o).argmin()
                if np.min(resampled_times) <= o <= np.max(resampled_times)
                else None
                for o in times
            ]
        )

        # find NaNs that start or end a time-series
        row_nans, col_nans = np.where(self._get_start_end_nans(data))
        nan_row_res_indices = np.array([index for index in ori2res[row_nans] if index is not None], dtype=np.int32)
        nan_col_res_indices = np.array([index is not None for index in ori2res[row_nans]], dtype=bool)

        if nan_row_res_indices.size:
            # mask out from output values the starting/ending NaNs
            res_temp_values[nan_row_res_indices, col_nans[nan_col_res_indices]] = np.nan
        # if temporal values outside the reference dates are required (extrapolation) masked them to NaN
        res_temp_values[~time_mask, :] = np.nan

        # build 1d array for interpolation. Spline functions require monotonically increasing values of x,
        # so .T is used
        input_x = temp_values.T[~np.isnan(data).T]
        input_y = data.T[~np.isnan(data).T]

        # build interpolation function
        if len(input_x) > 1:
            interp_func = self.get_interpolation_function(input_x, input_y)

            # interpolate non-NaN values in resampled time values
            new_data[~np.isnan(res_temp_values)] = interp_func(res_temp_values[~np.isnan(res_temp_values)])

        # return interpolated values
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

    def get_resampled_timestamp(self, timestamp: List[dt.datetime]) -> List[dt.datetime]:
        """Takes a list of timestamps and generates new list of timestamps according to ``resample_range``

        :param timestamp: list of timestamps
        :return: new list of timestamps
        """
        days: List[dt.datetime]
        if self.resample_range is None:
            return timestamp

        if not isinstance(self.resample_range, (tuple, list)):
            raise ValueError(f"Invalid resample_range {self.resample_range}, expected tuple")

        if tuple(map(type, self.resample_range)) == (str, str, int):
            resample_range = cast(Tuple[str, str, int], self.resample_range)
            start_date = dateutil.parser.parse(resample_range[0])
            end_date = dateutil.parser.parse(resample_range[1])
            step = dt.timedelta(days=resample_range[2])
            days = [start_date]
            while days[-1] + step < end_date:
                days.append(days[-1] + step)

        elif np.all([isinstance(date, str) for date in self.resample_range]):
            days = [dateutil.parser.parse(date) for date in cast(List[str], self.resample_range)]
        elif np.all([isinstance(date, dt.datetime) for date in self.resample_range]):
            days = list(cast(List[dt.datetime], self.resample_range))
        else:
            raise ValueError("Invalid format in {self.resample_range}, expected strings or datetimes")

        return days

    @staticmethod
    def _get_eopatch_time_series(
        eopatch: EOPatch, ref_date: Optional[dt.datetime] = None, scale_time: int = 1
    ) -> np.ndarray:
        """Returns a numpy array with seconds passed between the reference date and the timestamp of each image.

        An array is constructed as time_series[i] = (timestamp[i] - ref_date).total_seconds().
        If reference date is None the first date in the EOPatch's timestamp is taken.
        If EOPatch timestamp attribute is empty the method returns None.

        :param eopatch: the EOPatch whose timestamps are used to construct the time series
        :param ref_date: reference date relative to which the time is measured
        :param scale_time: scale seconds by factor. If `60`, time will be in minutes, if `3600` hours
        """
        if not eopatch.timestamp:
            return np.zeros(0, dtype=np.int64)

        if ref_date is None:
            ref_date = eopatch.timestamp[0]

        return np.asarray(
            [round((timestamp - ref_date).total_seconds() / scale_time) for timestamp in eopatch.timestamp],
            dtype=np.int64,
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute method that processes EOPatch and returns EOPatch"""
        # pylint: disable=too-many-locals
        feature_type, feature_name, new_feature_name = self.renamed_feature

        # Make a copy not to change original numpy array
        feature_data = eopatch[feature_type][feature_name].copy()
        time_num, height, width, band_num = feature_data.shape
        if time_num <= 1:
            raise ValueError(
                f"Feature {(feature_type, feature_name)} has time dimension of size {time_num}, "
                "required at least size 2"
            )

        # Apply a mask on data
        if self.mask_feature_parser is not None:
            for mask_type, mask_name in self.mask_feature_parser.get_features(eopatch):
                negated_mask = ~eopatch[mask_type][mask_name].astype(bool)
                feature_data = self._mask_feature_data(feature_data, negated_mask, mask_type)

        # Flatten array
        feature_data = np.reshape(feature_data, (time_num, height * width * band_num))

        # If resampling create new EOPatch
        new_eopatch = EOPatch() if self.resample_range else eopatch

        # Resample times
        times = self._get_eopatch_time_series(eopatch, scale_time=self.scale_time)
        new_eopatch.timestamp = self.get_resampled_timestamp(eopatch.timestamp)
        total_diff = int((new_eopatch.timestamp[0].date() - eopatch.timestamp[0].date()).total_seconds())
        resampled_times = (
            self._get_eopatch_time_series(new_eopatch, scale_time=self.scale_time) + total_diff // self.scale_time
        )

        # Add BBox to eopatch if it was created anew
        if new_eopatch.bbox is None:
            new_eopatch.bbox = eopatch.bbox

        # Replace duplicate acquisitions which have same values on the chosen timescale with their average
        feature_data, times = self._get_unique_times(feature_data, times)

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
        new_eopatch[feature_type][new_feature_name] = np.reshape(
            feature_data, (feature_data.shape[0], height, width, band_num)
        )

        # append features from old patch
        new_eopatch = self._copy_old_features(new_eopatch, eopatch)

        return new_eopatch


class LinearInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `numpy.interp` and `@numba.jit(nopython=True)`

    :param parallel: interpolation is calculated in parallel using as many CPUs as detected
        by the multiprocessing module.
    :param kwargs: parameters of InterpolationTask(EOTask)
    """

    def __init__(self, feature: SingleFeatureSpec, parallel: bool = False, **kwargs: Any):
        self.parallel = parallel
        super().__init__(feature, np.interp, **kwargs)

    def interpolate_data(self, data: np.ndarray, times: np.ndarray, resampled_times: np.ndarray) -> np.ndarray:
        """Interpolates data feature

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :param times: Array of reference times in second relative to the first timestamp
        :param resampled_times: Array of reference times in second relative to the first timestamp in initial timestamp
            array.
        :return: Array of interpolated values
        """
        if self.parallel:
            return interpolation_function_parallel(data, times, resampled_times)
        return interpolation_function(data, times, resampled_times)


class CubicInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.interp1d(kind='cubic')`"""

    def __init__(self, feature: SingleFeatureSpec, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, kind="cubic", **kwargs)


class SplineInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.UnivariateSpline`"""

    def __init__(
        self, feature: SingleFeatureSpec, *, spline_degree: int = 3, smoothing_factor: float = 0, **kwargs: Any
    ):
        super().__init__(feature, scipy.interpolate.UnivariateSpline, k=spline_degree, s=smoothing_factor, **kwargs)


class BSplineInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.BSpline`"""

    def __init__(self, feature: SingleFeatureSpec, *, spline_degree: int = 3, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.make_interp_spline, k=spline_degree, **kwargs)


class AkimaInterpolationTask(InterpolationTask):
    """Implements `eolearn.features.InterpolationTask` by using `scipy.interpolate.Akima1DInterpolator`"""

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
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        interpolation_object: Callable,
        resample_range: ResampleRangeType,
        *,
        result_interval: Optional[Tuple[float, float]] = None,
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

        :param data: Array in a shape of t x nobs, where nobs = h x w x n
        :param times: Array of reference times in second relative to the first timestamp
        :param resampled_times: Array of reference times in second relative to the first timestamp in initial timestamp
            array.
        :return: Array of interpolated values
        """
        if True in np.unique(np.isnan(data)):
            raise ValueError("Data must not contain any masked/invalid pixels or NaN values")

        interp_func = self.get_interpolation_function(times, data)

        time_mask = (resampled_times >= np.min(times)) & (resampled_times <= np.max(times))
        new_data = np.full((resampled_times.size,) + data.shape[1:], np.nan, dtype=data.dtype)
        new_data[time_mask] = interp_func(resampled_times[time_mask])
        return new_data

    def get_interpolation_function(self, times: np.ndarray, series: np.ndarray) -> Callable:
        """Initializes interpolation model

        :param times: Array of reference times in second relative to the first timestamp
        :param series: One dimensional array of time series
        :return: Initialized interpolation model class
        """
        return self.interpolation_object(times, series, axis=0, **self.interpolation_parameters)


class NearestResamplingTask(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='nearest')`
    """

    def __init__(self, feature: SingleFeatureSpec, resample_range: ResampleRangeType, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, resample_range, kind="nearest", **kwargs)


class LinearResamplingTask(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='linear')`
    """

    def __init__(self, feature: SingleFeatureSpec, resample_range: ResampleRangeType, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, resample_range, kind="linear", **kwargs)


class CubicResamplingTask(ResamplingTask):
    """
    Implements `eolearn.features.ResamplingTask` by using `scipy.interpolate.interp1d(kind='cubic')`
    """

    def __init__(self, feature: SingleFeatureSpec, resample_range: ResampleRangeType, **kwargs: Any):
        super().__init__(feature, scipy.interpolate.interp1d, resample_range, kind="cubic", **kwargs)
