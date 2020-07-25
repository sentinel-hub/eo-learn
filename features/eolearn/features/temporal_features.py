"""
Module handling processing of temporal features

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import itertools as it

import numpy as np

from eolearn.core import EOTask, FeatureType
from eolearn.core import MapFeatureTask
from eolearn.ml_tools.utilities import rolling_window


class AddSpatioTemporalFeaturesTask(EOTask):
    """ Task that implements and adds to eopatch the spatio-temporal features proposed in [1].

    This task assumes that the argmax/argmin of NDVI, NDVI slope and B4 are present in eopatch. The computed
    spatio-temporal features correspond to the concatenation of reflectance (green, red, near-infrared and short-wave
    infrared in [1]) values taken at dates where:

    1) NDVI is maximum
    2) NDVI is minimum
    3) red reflectances are maximum
    4) NDVI slope is maximum
    5) NDVI slope is minimum

    The features are added to the `data_timeless` attribute dictionary of eopatch.

    [1] Waldner et al. "Automated annual cropland mapping using knowledge-based temporal features", ISPRS Journal of
    Photogrammetry and Remote Sensing, 2015

    """
    def __init__(self, argmax_ndvi='ARGMAX_NDVI', argmin_ndvi='ARGMIN_NDVI', argmax_red='ARGMAX_B4',
                 argmax_ndvi_slope='ARGMAX_NDVI_SLOPE', argmin_ndvi_slope='ARGMIN_NDVI_SLOPE', feats_feature='STF',
                 data_feature='BANDS-S2-L1C', indices=None):
        """ Class constructor

        Initialisation of task variables. The name of the dictionary keys that will be used for the computation of the
        features needs to be specified. These features are assumed to be existing in the eopatch. The indices of the
        reflectances to be used as features is an input parameter. If `None` is used, the data attribute is supposed to
        have 13 bands and indices for green/red/infrared/short-wave-infrared are used.

        :param argmax_ndvi: Name of `argmax_ndvi` feature in eopatch. Default is `'ARGMAX_NDVI'`
        :type argmax_ndvi: str
        :param argmin_ndvi: Name of `argmin_ndvi` feature in eopatch. Default is `'ARGMIN_NDVI'`
        :type argmin_ndvi: str
        :param argmax_red: Name of `argmax_red` feature in eopatch. Default is `'ARGMAX_B4'`
        :type argmax_red: str
        :param argmax_ndvi_slope: Name of `argmax_ndvi_slope` feature in eopatch. Default is `'ARGMAX_NDVI_SLOPE'`
        :type argmax_ndvi_slope: str
        :param argmin_ndvi_slope: Name of `argmin_ndvi_slope` feature in eopatch. Default is `'ARGMIN_NDVI_SLOPE'`
        :type argmin_ndvi_slope: str
        :param feats_feature: Name of feature containing spatio-temporal features. Default is `'STF'`
        :type feats_feature: str
        :param data_feature: Name of feature containing the reflectances to be used as features. Default is
            `'BANDS-S2-L1C'`
        :type data_feature: str
        :param indices: List of indices from `data_feature` to be used as features. Default is `None`, corresponding to
                        [2, 3, 7, 11] indices
        :type indices: None or list of int
        """
        self.argmax_ndvi = argmax_ndvi
        self.argmin_ndvi = argmin_ndvi
        self.argmax_red = argmax_red
        self.argmax_ndvi_slope = argmax_ndvi_slope
        self.argmin_ndvi_slope = argmin_ndvi_slope
        self.feats_feature = feats_feature
        self.data_feature = data_feature
        if indices is None:
            indices = [2, 3, 7, 11]
        self.indices = indices

    def execute(self, eopatch):
        """ Compute spatio-temporal features for input eopatch

        :param eopatch: Input eopatch
        :return: eopatch with computed spatio-temporal features
        """
        # pylint: disable=invalid-name
        amax_ndvi, amin_ndvi = eopatch.data_timeless[self.argmax_ndvi], eopatch.data_timeless[self.argmin_ndvi]
        amax_ndvi_slope, amin_ndvi_slope = eopatch.data_timeless[self.argmax_ndvi_slope], \
                                           eopatch.data_timeless[self.argmin_ndvi_slope]
        amax_red = eopatch.data_timeless[self.argmax_red]

        stf_idx = [amax_ndvi, amin_ndvi, amax_ndvi_slope, amin_ndvi_slope, amax_red]

        bands = eopatch.data[self.data_feature][..., self.indices]

        _, h, w, _ = bands.shape
        hh, ww = np.ogrid[:h, :w]
        stf = np.concatenate([bands[ii.squeeze(), hh, ww] for ii in stf_idx if ii is not None], axis=-1)

        eopatch.data_timeless[self.feats_feature] = stf

        return eopatch


class AddMaxMinTemporalIndicesTask(EOTask):
    """ Task to compute temporal indices of the maximum and minimum of a data feature

        This class computes the `argmax` and `argmin` of a data feature in the input eopatch (e.g. NDVI, B4). The data
        can be masked out by setting the `mask_data` flag to `True`. In that case, the `'VALID_DATA'` mask feature is
        used for masking. If `mask_data` is `False`, the data is masked using the `'IS_DATA'` feature.

        Two new features are added to the `data_timeless` attribute.
    """
    def __init__(self, data_feature='NDVI', data_index=None, amax_data_feature='ARGMAX_NDVI',
                 amin_data_feature='ARGMIN_NDVI', mask_data=True):
        """ Task constructor

        :param data_feature: Name of the feature in data used for computation of max/min. Default is `'NDVI'`
        :type data_feature: str
        :param data_index: Index of to be extracted from last dimension in `data_feature`. If None, last dimension of
            data array is assumed ot be of size 1 (e.g. as in NDVI). Default is `None`
        :type data_index: int
        :param amax_data_feature: Name of feature to be associated to computed feature of argmax values
        :type amax_data_feature: str
        :param amin_data_feature: Name of feature to be associated to computed feature of argmin values
        :type amin_data_feature: str
        :param mask_data: Flag specifying whether to mask data with `'VALID_DATA'` mask. If `False`, the `'IS_DATA'`
                          mask is used
        """
        self.data_feature = data_feature
        self.data_index = data_index
        self.mask_data = mask_data
        self.amax_feature = amax_data_feature
        self.amin_feature = amin_data_feature

    def execute(self, eopatch):
        """ Compute argmax/argmin of specified `data_feature` and `data_index`

        :param eopatch: Input eopatch
        :return: eopatch with added argmax/argmin features
        """
        if self.mask_data:
            valid_data_mask = eopatch.mask['VALID_DATA']
        else:
            valid_data_mask = eopatch.mask['IS_DATA']

        if self.data_index is None:
            data = eopatch.data[self.data_feature]
        else:
            data = eopatch.data[self.data_feature][..., self.data_index]

        madata = np.ma.array(data,
                             dtype=np.float32,
                             mask=~valid_data_mask.astype(np.bool))

        argmax_data = np.ma.MaskedArray.argmax(madata, axis=0)
        argmin_data = np.ma.MaskedArray.argmin(madata, axis=0)

        if argmax_data.ndim == 2:
            argmax_data = argmax_data.reshape(argmax_data.shape + (1,))

        if argmin_data.ndim == 2:
            argmin_data = argmin_data.reshape(argmin_data.shape + (1,))

        eopatch.data_timeless[self.amax_feature] = argmax_data
        eopatch.data_timeless[self.amin_feature] = argmin_data

        return eopatch


class AddMaxMinNDVISlopeIndicesTask(EOTask):
    """ Task to compute the argmax and armgin of the NDVI slope

    This task computes the slope of the NDVI feature using central differences. The NDVI feature can be masked using the
    `'VALID_DATA'` mask. Current implementation loops through every location of eopatch, and is therefore slow.

    The NDVI slope at date t is comuted as $(NDVI_{t+1}-NDVI_{t-1})/(date_{t+1}-date_{t-1})$.
    """
    def __init__(self, data_feature='NDVI', argmax_feature='ARGMAX_NDVI_SLOPE', argmin_feature='ARGMIN_NDVI_SLOPE',
                 mask_data=True):
        """ Task constructor

        :param data_feature: Name of data feature with NDVI values. Default is `'NDVI'`
        :type data_feature: str
        :param argmax_feature: Name of feature with computed argmax values of the NDVI slope
        :type argmax_feature: str
        :param argmin_feature: Name of feature with computed argmin values of the NDVI slope
        :type argmin_feature: str
        :param mask_data: Flag for masking NDVI data. Default is `True`
        """
        self.data_feature = data_feature
        self.argmax_feature = argmax_feature
        self.argmin_feature = argmin_feature
        self.mask_data = mask_data

    def execute(self, eopatch):
        """ Computation of NDVI slope using finite central differences

        This implementation loops through every spatial location, considers the valid NDVI values and approximates their
        first order derivative using central differences. The argument of min and max is added to the eopatch.

        The NDVI slope at date t is comuted as $(NDVI_{t+1}-NDVI_{t-1})/(date_{t+1}-date_{t-1})$.

        :param eopatch: Input eopatch
        :return: eopatch with NDVI slope argmin/argmax features
        """
        # pylint: disable=invalid-name
        if self.mask_data:
            valid_data_mask = eopatch.mask['VALID_DATA']
        else:
            valid_data_mask = eopatch.mask['IS_DATA']

        ndvi = np.ma.array(eopatch.data[self.data_feature],
                           dtype=np.float32,
                           mask=~valid_data_mask.astype(np.bool))

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        if ndvi.ndim == 4:
            h, w = ndvi.shape[1: 3]
        else:
            raise ValueError('{} feature has incorrect number of dimensions'.format(self.data_feature))

        argmax_ndvi_slope, argmin_ndvi_slope = np.zeros((h, w, 1), dtype=np.uint8), np.zeros((h, w, 1), dtype=np.uint8)

        for ih, iw in it.product(range(h), range(w)):

            ndvi_curve = ndvi[:, ih, iw, :]
            valid_idx = np.where(~ndvi.mask[:, ih, iw])[0]

            ndvi_curve = ndvi_curve[valid_idx]
            valid_dates = all_dates[valid_idx]

            ndvi_slope = np.convolve(ndvi_curve.squeeze(), [1, 0, -1], 'valid') / np.convolve(valid_dates, [1, 0, -1],
                                                                                              'valid')

            # +1 to compensate for the 'valid' convolution which eliminates first and last
            argmax_ndvi_slope[ih, iw] = valid_idx[np.argmax(ndvi_slope) + 1]
            argmin_ndvi_slope[ih, iw] = valid_idx[np.argmin(ndvi_slope) + 1]

            del ndvi_curve, valid_idx, valid_dates, ndvi_slope

        eopatch.data_timeless[self.argmax_feature] = argmax_ndvi_slope
        eopatch.data_timeless[self.argmin_feature] = argmin_ndvi_slope

        return eopatch


class SurfaceExtractionTask(EOTask):
    """ Task that implements and adds to eopatch the derivative surface part of spatio-temporal features
    proposed in [1]. The features are added to the `data_timeless` and `mask_timeless attribute dictionary of eopatch.

    This task extracts maximal consecutive surface under the masked data curve, date length corresponding to maximal
    surface interval, rate of change in maximal interval and ndvi transition before start and end of this interval.

    [1] Valero et. al. "Production of dynamic cropland mask by processing remote sensing
    image series at high temporal and spatial resolutions" Remote Sensing, 2016.
    """

    def __init__(self, input_feature, output_feature_name, ndvi_feature, selected_data_points_feature,
                 base_surface=-1, ndvi_barren_soil_cutoff=0.1, valid_mask_feature=None, fill_value=-1):
        """

        :param input_feature: Input feature
        :param output_feature_name: Output feature name prefix
        :param ndvi_feature: NDVI feature for barren soil detection
        :param selected_data_points_feature: Mask feature indicating which data points both through time and
        location to consider for calculation
        :param base_surface: Minimal base value for data, used to more accurately calculate surface under curve.
        :param ndvi_barren_soil_cutoff: Cutoff for barren soil detection
        :param valid_mask_feature: Timeless mask feature indicating for which pixels to run feature extraction
        :param fill_value: Fill value for pixels set as not valid with valid_mask
        """
        self.input_feature = next(iter(self._parse_features(input_feature)))
        self.output_feature_name = output_feature_name
        self.ndvi_feature = next(iter(self._parse_features(ndvi_feature)))
        self.selected_data_points_feature = next(iter(
            self._parse_features(selected_data_points_feature, allowed_feature_types=[FeatureType.MASK, ])))
        self.valid_mask_feature = None if valid_mask_feature is None else \
            next(iter(self._parse_features(valid_mask_feature, allowed_feature_types=[FeatureType.MASK_TIMELESS, ])))

        self.base_surface = base_surface
        self.ndvi_barren_soil_cutoff = ndvi_barren_soil_cutoff

        self.fill_value = fill_value

    def execute(self, eopatch, **kwargs):
        # pylint: disable=invalid-name, too-many-locals
        data = eopatch[self.input_feature].squeeze(-1)
        t, h, w = data.shape

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        data_surf = np.full((h, w, 1), self.fill_value, dtype=float)
        data_len = np.full((h, w, 1), self.fill_value, dtype=float)
        data_rate = np.full((h, w, 1), self.fill_value, dtype=float)
        data_transition_before = np.full((h, w, 1), self.fill_value, dtype=bool)
        data_transition_after = np.full((h, w, 1), self.fill_value, dtype=bool)

        mask = eopatch[self.selected_data_points_feature].squeeze(-1)
        ndvi = eopatch[self.ndvi_feature].squeeze(-1)

        if self.valid_mask_feature:
            pixel_mask = eopatch[self.valid_mask_feature].squeeze()
        else:
            pixel_mask = np.ones((h, w))

        padded_mask = np.zeros(t + 2, dtype=bool)

        for ih, iw in np.asarray(np.where(pixel_mask)).T:
            if not np.any(mask[:, ih, iw]):  # If no pixels are selected, take the fill value
                continue
            padded_mask[1:-1] = mask[:, ih, iw]

            data_surf[ih, iw], data_len[ih, iw], data_rate[ih, iw], (start, end) = \
                self.derivative_features(padded_mask, all_dates, data[:, ih, iw], self.base_surface)

            data_transition_before[ih, iw] = np.any(ndvi[:start, ih, iw] >= self.ndvi_barren_soil_cutoff)
            data_transition_after[ih, iw] = np.any(ndvi[end + 1:, ih, iw] >= self.ndvi_barren_soil_cutoff)

        eopatch[FeatureType.DATA_TIMELESS][self.output_feature_name] = \
            np.concatenate([data_surf, data_len, data_rate], -1)
        eopatch[FeatureType.MASK_TIMELESS][self.output_feature_name + '_transition'] = \
            np.concatenate([data_transition_before, data_transition_after], -1)

        return eopatch

    @staticmethod
    def derivative_features(mask, valid_dates, data, base_surface_min):
        """Calculates derivative based features for provided data points selected by
        mask (increasing data points, decreasing data points)
        :param mask: Mask indicating data points considered
        :type mask: np.array
        :param valid_dates: Dates (x-axis for surface calculation)
        :type valid_dates: np.array
        :param data: Base data
        :type data: np.array
        :param base_surface_min: Base surface value (added to each measurement)
        :type base_surface_min: float
        :return: Tuple of: maximal consecutive surface under the data curve,
                           date length corresponding to maximal surface interval,
                           rate of change in maximal interval,
                           (starting date index of maximal interval, ending date index of interval)
        """
        # index of 1 that have 0 before them, shifted by one to right
        up_mask = (mask[1:] == 1) & (mask[:-1] == 0)

        # Index of 1 that have 0 after them, correct indices
        down_mask = (mask[:-1] == 1) & (mask[1:] == 0)

        fst_der = np.where(up_mask[:-1])[0]
        snd_der = np.where(down_mask[1:])[0]
        der_ind_max = -1
        der_int_max = -1

        for ind, (start, end) in enumerate(zip(fst_der, snd_der)):

            integral = np.trapz(
                data[start:end + 1] - base_surface_min,
                valid_dates[start:end + 1])

            if abs(integral) >= abs(der_int_max):
                der_int_max = integral
                der_ind_max = ind

        start_ind = fst_der[der_ind_max]
        end_ind = snd_der[der_ind_max]

        der_len = valid_dates[end_ind] - valid_dates[start_ind]
        der_rate = (data[end_ind] - data[start_ind]) / der_len if der_len else 0

        return der_int_max, der_len, der_rate, (start_ind, end_ind)


class MaxMeanLenTask(EOTask):
    """ Task that implements and adds to eopatch the surface based parts of spatio-temporal features
    proposed in [1]. The features are added to the `data_timeless` attribute dictionary of eopatch.

    This task extracts the length and surface under the longest interval with values greater or equal to
    (1 - `interval_tolerance`) of `max_mean_feature.

    [1] Valero et. al. "Production of adynamic cropland mask by processing remote sensing
    image series at high temporal and spatial resolutions" Remote Sensing, 2016.
    """
    def __init__(self, input_feature, max_mean_feature, output_feature_name, interval_tolerance, base_surface_min=-1,
                 valid_mask_feature=None, fill_value=-1):
        """
        :param input_feature: Input feature
        :param max_mean_feature: Feature considered as maximal mean for interval calculation
        :param output_feature_name: Output feature name prefix
        :param interval_tolerance: Tolerance for interval data height calculation
        :param base_surface_min: Minimal base value for data, used to more accurately calculate surface under curve.
        :param valid_mask_feature: Timeless mask feature indicating for which pixels to run feature extraction
        :param fill_value: Fill value for pixels set as not valid with valid_mask
        """
        self.input_feature = next(iter(self._parse_features(input_feature)))
        self.max_mean_feature = next(iter(self._parse_features(max_mean_feature)))

        self.output_feature_name = output_feature_name

        self.interval_tolerance = interval_tolerance
        self.base_surface_min = base_surface_min

        self.valid_mask_feature = None if valid_mask_feature is None else \
            next(iter(self._parse_features(valid_mask_feature, allowed_feature_types=[FeatureType.MASK_TIMELESS, ])))

        self.fill_value = fill_value

    def execute(self, eopatch, **kwargs):
        # pylint: disable=invalid-name, too-many-locals
        data = eopatch[self.input_feature]
        _, h, w, _ = data.shape

        max_mean = eopatch[self.max_mean_feature]

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        higher_mask = data >= max_mean - self.interval_tolerance * np.abs(max_mean)
        padded_higher = np.concatenate([[np.zeros_like(higher_mask[0])], higher_mask, [np.zeros_like(higher_mask[0])]])

        increasing_mask = ((padded_higher[1:] == 1) & (padded_higher[:-1] == 0)).squeeze(-1)
        decreasing_mask = ((padded_higher[:-1] == 1) & (padded_higher[1:] == 0)).squeeze(-1)

        data_max_mean_len = np.full((h, w, 1), self.fill_value, dtype=float)
        data_max_mean_surf = np.full((h, w, 1), self.fill_value, dtype=float)

        if self.valid_mask_feature:
            pixel_mask = eopatch[self.valid_mask_feature].squeeze()
        else:
            pixel_mask = np.ones((h, w))

        for ih, iw in np.asarray(np.where(pixel_mask)).T:
            times_up = all_dates[increasing_mask[:-1, ih, iw]]
            times_down = all_dates[decreasing_mask[1:, ih, iw]]

            times_diff = times_down - times_up
            max_ind = np.argmax(times_diff)
            data_max_mean_len[ih, iw] = times_diff[max_ind]

            fst = np.where(increasing_mask[:-1, ih, iw])[0]
            snd = np.where(decreasing_mask[1:, ih, iw])[0]

            surface = np.trapz(data[:, ih, iw].squeeze(-1)[fst[max_ind]:snd[max_ind] + 1] - self.base_surface_min,
                               all_dates[fst[max_ind]:snd[max_ind] + 1])
            data_max_mean_surf[ih, iw] = surface

        eopatch[FeatureType.DATA_TIMELESS][self.output_feature_name + '_max_mean_len'] = data_max_mean_len
        eopatch[FeatureType.DATA_TIMELESS][self.output_feature_name + '_max_mean_surf'] = data_max_mean_surf

        return eopatch


class TemporalRollingWindowTask(MapFeatureTask):
    """ Task that applies a numpy universal function along a time sliced rolling window

    Applies a function provided over a time rolling window over 1 dimensional `input_feature` along with
    additional `**kwargs`.

    For a feature with shape `(t, h, w, 1)` the function is applied to view of numpy array of shape
    `(t - window_size, h, t, 1, window_size)`

    Using a function `np.max` produces a `(h, w, t - window_size)` output with the last dimension being the max
    of sliding window of size `window_size` for each pixel in the input data.

    """
    def __init__(self, input_feature, output_feature, function, window_size, **kwargs):
        super().__init__(input_feature, output_feature, **kwargs)
        self.inner_function = function
        self.window_size = window_size

    def map_method(self, feature):

        rtr = np.moveaxis(self.inner_function(rolling_window(feature, (self.window_size, 0, 0, 0)), axis=-1,
                                              **self.kwargs).squeeze(-1), 0, 2)
        return rtr
