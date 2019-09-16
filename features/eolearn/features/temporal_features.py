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


class AddStreamTemporalFeaturesTask(EOTask):
    # pylint: disable=too-many-instance-attributes
    """ Task that implements and adds to eopatch the spatio-temporal features proposed in [1].
    The features are added to the `data_timeless` attribute dictionary of eopatch.
    [1] Valero et. al. "Production of adynamic cropland mask by processing remote sensing
    image series at high temporal and spatial resolutions" Remote Sensing, 2016.
    """

    def __init__(self, data_feature=(FeatureType.DATA, 'NDVI'), data_index=None,
                 ndvi_feature_name=(FeatureType.DATA, 'NDVI'), mask_data=True, *,
                 max_val_feature='max_val', min_val_feature='min_val', mean_val_feature='mean_val',
                 sd_val_feature='sd_val', diff_max_feature='diff_max', diff_min_feature='diff_min',
                 diff_diff_feature='diff_diff', max_mean_feature='max_mean_feature',
                 max_mean_len_feature='max_mean_len', max_mean_surf_feature='max_mean_surf',
                 pos_surf_feature='pos_surf', pos_len_feature='pos_len', pos_rate_feature='pos_rate',
                 neg_surf_feature='neg_surf', neg_len_feature='neg_len', neg_rate_feature='neg_rate',
                 pos_transition_feature='pos_tran', neg_transition_feature='neg_tran',
                 feature_name_prefix=None, window_size=2, interval_tolerance=0.1, base_surface_min=-1.,
                 ndvi_barren_soil_cutoff=0.1):
        """
        :param data_feature: Name of data feature with values that are considered. Default is `'NDVI'`
        :type data_feature: object
        :param data_index: Index of to be extracted from last dimension in `data_feature`. If None, last dimension of
            data array is assumed ot be of size 1 (e.g. as in NDVI). Default is `None`
        :type data_index: int
        :param ndvi_feature_name: Name of data feature with NDVI values for bare soil transition considerations.
        If None, soil transitions are not calculated and set as 0
        :type ndvi_feature_name: obj
        :param mask_data: Flag specifying whether to mask data with `'VALID_DATA'` mask. If `False`, the `'IS_DATA'`
                          mask is used
        :param max_val_feature: Name of feature with computed max
        :type max_val_feature: str
        :param min_val_feature: Name of feature with computed min
        :type min_val_feature: str
        :param mean_val_feature: Name of feature with computed mean
        :type mean_val_feature: str
        :param sd_val_feature: Name of feature with computed standard deviation
        :param sd_val_feature: str
        :param diff_max_feature: Name of feature with computed max difference in a temporal sliding window
        :param diff_max_feature: str
        :param diff_min_feature: Name of feature with computed min difference in a temporal sliding window
        :param diff_min_feature: str
        :param diff_diff_feature: Name of feature with computed difference of difference in a temporal sliding window
        :param diff_diff_feature: str
        :param max_mean_feature: Name of feature with computed max of mean in a sliding window
        :param max_mean_feature: str
        :param max_mean_len_feature: Name of feature with computed length of time interval corresponding to max_mean
        :param max_mean_len_feature: str
        :param max_mean_surf_feature: Name of feature with computed surface under curve corresponding to max_mean
        :param max_mean_surf_feature: str
        :param pos_surf_feature: Name of feature with computed largest surface under curve where first derivative
        is positive
        :param pos_surf_feature: str
        :param pos_len_feature: Name of feature with computed length of time interval corresponding to pos_surf
        :param pos_len_feature: str
        :param pos_rate_feature: Name of feature with computed rate of change corresponding to pos_surf
        :param pos_rate_feature: str
        :param neg_surf_feature: Name of feature with computed largest surface under curve where first derivative
        is negative
        :param neg_surf_feature: str
        :param neg_len_feature: Name of feature with computed length of time interval corresponding to neg_surf
        :param neg_len_feature: str
        :param neg_rate_feature: Name of feature with computed rate of change corresponding to neg_surf
        :param neg_rate_feature: str
        :param pos_transition_feature: Name of feature to be associated to computed feature of argmax values
        :param pos_transition_feature: str
        :param neg_transition_feature: Name of feature to be associated to computed feature of argmax values
        :param neg_transition_feature: str
        :param feature_name_prefix: String to be used as prefix in names for calculated features.
        Default: value of data_feature
        :param feature_name_prefix: str
        :param window_size: Size of sliding temporal window
        :param window_size: int
        :param interval_tolerance: Tolerance for calculation of max_mean family of data features
        :param interval_tolerance: float
        :param base_surface_min: Minimal base value for data, used to more accurately calculate surface under curve.
        Default for indices like values is -1.0.
        :param base_surface_min: float
        :param ndvi_barren_soil_cutoff: Cutoff for bare soil detection
        :type ndvi_barren_soil_cutoff: 0.1
        """
        # pylint: disable=too-many-locals
        self.data_feature = next(iter(self._parse_features(data_feature, default_feature_type=FeatureType.DATA)))
        self.data_index = data_index or 0
        self.mask_data = mask_data
        self.ndvi_feature_name = next(iter(self._parse_features(ndvi_feature_name,
                                                                default_feature_type=FeatureType.DATA)))

        if feature_name_prefix:
            self.feature_name_prefix = feature_name_prefix
            if not feature_name_prefix.endswith("_"):
                self.feature_name_prefix += "_"
        else:
            self.feature_name_prefix = data_feature + "_"

        self.max_val_feature = self.feature_name_prefix + max_val_feature
        self.min_val_feature = self.feature_name_prefix + min_val_feature
        self.mean_val_feature = self.feature_name_prefix + mean_val_feature
        self.sd_val_feature = self.feature_name_prefix + sd_val_feature
        self.diff_max_feature = self.feature_name_prefix + diff_max_feature
        self.diff_min_feature = self.feature_name_prefix + diff_min_feature
        self.diff_diff_feature = self.feature_name_prefix + diff_diff_feature
        self.max_mean_feature = self.feature_name_prefix + max_mean_feature
        self.max_mean_len_feature = self.feature_name_prefix + max_mean_len_feature
        self.max_mean_surf_feature = self.feature_name_prefix + max_mean_surf_feature
        self.pos_surf_feature = self.feature_name_prefix + pos_surf_feature
        self.pos_len_feature = self.feature_name_prefix + pos_len_feature
        self.pos_rate_feature = self.feature_name_prefix + pos_rate_feature
        self.neg_surf_feature = self.feature_name_prefix + neg_surf_feature
        self.neg_len_feature = self.feature_name_prefix + neg_len_feature
        self.neg_rate_feature = self.feature_name_prefix + neg_rate_feature
        self.pos_transition_feature = self.feature_name_prefix + pos_transition_feature
        self.neg_transition_feature = self.feature_name_prefix + neg_transition_feature

        self.window_size = window_size
        self.interval_tolerance = interval_tolerance
        self.base_surface_min = base_surface_min

        self.ndvi_barren_soil_cutoff = ndvi_barren_soil_cutoff

    def execute(self, eopatch):
        """ Compute spatio-temporal features for input eopatch
        :param eopatch: Input eopatch
        :return: eopatch with computed spatio-temporal features
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        data = eopatch[self.data_feature[0]][self.data_feature[1]][..., self.data_index]
        valid_data_mask = np.ones_like(data)

        if data.ndim == 3:
            _, h, w = data.shape
        else:
            raise ValueError('{} feature has incorrect number of dimensions'.format(self.data_feature))

        madata = np.ma.array(data, dtype=np.float32, mask=~valid_data_mask.astype(np.bool))

        # Vectorized
        data_max_val = np.ma.MaskedArray.max(madata, axis=0).filled()
        data_min_val = np.ma.MaskedArray.min(madata, axis=0).filled()
        data_mean_val = np.ma.MaskedArray.mean(madata, axis=0).filled()
        data_sd_val = np.ma.MaskedArray.std(madata, axis=0).filled()

        data_diff_max = np.empty((h, w))
        data_diff_min = np.empty((h, w))
        # data_diff_diff = np.empty((h, w)) # Calculated later

        data_max_mean = np.empty((h, w))
        data_max_mean_len = np.empty((h, w))
        data_max_mean_surf = np.empty((h, w))

        data_pos_surf = np.empty((h, w))
        data_pos_len = np.empty((h, w))
        data_pos_rate = np.empty((h, w))

        data_neg_surf = np.empty((h, w))
        data_neg_len = np.empty((h, w))
        data_neg_rate = np.empty((h, w))

        data_pos_tr = np.empty((h, w))
        data_neg_tr = np.empty((h, w))
        for ih, iw in it.product(range(h), range(w)):
            data_curve = madata[:, ih, iw]
            valid_idx = np.where(~madata.mask[:, ih, iw])[0]

            data_curve = data_curve[valid_idx].filled()

            valid_dates = all_dates[valid_idx]

            sw_max = np.max(rolling_window(data_curve, self.window_size), -1)
            sw_min = np.min(rolling_window(data_curve, self.window_size), -1)

            sw_diff = sw_max - sw_min

            data_diff_max[ih, iw] = np.max(sw_diff)
            data_diff_min[ih, iw] = np.min(sw_diff)

            sw_mean = np.mean(rolling_window(data_curve, self.window_size), -1)
            max_mean = np.max(sw_mean)

            data_max_mean[ih, iw] = max_mean

            # Calculate max mean interval
            # Work with mean windowed or whole set?
            workset = data_curve  # or sw_mean, which is a bit more smoothed
            higher_mask = workset >= max_mean - ((1-self.interval_tolerance) * abs(max_mean))

            # Just normalize to have 0 on each side
            higher_mask_norm = np.zeros(len(higher_mask) + 2)
            higher_mask_norm[1:len(higher_mask)+1] = higher_mask

            # index of 1 that have 0 before them, SHIFTED BY ONE TO RIGHT
            up_mask = (higher_mask_norm[1:] == 1) & (higher_mask_norm[:-1] == 0)

            # Index of 1 that have 0 after them, correct indices
            down_mask = (higher_mask_norm[:-1] == 1) & (higher_mask_norm[1:] == 0)

            # Calculate length of interval as difference between times of first and last high enough observation,
            # in particular, if only one such observation is high enough, the length of such interval is 0
            # One can extend this to many more ways of calculating such length:
            # take forward/backward time differences, interpolate in between (again...) and treat this as
            # continuous problem, take mean of the time intervals between borders...
            times_up = valid_dates[up_mask[:-1]]
            times_down = valid_dates[down_mask[1:]]

            # There may be several such intervals, take the longest one
            times_diff = times_down - times_up
            # if there are no such intervals, the signal is constant,
            # set everything to zero and continue
            if times_diff.size == 0:
                data_max_mean_len[ih, iw] = 0
                data_max_mean_surf[ih, iw] = 0

                data_pos_surf[ih, iw] = 0
                data_pos_len[ih, iw] = 0
                data_pos_rate[ih, iw] = 0

                data_neg_surf[ih, iw] = 0
                data_neg_len[ih, iw] = 0
                data_neg_rate[ih, iw] = 0

                if self.ndvi_feature_name:
                    data_pos_tr[ih, iw] = 0
                    data_neg_tr[ih, iw] = 0
                continue

            max_ind = np.argmax(times_diff)
            data_max_mean_len[ih, iw] = times_diff[max_ind]

            fst = np.where(up_mask[:-1])[0]
            snd = np.where(down_mask[1:])[0]

            surface = np.trapz(data_curve[fst[max_ind]:snd[max_ind]+1] - self.base_surface_min,
                               valid_dates[fst[max_ind]:snd[max_ind]+1])
            data_max_mean_surf[ih, iw] = surface

            # Derivative based features
            # How to approximate derivative?
            derivatives = np.gradient(data_curve, valid_dates)

            # Positive derivative
            pos = np.zeros(len(derivatives) + 2)
            pos[1:len(derivatives)+1] = derivatives >= 0

            pos_der_int, pos_der_len, pos_der_rate, (start, _) = \
                self.derivative_features(pos, valid_dates, data_curve, self.base_surface_min)

            data_pos_surf[ih, iw] = pos_der_int
            data_pos_len[ih, iw] = pos_der_len
            data_pos_rate[ih, iw] = pos_der_rate

            neg = np.zeros(len(derivatives) + 2)
            neg[1:len(derivatives)+1] = derivatives <= 0

            neg_der_int, neg_der_len, neg_der_rate, (_, end) = \
                self.derivative_features(neg, valid_dates, data_curve, self.base_surface_min)

            data_neg_surf[ih, iw] = neg_der_int
            data_neg_len[ih, iw] = neg_der_len
            data_neg_rate[ih, iw] = neg_der_rate

            if self.ndvi_feature_name:
                data_pos_tr[ih, iw] = \
                    np.any(eopatch[self.ndvi_feature_name[0]][self.ndvi_feature_name[1]][:start+1, ih, iw, 0] <=
                           self.ndvi_barren_soil_cutoff)
                data_neg_tr[ih, iw] = \
                    np.any(eopatch[self.ndvi_feature_name[0]][self.ndvi_feature_name[1]][end:, ih, iw, 0] <=
                           self.ndvi_barren_soil_cutoff)

        eopatch.data_timeless[self.max_val_feature] = data_max_val[..., np.newaxis]
        eopatch.data_timeless[self.min_val_feature] = data_min_val[..., np.newaxis]
        eopatch.data_timeless[self.mean_val_feature] = data_mean_val[..., np.newaxis]
        eopatch.data_timeless[self.sd_val_feature] = data_sd_val[..., np.newaxis]

        eopatch.data_timeless[self.diff_max_feature] = data_diff_max[..., np.newaxis]
        eopatch.data_timeless[self.diff_min_feature] = data_diff_min[..., np.newaxis]
        eopatch.data_timeless[self.diff_diff_feature] = (data_diff_max - data_diff_min)[..., np.newaxis]

        eopatch.data_timeless[self.max_mean_feature] = data_max_mean[..., np.newaxis]
        eopatch.data_timeless[self.max_mean_len_feature] = data_max_mean_len[..., np.newaxis]
        eopatch.data_timeless[self.max_mean_surf_feature] = data_max_mean_surf[..., np.newaxis]

        eopatch.data_timeless[self.pos_len_feature] = data_pos_len[..., np.newaxis]
        eopatch.data_timeless[self.pos_surf_feature] = data_pos_surf[..., np.newaxis]
        eopatch.data_timeless[self.pos_rate_feature] = data_pos_rate[..., np.newaxis]
        eopatch.data_timeless[self.pos_transition_feature] = data_pos_tr[..., np.newaxis]

        eopatch.data_timeless[self.neg_len_feature] = data_neg_len[..., np.newaxis]
        eopatch.data_timeless[self.neg_surf_feature] = data_neg_surf[..., np.newaxis]
        eopatch.data_timeless[self.neg_rate_feature] = data_neg_rate[..., np.newaxis]
        eopatch.data_timeless[self.neg_transition_feature] = data_neg_tr[..., np.newaxis]

        return eopatch

    def get_data(self, patch):
        """Extracts and concatenates newly extracted features contained in the provided eopatch
        :param patch: Input eopatch
        :type patch: eolearn.core.EOPatch
        :return: Tuple of two lists: names of extracted features and their values
        """
        names = [self.max_val_feature, self.min_val_feature, self.mean_val_feature, self.sd_val_feature,
                 self.diff_max_feature, self.diff_min_feature, self.diff_diff_feature,
                 self.max_mean_feature, self.max_mean_len_feature, self.max_mean_surf_feature,
                 self.pos_len_feature, self.pos_surf_feature, self.pos_rate_feature, self.pos_transition_feature,
                 self.neg_len_feature, self.neg_surf_feature, self.neg_rate_feature, self.neg_transition_feature]

        dim_x, dim_y, _ = patch.data_timeless[names[0]].shape

        data = np.zeros((dim_x, dim_y, len(names)))
        for ind, name in enumerate(names):
            data[..., ind] = patch.data_timeless[name].squeeze()

        return names, data

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
