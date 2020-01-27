"""
Module handling processing of temporal features

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import itertools as it

import numpy as np

from eolearn.core import EOTask


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
        valid_data_mask = eopatch.mask['VALID_DATA'] if self.mask_data else eopatch.mask['IS_DATA']

        data = eopatch.data[self.data_feature] if self.data_index is None \
            else eopatch.data[self.data_feature][..., self.data_index][..., np.newaxis]

        madata = np.ma.array(data,
                             dtype=np.float32,
                             mask=np.logical_or(~valid_data_mask.astype(np.bool), np.isnan(data)))

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
