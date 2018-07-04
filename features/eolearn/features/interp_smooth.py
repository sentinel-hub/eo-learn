""" Module for interpolating, smoothing and resampling features in EOPatch """

from eolearn.core import EOTask, FeatureType

import numpy as np
from scipy import interpolate


class BSplineInterpolation(EOTask):
    """
    Task for interpolation of time-series of user specified feature using B-splines. Note that
    the interpolated feature is overwritten during the process. After the execution of the task
    the feature will contain interpolated values and not the original ones.

    The task takes set of valid data points (timestamp[i], feature_value[i]) and determines a
    a smooth spline approximation of degree k on the interval timestamp[0] <= timestamp <= timestamp[-1].
    The task requires that there exists eopatch.mask['VALID_DATA'] feature. All pixels with
    eopatch.mask['VALID_DATA'] == 0 are masked in interpolation.  In case the interpolation needs
    to be avaulated outside the interpolated region the value at the boundary value is returned by
    default.

    The task uses scipy.interpolate.splrep method to do the spline interpolation. The splrep method's
    arguments can be passed via splrep_args dictionary.
    """

    def __init__(self, feature_name, splrep_args):
        self.feature_name = feature_name
        self.splrep_args = splrep_args

    def _invalid_to_nan(self, eopatch, valid):
        """
        Sets all invalid elements (where valid element == 0) in feature array to np.nan.
        """
        valid = valid.squeeze(axis=-1)

        feature = eopatch.data[self.feature_name]

        if len(feature.shape) > 3:
            for band in np.arange(0, feature.shape[-1]):
                feature[..., band] = np.where(valid == 0, np.nan, feature[..., band])
        else:
            feature = np.where(valid == 0, np.nan, feature)

        eopatch.data[self.feature_name] = feature

    def _flatten_feature(self, eopatch):
        """
        Flattens the array in spatial dimensions. The method reshapes
        an array of shape (t, h, w, b) to (t, h x w, b), where
        t is number of time frames, h and w are the height and width
        of a frame, respectively, and b is optional.
        """
        ftr_shape = eopatch.data[self.feature_name].shape

        if len(ftr_shape) > 3:
            eopatch.data[self.feature_name] = np.reshape(eopatch.data[self.feature_name],
                                                         (ftr_shape[0], ftr_shape[1] * ftr_shape[2], ftr_shape[3]))
        else:
            eopatch.data[self.feature_name] = np.reshape(eopatch.data[self.feature_name],
                                                         (ftr_shape[0], ftr_shape[1] * ftr_shape[2]))

    @staticmethod
    def do_spline_fit(arr, timestamps, **kwargs):
        nans = np.isnan(arr)

        spline = interpolate.splrep(timestamps[~nans], arr[~nans], **kwargs)

        return interpolate.splev(timestamps, spline, der=0, ext=3)

    def execute(self, eopatch):
        if self.feature_name not in eopatch.features[FeatureType.DATA].keys():
            raise ValueError('Feature {} not found in eopatch.data.'.format(self.feature_name))

        ftr_shape = eopatch.data[self.feature_name].shape

        if 'VALID_DATA' not in eopatch.features[FeatureType.MASK].keys():
            valid = np.ones(ftr_shape if len(ftr_shape) < 4 else ftr_shape[:-1])
        else:
            valid = eopatch.mask['VALID_DATA']

        # set invalid pixels to np.nan
        self._invalid_to_nan(eopatch, valid)
        # flatten the array in spatial dimension
        self._flatten_feature(eopatch)

        # do interpolation
        if len(ftr_shape) > 3:
            for band in np.arange(0, ftr_shape[-1]):
                eopatch.data[self.feature_name][..., band] = np.apply_along_axis(BSplineInterpolation.do_spline_fit,
                                                                                 axis=0,
                                                                                 arr=eopatch.data[self.feature_name][
                                                                                     ..., band],
                                                                                 timestamps=eopatch.time_series(),
                                                                                 **self.splrep_args)
        else:
            eopatch.data[self.feature_name] = np.apply_along_axis(BSplineInterpolation.do_spline_fit, axis=0,
                                                                  arr=eopatch.data[self.feature_name],
                                                                  timestamps=eopatch.time_series(),
                                                                  **self.splrep_args)

        # reshape back
        eopatch.data[self.feature_name] = np.reshape(eopatch.data[self.feature_name], ftr_shape)

        return eopatch
