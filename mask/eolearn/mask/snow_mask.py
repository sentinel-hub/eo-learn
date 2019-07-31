"""
Module for snow masking
"""

import logging

import numpy as np
from PIL import Image
from skimage.morphology import disk, dilation

from eolearn.core import EOTask, FeatureType


LOGGER = logging.getLogger(__name__)


class SnowMask(EOTask):
    """
    The task calculates the snow mask using the given thresholds.

    THe default values were optimised based on the Sentinel-2 L1C processing level. Values might not be optimal for L2A
    processing level
    """

    NDVI_THRESHOLD = 0.1

    def __init__(self, data_feature, band_indices, dilation_size=0, ndsi_threshold=0.4, brightness_threshold=0.3,
                 undefined_value=np.NaN, mask_name='SNOW_MASK'):
        """
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands 2, 3, 7, 11, i.e. (FeatureType.DATA, 'BANDS')
        :type data_feature: tuple(FeatureType, str)
        :param band_indices: A list containing the indices at which the required bands can be found in the data_feature.
            The required bands are B03, B04, B08 and B11 and the indices should be provided in this order. If the
            'BANDS' array contains all 13 L1C bands, then `band_indices=[2, 3, 7, 11]`. If the 'BANDS' are the 12 bands
            with L2A values, then `band_indices=[2, 3, 7, 10]`
        :type band_indices: list(int)
        :param dilation_size: Size of the disk in pixels for performing dilation. Value 0 means do not perform
                              this post-processing step.
        :type dilation_size: int
        :param ndsi_threshold: Minimum value of the NDSI required to classify the pixel as snow
        :type ndsi_threshold: float
        :param brightness_threshold: Minimum value of the red band for a pixel to be classified as bright
        :type brightness_threshold: float
        :param undefined_value: Value assigned to invalid values derived form computation of normalised indices
        :type undefined_value: float
        """

        self.bands_feature = self._parse_features(data_feature).next()
        self.band_indices = band_indices
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold
        self.dilation_size = dilation_size
        self.undefined_value = undefined_value
        self.mask_name = mask_name

    def execute(self, eopatch):
        bands = eopatch[self.bands_feature][..., self.band_indices]
        with np.errstate(divide='ignore'):
            # (B03 - B11) / (B03 + B11)
            ndsi = (bands[..., 0] - bands[..., 3]) / (bands[..., 0] + bands[..., 3])
            # (B08 - B04) / (B08 + B04)
            ndvi = (bands[..., 2] - bands[..., 1]) / (bands[..., 2] + bands[..., 1])

        ndsi_invalid, ndvi_invalid = ~np.isfinite(ndsi), ~np.isfinite(ndvi)
        ndsi[ndsi_invalid] = self.undefined_value
        ndvi[ndvi_invalid] = self.undefined_value

        snow_mask = np.where(np.logical_and(np.logical_or(ndsi >= self.ndsi_threshold,
                                                          np.abs(ndvi - self.NDVI_THRESHOLD) < self.NDVI_THRESHOLD / 2),
                                            bands[..., 0] >= self.brightness_threshold), 1, 0)
        if self.dilation_size:
            snow_mask = np.array([dilation(mask, disk(self.dilation_size)) for mask in snow_mask])

        snow_mask[np.logical_or(ndsi_invalid, ndvi_invalid)] = self.undefined_value
        eopatch.add_feature(FeatureType.MASK, self.mask_name, snow_mask[..., np.newaxis].astype('uint8'))
        return eopatch


class TheiaSnowMask(EOTask):
    """ Task to add a snow mask to an EOPatch

    This task computes a snow mask for the input EOPatch. The `data_feature` to be used as input to the
    classifier is a mandatory argument. If all of the needed features exist already, the classifier is run.
    Otherwise, if `data_feature` does not exist, a new OGC request at the given resolution is made and the
    classifier is run. This design should allow faster
    execution of the classifier, and reduce the number of requests. `linear` interpolation is used for resampling of
    the `data_feature` and cloud probability map, while `nearest` interpolation is used to upsample the binary cloud
    mask.
    """
    def __init__(self, data_feature, band_indices, cloud_mask_feature, dem_feature, dilation_size=0, r_f=12, d_z=100,
                 r_b=0.1, r_d=0.3, n_1=0.4, n_2=0.15, r_1=0.2, r_2=0.040, f_s=0.001, f_t=0.1):
        """
        Initialize the snow mask task.
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands B3, B4, B11 and B10

            Example: (FeatureType.DATA, 'ALL-BANDS')

        :type data_feature: tuple(FeatureType, str)
        :param band_indices: A list containing the indices at which the required bands can be found the feature
        :type band_indices: list(int)
        :param cloud_mask_feature: EOPatch CLM feature represented by a tuple in the form of
                                   `(FeatureType, 'feature_name')` containing the cloud mask
        :type cloud_mask_feature: tuple(FeatureType, str)
        :param dem_feature: EOPatch DEM feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the digital elevation model
        :type dem_feature: tuple(FeatureType, str)
        :param dilation_size: Size of the disk in pixels for performing dilation. Value 0 means do not perform
                              this post-processing step.
        :type dilation_size: int
        :param r_f: Resize factor to produce the down-sampled red band
        :type r_f: int
        :param d_z: Size of elevation band in the digital elevation model used to define z_s
        :type d_z: float
        :param r_b: Minimum value of the red band reflectance to return a non-snow pixel to the cloud mask
        :type r_b: float
        :param r_d: Maximum value of the down-sampled red band reflectance to define a dark cloud pixel
        :type r_d: float
        :param n_1: Minimum value of the NDSI for the pass 1 snow test
        :type n_1: float
        :param n_2: Minimum value of the NDSI for the pass 2 snow test
        :type n_2: float
        :param r_1: Minimum value of the red band reflectance the pass 1 snow test
        :type r_1: float
        :param r_2: Minimum value of the red band reflectance the pass 2 snow test
        :type r_2: float
        :param f_s: Minimum snow fraction in the image to activate the pass 2 snow test
        :type f_s: float
        :param f_t: Minimum snow fraction in an elevation band to define z_s
        :type f_t: float
        """
        self.r_f = r_f
        self.d_z = d_z
        self.r_b = r_b
        self.r_d = r_d
        self.n_1 = n_1
        self.n_2 = n_2
        self.r_1 = r_1
        self.r_2 = r_2
        self.f_s = f_s
        self.f_t = f_t
        self.dilation_size = dilation_size
        self.band_indices = band_indices
        self.bands_feature = self._parse_features(data_feature).next()
        self.dem_feature = self._parse_features(dem_feature).next()
        self.clm_feature = self._parse_features(cloud_mask_feature).next()

    def resample_red(self, values):
        """
        Method to resample the values of the red band
        :param values: input values
        :return: resampled values
        """
        size = np.array(values.shape) // self.r_f
        res = Image.fromarray(values).resize(size, Image.BICUBIC).resize(values.shape, Image.NEAREST)
        return np.array(res)

    def execute(self, eopatch):
        bands = eopatch[self.bands_feature[0]][self.bands_feature[1]][:, :, :, self.band_indices[:-1]]
        dates = bands.shape[0]
        b_10 = eopatch[self.bands_feature[0]][self.bands_feature[1]][:, :, :, self.band_indices[-1]]
        dem = eopatch[self.dem_feature[0]][self.dem_feature[1]][:, :, 0]
        clm = eopatch[self.clm_feature[0]][self.clm_feature[1]]

        ndsi = (bands[:, :, :, 0] - bands[:, :, :, 2]) / (bands[:, :, :, 0] + bands[:, :, :, 2])

        cirrus_mask = np.zeros(shape=b_10.shape)
        clm_temp = np.zeros(shape=clm.shape)
        for date in range(dates):
            cirrus_mask[date] = np.where(b_10[date] > 0.015 + 0.00001 * dem, 1, 0)
            clm_temp[date] = np.logical_or(np.where(
                np.logical_and(clm[date, :, :, 0] == 1,
                               self.resample_red(bands[date, :, :, 1]) > self.r_d),
                1, 0), cirrus_mask[date]).astype(int).reshape(clm_temp[date].shape)

        pass1 = np.where(np.logical_and(np.logical_not(clm_temp[:, :, :, 0]),
                                        np.logical_and(ndsi > self.n_1, bands[:, :, :, 1] > self.r_1)), 1, 0)

        dem_edges = np.arange(np.min(dem), np.max(dem), self.d_z)

        clm_pass1 = np.where(
            np.logical_or(
                clm_temp[:, :, :, 0], (bands[:, :, :, 1] > self.r_b) & np.logical_not(pass1) & clm[:, :, :, 0]), 1, 0)

        if np.max(dem) not in dem_edges:
            dem_edges = np.append(dem_edges, np.array([np.max(dem_edges)]))
        nbins = len(dem_edges) - 1
        cloud_free_pixels = np.zeros((dates, nbins))
        if nbins > 0:
            for date in range(dates):
                cloud_free_pixels[date, :] = np.histogram(dem.flatten()[np.where(
                    np.logical_not(clm_pass1[date, :, :].flatten()))], bins=dem_edges)[0]
        snow_frac = np.zeros((dates, nbins))
        cloud_frac = np.zeros((dates, nbins))
        for date in range(dates):
            for i in range(nbins):
                if cloud_free_pixels[date, i] > 0:
                    in_range = np.where((dem >= dem_edges[i]) & (dem < int(dem_edges[i + 1])))
                    in_range_clm = np.where(
                        np.logical_and((dem >= dem_edges[i]) & (dem < int(dem_edges[i + 1])),
                                       np.logical_not(clm_pass1[date])))
                    snow_frac[date, i] = np.sum(pass1[date][in_range_clm]) / cloud_free_pixels[date, i]
                    cloud_frac[date, i] = np.sum(clm_pass1[date][in_range]) / cloud_free_pixels[date, i]

        total_snow_frac = np.array([np.sum(pass1[date]) / pass1[date].size for date in range(dates)])
        snowmask = np.zeros(list(ndsi.shape) + [1])
        pass2 = np.zeros(pass1.shape)
        for date in range(dates):
            if total_snow_frac[date] > self.f_s:
                if np.any(snow_frac[date] > self.f_t):
                    izs = np.argmax(snow_frac[date] > self.f_t)
                    z_s = dem_edges[max(izs - 2, 0)]
                    pass2[date, :, :] = np.where(
                        np.logical_and(dem[:, :, 0] > z_s, np.logical_and(
                            np.logical_not(clm_temp[date, :, :, 0]),
                            np.logical_and(ndsi[date] > self.n_2, bands[date, :, :, 1] > self.r_2))), 1, 0)
                else:
                    pass2[date] = np.zeros(pass1.shape[1:])
            if self.dilation_size:
                snowmask[date, :, :, :] = dilation(np.logical_or(pass1[date], pass2[date]),
                                                   disk(self.dilation_size)).reshape(list(pass1.shape[1:]) + [1])
            else:
                snowmask[date, :, :, :] = np.logical_or(pass1[date], pass2[date]).reshape(list(pass1.shape[1:]) + [1])

        eopatch.add_feature(FeatureType.MASK, 'SNOW_THEIA', snowmask)

        return eopatch
