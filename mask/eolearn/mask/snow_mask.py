from eolearn.core import EOTask, FeatureType

import numpy as np
import logging
from PIL import Image
from skimage.morphology import disk, dilation

INTERP_METHODS = ['nearest', 'linear']

LOGGER = logging.getLogger(__name__)

class TheiaSnowMask(EOTask):
    """ Task to add a snow mask to an EOPatch

    This task computes a snow mask for the input EOPatch. The `data_feature` to be used as input to the
    classifier is a mandatory argument. If all of the needed features exist already, the classifier is run.
    Otherwise, if `data_feature` does not exist, a new OGC request at the given resolution is made and the
    classifier is run. This design should allow faster
    execution of the classifier, and reduce the number of requests. `linear` interpolation is used for resampling of
    the `data_feature` and cloud probability map, while `nearest` interpolation is used to upsample the binary cloud
    mask.

    This implementation should allow usage with any cloud detector implemented for different data sources (S2, L8, ..).
    """
    def __init__(self, data_feature, band_indices, cloud_mask_feature, DEM_feature, dilation_size=0, r_f=12, d_z=100, r_b=0.1, r_d=0.3, n1=0.4, n2=0.15, r1=0.2, r2=0.040, f_s=0.001, f_t=0.1):
        """
        Initialize the snow mask task.
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands B3, B4, B11 and B10

            Example: (FeatureType.DATA, 'ALL-BANDS')

        :type data_feature: tuple(FeatureType, str)
        :param band_indices: A list containing the indices at which the required bands can be found the feature
        :type band_indices: list(int)
        :param cloud_mask_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the cloud mask
        :type cloud_mask_feature: tuple(FeatureType, str)
        :param DEM_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the digital elevation model
        :type DEM_feature: tuple(FeatureType, str)
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
        :param n1: Minimum value of the NDSI for the pass 1 snow test
        :type n1: float
        :param n2: Minimum value of the NDSI for the pass 2 snow test
        :type n2: float
        :param r1: Minimum value of the red band reflectance the pass 1 snow test
        :type r1: float
        :param r2: Minimum value of the red band reflectance the pass 2 snow test
        :type r2: float
        :param f_s: Minimum snow fraction in the image to activate the pass 2 snow test
        :type f_s: float
        :param f_t: Minimum snow fraction in an elevation band to define z_s
        :type f_t: float
        """
        self.r_f = r_f
        self.d_z = d_z,
        self.r_b = r_b
        self.r_d = r_d
        self.n1 = n1
        self.n2 = n2
        self.r1 = r1
        self.r2 = r2
        self.f_s = f_s
        self.f_t = f_t
        self.bands = list(self._parse_features(data_feature, default_feature_type=FeatureType.DATA))[0]
        self.band_indices = band_indices
        self.DEM = list(self._parse_features(DEM_feature, default_feature_type=FeatureType.DATA_TIMELESS))[0]
        self.CLM = list(self._parse_features(cloud_mask_feature, default_feature_type=FeatureType.MASK))[0]
        self.dilation_size = dilation_size


    def resample_red(self, values):
        size = np.array(values.shape) // 12
        res = Image.fromarray(values).resize(size, Image.BICUBIC).resize(values.shape, Image.NEAREST)
        return np.array(res)


    def execute(self, eopatch):
        bands = eopatch[self.bands[0]][self.bands[1]][:, :, :, self.band_indices[:-1]]
        dates = bands.shape[0]
        B10 = eopatch[self.bands[0]][self.bands[1]][:, :, :, self.band_indices[-1]]
        DEM = eopatch[self.DEM[0]][self.DEM[1]][:, :, 0]
        clm = eopatch[self.CLM[0]][self.CLM[1]]

        NDSI = (bands[:, :, :, 0] - bands[:, :, :, 2]) / (bands[:, :, :, 0] + bands[:, :, :, 2])

        cirrus_mask = np.zeros(shape=B10.shape)
        clm_temp = np.zeros(shape=clm.shape)
        for date in range(dates):
            cirrus_mask[date] = np.where(B10[date] > 0.015 + 0.00001 * DEM, 1, 0)
            clm_temp[date] = np.logical_or(np.where(
                np.logical_and(clm[date, :, :, 0] == 1,
                               self.resample_red(bands[date, :, :, 1]) > self.r_d),
                1, 0), cirrus_mask[date]).astype(int).reshape(clm_temp[date].shape)

        pass1 = np.where(np.logical_and(np.logical_not(clm_temp[:, :, :, 0]),
                                        np.logical_and(NDSI > self.n1, bands[:, :, :, 1] > self.r1)), 1, 0)

        DEM_edges = np.arange(np.min(DEM), np.max(DEM), self.d_z)

        clm_pass1 = np.where(
            np.logical_or(
                clm_temp[:, :, :, 0], (bands[:, :, :, 1] > self.r_b) & np.logical_not(pass1) & clm[:, :, :, 0]), 1, 0)

        if np.max(DEM) not in DEM_edges:
            DEM_edges = np.append(DEM_edges, np.array([np.max(DEM_edges)]))
        nbins = len(DEM_edges) - 1
        cloud_free_pixels = np.zeros((dates, nbins))
        if nbins > 0:
            for date in range(dates):
                cloud_free_pixels[date, :] = \
                np.histogram(DEM.flatten()[np.where(np.logical_not(clm_pass1[date, :, :].flatten()))], bins=DEM_edges)[0]
        snow_frac = np.zeros((dates, nbins))
        cloud_frac = np.zeros((dates, nbins))
        for date in range(dates):
            for bn in range(nbins):
                if cloud_free_pixels[date, bn] > 0:
                    in_range = np.where((DEM >= DEM_edges[bn]) & (DEM < int(DEM_edges[bn + 1])))
                    in_range_clm = np.where(
                        np.logical_and((DEM >= DEM_edges[bn]) & (DEM < int(DEM_edges[bn + 1])),
                                       np.logical_not(clm_pass1[date])))
                    snow_frac[date, bn] = np.sum(pass1[date][in_range_clm]) / cloud_free_pixels[date, bn]
                    cloud_frac[date, bn] = np.sum(clm_pass1[date][in_range]) / cloud_free_pixels[date, bn]

        total_snow_frac = np.array([np.sum(pass1[date]) / pass1[date].size for date in range(dates)])
        snowmask = np.zeros(list(NDSI.shape) + [1])
        pass2 = np.zeros(pass1.shape)
        for date in range(dates):
            if total_snow_frac[date] > self.f_s:
                try:
                    izs = np.argmax(snow_frac[date] > self.f_t)
                    zs = DEM_edges[max(izs - 2, 0)]
                    pass2[date, :, :] = np.where(np.logical_and(DEM[:, :, 0] > zs,
                                                             np.logical_and(np.logical_not(clm_temp[date, :, :, 0]),
                                                                            np.logical_and(NDSI[date] > self.n2,
                                                                                           bands[date, :, :,
                                                                                           1] > self.r2))), 1, 0)
                except:
                    pass2[date] = np.zeros(pass1.shape[1:])
            if self.dilation_size:
                snowmask[date, :, :, :] = dilation(np.logical_or(pass1[date], pass2[date]), disk(self.dilation_size)).reshape(list(pass1.shape[1:]) + [1])
            else:
                snowmask[date, :, :, :] = np.logical_or(pass1[date], pass2[date]).reshape(list(pass1.shape[1:]) + [1])

        eopatch.add_feature(FeatureType.MASK, 'SNOW_THEIA', snowmask)

        return eopatch
