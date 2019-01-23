from eolearn.core import EOTask, FeatureType

import numpy as np
import logging
from PIL import Image
from skimage.morphology import disk, dilation

INTERP_METHODS = ['nearest', 'linear']

LOGGER = logging.getLogger(__name__)

class CalculateSnowMaskTask(EOTask):
    """
    The task calculates the snow mask using the given thresholds.
    """

    def __init__(self, data_feature, band_indices, dilation_size=0, NDSI_threshold=0.4, brightness_threshold=0.3):
        """
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands 2, 3, 7, 11

            Example: (FeatureType.DATA, 'ALL-BANDS')
        :type data_feature: tuple(FeatureType, str)
        :param band_indices: A list containing the indices at which the required bands can be found the feature
        :type band_indices: list(int)
        :param dilation_size: Size of the disk in pixels for performing dilation. Value 0 means do not perform
                          this post-processing step.
        :type dilation_size: int
        :param NDSI_threshold: Minimum value of the NDSI required to classify the pixel as snow
        :type NDSI_threshold: float
        :param brightness_threshold: Minimum value of the red band for a pixel to be classified as bright
        :type brightness_threshold: float
        """

        self.bands = list(self._parse_features(data_feature))[0]
        self.band_indices = band_indices
        self.NDSI_threshold = NDSI_threshold
        self.brightness_threshold = brightness_threshold
        self.dilation_size = dilation_size

    def execute(self, eopatch):
        bands = eopatch[self.bands[0]][self.bands[1]][:, :, :, self.band_indices]
        dates = bands.shape[0]
        NDSI = (bands[:, :, :, 0] - bands[:, :, :, 3]) / (bands[:, :, :, 0] + bands[:, :, :, 3])

        NDVI = (bands[:, :, :, 2] - bands[:, :, :, 1]) / (bands[:, :, :, 2] + bands[:, :, :, 1])

        calc_truth = np.where(np.logical_and(np.logical_or(NDSI >= self.NDSI_threshold, np.abs(NDVI - 0.1) < 0.05),
                                             bands[:, :, :, 0] >= self.brightness_threshold), 1, 0)
        if self.dilation_size:
            dilated_mask = np.zeros(shape=calc_truth)
            for date in range(dates):
                dilated_mask[date] = dilation(calc_truth[date], disk(self.dilation_size))
            eopatch.add_feature(FeatureType.MASK, 'SNOW',
                                dilated_mask.reshape(list(calc_truth.shape) + [1]).astype('uint8'))
        else:
            eopatch.add_feature(FeatureType.MASK, 'SNOW',
                                calc_truth.reshape(list(calc_truth.shape) + [1]).astype('uint8'))

        return eopatch