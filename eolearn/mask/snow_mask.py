"""
Module for snow masking

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import Feature

LOGGER = logging.getLogger(__name__)


class SnowMaskTask(EOTask):
    """The task calculates the snow mask using the given thresholds.

    The default values were optimised based on the Sentinel-2 L1C processing level. Values might not be optimal for L2A
    processing level
    """

    NDVI_THRESHOLD = 0.1

    def __init__(
        self,
        data_feature: Feature,
        band_indices: list[int],
        ndsi_threshold: float = 0.4,
        brightness_threshold: float = 0.3,
        dilation_size: int = 0,
        undefined_value: int = 0,
        mask_name: str = "SNOW_MASK",
    ):
        """
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands 2, 3, 7, 11, i.e. (FeatureType.DATA, 'BANDS')
        :param band_indices: A list containing the indices at which the required bands can be found in the data_feature.
            The required bands are B03, B04, B08 and B11 and the indices should be provided in this order. If the
            'BANDS' array contains all 13 L1C bands, then `band_indices=[2, 3, 7, 11]`. If the 'BANDS' are the 12 bands
            with L2A values, then `band_indices=[2, 3, 7, 10]`
        :param ndsi_threshold: Minimum value of the NDSI required to classify the pixel as snow
        :param brightness_threshold: Minimum value of the red band for a pixel to be classified as bright
        """
        self.bands_feature = self.parse_feature(data_feature, allowed_feature_types={FeatureType.DATA})
        self.band_indices = band_indices
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold
        self.disk_size = 2 * dilation_size + 1
        self.undefined_value = undefined_value
        self.mask_feature = (FeatureType.MASK, mask_name)

    def _apply_dilation(self, snow_masks: np.ndarray) -> np.ndarray:
        """Apply binary dilation for each mask in the series"""
        if self.disk_size > 0:
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.disk_size, self.disk_size))
            snow_masks = np.array([cv2.dilate(mask.astype(np.uint8), disk) for mask in snow_masks])
        return snow_masks.astype(bool)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bands = eopatch[self.bands_feature][..., self.band_indices]
        with np.errstate(divide="ignore", invalid="ignore"):
            # (B03 - B11) / (B03 + B11)
            ndsi = (bands[..., 0] - bands[..., 3]) / (bands[..., 0] + bands[..., 3])
            # (B08 - B04) / (B08 + B04)
            ndvi = (bands[..., 2] - bands[..., 1]) / (bands[..., 2] + bands[..., 1])

        ndsi_invalid, ndvi_invalid = ~np.isfinite(ndsi), ~np.isfinite(ndvi)
        ndsi[ndsi_invalid] = self.undefined_value
        ndvi[ndvi_invalid] = self.undefined_value

        ndi_criterion = (ndsi >= self.ndsi_threshold) | (np.abs(ndvi - self.NDVI_THRESHOLD) < self.NDVI_THRESHOLD / 2)
        brightnes_criterion = bands[..., 0] >= self.brightness_threshold
        snow_mask = np.where(ndi_criterion & brightnes_criterion, 1, 0)

        snow_mask = self._apply_dilation(snow_mask)

        snow_mask[ndsi_invalid | ndvi_invalid] = self.undefined_value

        eopatch[self.mask_feature] = snow_mask[..., np.newaxis].astype(bool)
        return eopatch
