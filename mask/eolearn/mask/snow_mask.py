"""
Module for snow masking

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import itertools
import logging
from abc import ABCMeta
from typing import Any

import numpy as np
from skimage.morphology import binary_dilation, disk

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import FeatureSpec

from .utils import resize_images

LOGGER = logging.getLogger(__name__)


class BaseSnowMaskTask(EOTask, metaclass=ABCMeta):
    """Base class for snow detection and masking"""

    def __init__(
        self,
        data_feature: FeatureSpec,
        band_indices: list[int],
        dilation_size: int = 0,
        undefined_value: int = 0,
        mask_name: str = "SNOW_MASK",
    ):
        """
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
        :param band_indices: A list containing the indices at which the required bands can be found in the data_feature.
        :param dilation_size: Size of the disk in pixels for performing dilation. Value 0 means do not perform
            this post-processing step.
        """
        self.bands_feature = self.parse_feature(data_feature, allowed_feature_types={FeatureType.DATA})
        self.band_indices = band_indices
        self.dilation_size = dilation_size
        self.undefined_value = undefined_value
        self.mask_feature = (FeatureType.MASK, mask_name)

    def _apply_dilation(self, snow_masks: np.ndarray) -> np.ndarray:
        """Apply binary dilation for each mask in the series"""
        if self.dilation_size:
            snow_masks = np.array([binary_dilation(mask, disk(self.dilation_size)) for mask in snow_masks])
        return snow_masks


class SnowMaskTask(BaseSnowMaskTask):
    """The task calculates the snow mask using the given thresholds.

    The default values were optimised based on the Sentinel-2 L1C processing level. Values might not be optimal for L2A
    processing level
    """

    NDVI_THRESHOLD = 0.1

    def __init__(
        self,
        data_feature: FeatureSpec,
        band_indices: list[int],
        ndsi_threshold: float = 0.4,
        brightness_threshold: float = 0.3,
        **kwargs: Any,
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
        super().__init__(data_feature, band_indices, **kwargs)
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold

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


class TheiaSnowMaskTask(BaseSnowMaskTask):
    """Task to add a snow mask to an EOPatch. The input data is either Sentinel-2 L1C or L2A level

    Original implementation and documentation available at https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow

    ATBD https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow/blob/master/doc/atbd/ATBD_CES-Neige.pdf

    This task computes a snow mask for the input EOPatch. The `data_feature` to be used as input to the
    classifier is a mandatory argument. If all required features exist already, the classifier is run.
    `linear` interpolation is used for resampling of the `data_feature` and cloud probability map, while `nearest`
    interpolation is used to upsample the binary cloud mask.
    """

    B10_THR = 0.015
    DEM_FACTOR = 0.00001

    def __init__(
        self,
        data_feature: FeatureSpec,
        band_indices: list[int],
        cloud_mask_feature: FeatureSpec,
        dem_feature: FeatureSpec,
        dem_params: tuple[float, float] = (100, 0.1),
        red_params: tuple[float, float, float, float, float] = (12, 0.3, 0.1, 0.2, 0.040),
        ndsi_params: tuple[float, float, float] = (0.4, 0.15, 0.001),
        b10_index: int | None = None,
        **kwargs: Any,
    ):
        """
        :param data_feature: EOPatch feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the bands B3, B4, and B11

            Example: `(FeatureType.DATA, 'ALL-BANDS')`
        :param band_indices: A list containing the indices at which the required bands can be found in the bands
            feature. If all L1C band values are provided, `band_indices=[2, 3, 11]`. If all L2A band values are
            provided, then `band_indices=[2, 3, 10]`
        :param cloud_mask_feature: EOPatch CLM feature represented by a tuple in the form of
            `(FeatureType, 'feature_name')` containing the cloud mask
        :param dem_feature: EOPatch DEM feature represented by a tuple in the form of `(FeatureType, 'feature_name')`
            containing the digital elevation model
        :param b10_index: Array index where the B10 band is stored in the bands feature. This is used to refine the
            initial cloud mask
        :param dem_params: Tuple with parameters pertaining DEM processing. The first value specifies the bin size
            used to group DEM values, while the second value specifies the minimum snow fraction in an elevation band
            to define z_s. With reference to the ATBD, the tuple is (d_z, f_t)
        :param red_params: Tuple specifying parameters to process the B04 red band. The first parameter defines the
            scaling factor for down-sampling the red band, the second parameter is the maximum value of the
            down-sampled red band for a dark cloud pixel, the third parameter is the minimum value
            to return a non-snow pixel to the cloud mask, the fourth is the minimum reflectance value to pass the 1st
            snow test, and the fifth is the minimum reflectance value to pass the 2nd snow test. With reference to the
            ATBD, the tuple is (r_f, r_d, r_b, r_1, r_2)
        :param ndsi_params: Tuple specifying parameters for the NDSI. First parameter is the minimum value to pass the
            1st snow test, the second parameter is the minimum value to pass the 2nd snow test, and the third parameter
            is the minimum snow fraction in the image to activate the pass 2 snow test. With reference to the
            ATBD, the tuple is (n_1, n_2, f_s)
        """
        super().__init__(data_feature, band_indices, **kwargs)
        self.dem_feature = self.parse_feature(dem_feature)
        self.clm_feature = self.parse_feature(cloud_mask_feature)
        self.dem_params = dem_params
        self.red_params = red_params
        self.ndsi_params = ndsi_params
        self.b10_index = b10_index

    def _resample_red(self, input_array: np.ndarray) -> np.ndarray:
        """Method to resample the values of the red band

        The input array is first down-scaled using bicubic interpolation and up-scaled back using nearest neighbour
        interpolation
        """
        _, height, width = input_array.shape
        size = (height // self.red_params[0], width // self.red_params[0])
        downscaled = resize_images(input_array[..., np.newaxis], new_size=size)
        return resize_images(downscaled, new_size=(height, width)).squeeze()

    def _adjust_cloud_mask(
        self, bands: np.ndarray, cloud_mask: np.ndarray, dem: np.ndarray, b10: np.ndarray | None
    ) -> np.ndarray:
        """Adjust existing cloud mask using cirrus band if L1C data and resampled red band

        Add to the existing cloud mask pixels found thresholding down-sampled red band and cirrus band/DEM
        """
        if b10 is not None:
            clm_b10 = b10 > self.B10_THR + self.DEM_FACTOR * dem
        else:
            clm_b10 = np.full_like(cloud_mask, True)

        criterion = (cloud_mask == 1) & (self._resample_red(bands[..., 1]) > self.red_params[1])
        return criterion | clm_b10

    def _apply_first_pass(
        self, bands: np.ndarray, ndsi: np.ndarray, clm: np.ndarray, dem: np.ndarray, clm_temp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Apply first pass of snow detection"""
        snow_mask_pass1 = ~clm_temp & (ndsi > self.ndsi_params[0]) & (bands[..., 1] > self.red_params[3])

        clm_pass1 = clm_temp | ((bands[..., 1] > self.red_params[2]) & ~snow_mask_pass1 & clm.astype(bool))

        min_dem, max_dem = np.min(dem), np.max(dem)
        dem_edges = np.linspace(min_dem, max_dem, int(np.ceil((max_dem - min_dem) / self.dem_params[0])))
        nbins = len(dem_edges) - 1

        dem_hist_clear_pixels, snow_frac = None, None
        if nbins > 0:
            snow_frac = np.zeros(shape=(bands.shape[0], nbins))
            dem_hist_clear_pixels = np.array([np.histogram(dem[~mask], bins=dem_edges)[0] for mask in clm_pass1])

            for date, nbin in itertools.product(range(bands.shape[0]), range(nbins)):
                if dem_hist_clear_pixels[date, nbin] > 0:
                    dem_mask = (dem_edges[nbin] <= dem) & (dem < dem_edges[nbin + 1])
                    in_dem_range_clear = np.where(dem_mask & ~clm_pass1[date])
                    snow_frac[date, nbin] = (
                        np.sum(snow_mask_pass1[date][in_dem_range_clear]) / dem_hist_clear_pixels[date, nbin]
                    )
        return snow_mask_pass1, snow_frac, dem_edges

    def _apply_second_pass(
        self,
        bands: np.ndarray,
        ndsi: np.ndarray,
        dem: np.ndarray,
        clm_temp: np.ndarray,
        snow_mask_pass1: np.ndarray,
        snow_frac: np.ndarray | None,
        dem_edges: np.ndarray,
    ) -> np.ndarray:
        """Second pass of snow detection"""
        _, height, width, _ = bands.shape
        total_snow_frac = np.sum(snow_mask_pass1, axis=(1, 2)) / (height * width)
        snow_mask_pass2 = np.full_like(snow_mask_pass1, False)
        for date in range(bands.shape[0]):
            if (
                (total_snow_frac[date] > self.ndsi_params[2])
                and snow_frac is not None
                and np.any(snow_frac[date] > self.dem_params[1])
            ):
                z_s = dem_edges[np.max(np.argmax(snow_frac[date] > self.dem_params[1]) - 2, 0)]

                snow_mask_pass2[date, :, :] = (
                    (dem > z_s)
                    & ~clm_temp[date]
                    & (ndsi[date] > self.ndsi_params[1])
                    & (bands[date, ..., 1] > self.red_params[-1])
                )

        return snow_mask_pass2

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Run multi-pass snow detection"""
        bands = eopatch[self.bands_feature][..., self.band_indices]
        b10 = eopatch[self.bands_feature][..., self.b10_index] if self.b10_index is not None else None
        dem = eopatch[self.dem_feature][..., 0]
        clm = eopatch[self.clm_feature][..., 0]

        with np.errstate(divide="ignore", invalid="ignore"):
            # (B03 - B11) / (B03 + B11)
            ndsi = (bands[..., 0] - bands[..., 2]) / (bands[..., 0] + bands[..., 2])

        ndsi_invalid = ~np.isfinite(ndsi)
        ndsi[ndsi_invalid] = self.undefined_value

        clm_temp = self._adjust_cloud_mask(bands, clm, dem, b10)

        snow_mask_pass1, snow_frac, dem_edges = self._apply_first_pass(bands, ndsi, clm, dem, clm_temp)

        snow_mask_pass2 = self._apply_second_pass(bands, ndsi, dem, clm_temp, snow_mask_pass1, snow_frac, dem_edges)

        snow_mask = self._apply_dilation(snow_mask_pass1 | snow_mask_pass2)

        eopatch[self.mask_feature] = snow_mask[..., np.newaxis].astype(bool)

        return eopatch
