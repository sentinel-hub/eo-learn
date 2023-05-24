"""
Module for cloud masking

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import Protocol, cast

import cv2
import numpy as np
from lightgbm import Booster
from skimage.morphology import disk

from sentinelhub import BBox, bbox_to_resolution

from eolearn.core import EOPatch, EOTask, FeatureType, execute_with_mp_lock
from eolearn.core.utils.common import _apply_to_spatial_axes

from .utils import resize_images

LOGGER = logging.getLogger(__name__)


class ClassifierType(Protocol):
    """Defines the necessary classifier interface."""

    # pylint: disable-next=missing-function-docstring,invalid-name
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa[N803]
        ...

    # pylint: disable-next=missing-function-docstring,invalid-name
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa[N803]
        ...


class CloudMaskTask(EOTask):
    """Cloud masking with an improved s2cloudless model and the SSIM-based multi-temporal classifier.

    Its intended output is a cloud mask that is based on the outputs of both
    individual classifiers (a dilated intersection of individual binary masks).
    Additional cloud masks and probabilities can be added for either classifier or both.

    Prior to feature extraction and classification, it is recommended that the input be
    downscaled by specifying the source and processing resolutions. This should be done
    for the following reasons:

    - faster execution
    - lower memory consumption
    - noise mitigation

    Resizing is performed with linear interpolation. After classification, the cloud
    probabilities are themselves upscaled to the original dimensions, before proceeding
    with masking operations.

    Example usage:

    .. code-block:: python

        # Only output the combined mask
        task1 = CloudMaskTask(processing_resolution='120m',
                              mask_feature='CLM_INTERSSIM',
                              average_over=16,
                              dilation_size=8)

        # Only output monotemporal masks. Only monotemporal processing is done.
        task2 = CloudMaskTask(processing_resolution='120m',
                              mono_features=(None, 'CLM_S2C'),
                              mask_feature=None,
                              average_over=16,
                              dilation_size=8)
    """

    MODELS_FOLDER = os.path.join(os.path.dirname(__file__), "models")
    MONO_CLASSIFIER_NAME = "pixel_s2_cloud_detector_lightGBM_v0.2.txt"
    MULTI_CLASSIFIER_NAME = "ssim_s2_cloud_detector_lightGBM_v0.2.txt"

    def __init__(
        self,
        data_feature: tuple[FeatureType, str] = (FeatureType.DATA, "BANDS-S2-L1C"),
        is_data_feature: tuple[FeatureType, str] = (FeatureType.MASK, "IS_DATA"),
        all_bands: bool = True,
        processing_resolution: None | float | tuple[float, float] = None,
        max_proc_frames: int = 11,
        mono_features: tuple[str | None, str | None] | None = None,
        multi_features: tuple[str | None, str | None] | None = None,
        mask_feature: tuple[FeatureType, str] | None = (FeatureType.MASK, "CLM_INTERSSIM"),
        mono_threshold: float = 0.4,
        multi_threshold: float = 0.5,
        average_over: int | None = 4,
        dilation_size: int | None = 2,
        mono_classifier: ClassifierType | None = None,
        multi_classifier: ClassifierType | None = None,
    ):
        """
        :param data_feature: A data feature which stores raw Sentinel-2 reflectance bands.
            Default value: `'BANDS-S2-L1C'`.
        :param is_data_feature: A mask feature which indicates whether data is valid.
            Default value: `'IS_DATA'`.
        :param all_bands: Flag, which indicates whether images will consist of all 13 Sentinel-2 bands or only
            the required 10.
        :param processing_resolution: Resolution to be used during the computation of cloud probabilities and masks,
            expressed in meters. Resolution is given as a pair of x and y resolutions. If a single value is given,
            it is used for both dimensions. Default `None` represents source resolution.
        :param max_proc_frames: Maximum number of frames (including the target, for multi-temporal classification)
            considered in a single batch iteration (To keep memory usage at agreeable levels, the task operates on
            smaller batches of time frames).
        :param mono_features: Tuple of keys to be used for storing cloud probabilities and masks (in that order!) of
            the mono classifier. The probabilities are added as a data feature, while masks are added as a mask
            feature. By default, none of them are added.
        :param multi_features: Tuple of keys used for storing cloud probabilities and masks of the multi classifier.
            The probabilities are added as a data feature, while masks are added as a mask feature. By default,
            none of them are added.
        :param mask_feature: Name of the output intersection feature. Default value: `'CLM_INTERSSIM'`. If `None` the
            intersection feature is not computed.
        :param mono_threshold: Cloud probability threshold for the mono classifier.
        :param multi_threshold: Cloud probability threshold for the multi classifier.
        :param average_over: Size of the pixel neighbourhood used in the averaging post-processing step.
            A value of `0` skips this post-processing step. Default value mimics the default for
            s2cloudless: `4`.
        :param dilation_size: Size of the dilation post-processing step. A value of `0` or `None` skips this
            post-processing step. Default value mimics the default for s2cloudless: `2`.
        :param mono_classifier: Classifier used for mono-temporal cloud detection (`s2cloudless` or equivalent).
            Must work on the 10 selected reflectance bands as features `("B01", "B02", "B04", "B05", "B08", "B8A",
            "B09", "B10", "B11", "B12")`. Default value: `None` (s2cloudless is used)
        :param multi_classifier: Classifier used for multi-temporal cloud detection.
            Must work on the 90 multi-temporal features:

            - raw reflectance value in the target frame,
            - average value within a spatial window in the target frame,
            - maximum, mean and standard deviation of the structural similarity (SSIM)
            - indices between a spatial window in the target frame and every other,
            - minimum and mean reflectance of all available time frames,
            - maximum and mean difference in reflectances between the target frame and every other.

            Default value: None (SSIM-based model is used)
        """
        self.proc_resolution = self._parse_resolution_arg(processing_resolution)

        self._mono_classifier = mono_classifier
        self._multi_classifier = multi_classifier

        self.data_feature = self.parse_feature(data_feature)
        self.is_data_feature = self.parse_feature(is_data_feature)
        self.band_indices = (0, 1, 3, 4, 7, 8, 9, 10, 11, 12) if all_bands else tuple(range(10))

        self.max_proc_frames = max_proc_frames

        if mono_features is not None and isinstance(mono_features, tuple):
            self.mono_features = mono_features
        else:
            self.mono_features = (None, None)

        if multi_features is not None and isinstance(multi_features, tuple):
            self.multi_features = multi_features
        else:
            self.multi_features = (None, None)

        self.mask_feature = mask_feature if mask_feature is None else self.parse_feature(mask_feature)

        self.mono_threshold = mono_threshold
        self.multi_threshold = multi_threshold

        if average_over is not None and average_over > 0:
            self.avg_kernel = disk(average_over) / np.sum(disk(average_over))
        else:
            self.avg_kernel = None

        if dilation_size is not None and dilation_size > 0:
            self.dil_kernel = disk(dilation_size).astype(np.uint8)
        else:
            self.dil_kernel = None

    @staticmethod
    def _parse_resolution_arg(resolution: None | float | tuple[float, float]) -> tuple[float, float] | None:
        """Parses initialization resolution argument"""
        if isinstance(resolution, (int, float)):
            resolution = resolution, resolution

        return resolution

    @property
    def mono_classifier(self) -> ClassifierType:
        """An instance of pre-trained mono-temporal cloud classifier. Loaded only the first time it is required."""
        if self._mono_classifier is None:
            path = os.path.join(self.MODELS_FOLDER, self.MONO_CLASSIFIER_NAME)
            self._mono_classifier = Booster(model_file=path)

        return self._mono_classifier

    @property
    def multi_classifier(self) -> ClassifierType:
        """An instance of pre-trained multi-temporal cloud classifier. Loaded only the first time it is required."""
        if self._multi_classifier is None:
            path = os.path.join(self.MODELS_FOLDER, self.MULTI_CLASSIFIER_NAME)
            self._multi_classifier = Booster(model_file=path)

        return self._multi_classifier

    @staticmethod
    def _run_prediction(classifier: ClassifierType, features: np.ndarray) -> np.ndarray:
        """Uses classifier object on given data"""
        is_booster = isinstance(classifier, Booster)

        predict_method = classifier.predict if is_booster else classifier.predict_proba
        prediction = execute_with_mp_lock(predict_method, features)

        return prediction if is_booster else prediction[..., 1]

    def _scale_factors(self, reference_shape: tuple[int, int], bbox: BBox) -> tuple[tuple[float, float], float]:
        """Compute the resampling factors for height and width of the input array and sigma

        :param reference_shape: Tuple specifying height and width in pixels of high-resolution array
        :param bbox: An EOPatch bounding box
        :return: Rescale factor for rows and columns
        """
        height, width = reference_shape
        res_x, res_y = bbox_to_resolution(bbox, width=width, height=height)

        process_res_x, process_res_y = (res_x, res_y) if self.proc_resolution is None else self.proc_resolution

        rescale = res_y / process_res_y, res_x / process_res_x
        sigma = 200 / (process_res_x + process_res_y)

        return rescale, sigma

    def _red_ssim(
        self,
        *,
        data_x: np.ndarray,
        data_y: np.ndarray,
        valid_mask: np.ndarray,
        mu1: np.ndarray,
        mu2: np.ndarray,
        sigma1_2: np.ndarray,
        sigma2_2: np.ndarray,
        sigma: float,
        const1: float = 1e-6,
        const2: float = 1e-5,
    ) -> np.ndarray:
        """Slightly reduced (pre-computed) SSIM computation"""
        # Increase precision and mask invalid regions
        valid_mask = valid_mask.astype(np.float64)
        data_x = data_x.astype(np.float64) * valid_mask
        data_y = data_y.astype(np.float64) * valid_mask

        # Init
        mu1_2 = mu1 * mu1
        mu2_2 = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma12 = cv2.GaussianBlur(data_x * data_y, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        sigma12 -= mu1_mu2

        # Formula
        numerator = (2.0 * mu1_mu2 + const1) * (2.0 * sigma12 + const2)
        denominator = (mu1_2 + mu2_2 + const1) * (sigma1_2 + sigma2_2 + const2)

        return np.divide(numerator, denominator)

    def _win_avg(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Spatial window average"""
        return cv2.GaussianBlur(data.astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    def _win_prevar(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Incomplete spatial window variance"""
        return cv2.GaussianBlur((data * data).astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    def _average(self, data: np.ndarray) -> np.ndarray:
        return cv2.filter2D(data.astype(np.float64), -1, self.avg_kernel, borderType=cv2.BORDER_REFLECT)

    def _dilate(self, data: np.ndarray) -> np.ndarray:
        return (cv2.dilate(data.astype(np.uint8), self.dil_kernel) > 0).astype(np.uint8)

    def _average_all(self, data: np.ndarray) -> np.ndarray:
        """Average over each spatial slice of data"""
        if self.avg_kernel is not None:
            return _apply_to_spatial_axes(self._average, data, (1, 2))

        return data

    def _dilate_all(self, data: np.ndarray) -> np.ndarray:
        """Dilate over each spatial slice of data"""
        if self.dil_kernel is not None:
            return _apply_to_spatial_axes(self._dilate, data, (1, 2))

        return data

    def _ssim_stats(
        self,
        bands: np.ndarray,
        is_data: np.ndarray,
        local_avg: np.ndarray,
        local_var: np.ndarray,
        rel_tdx: int,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate SSIM stats"""
        ssim_max = np.empty((1, *bands.shape[1:]), dtype=np.float32)
        ssim_mean = np.empty_like(ssim_max)
        ssim_std = np.empty_like(ssim_max)

        bands_r = np.delete(bands, rel_tdx, axis=0)
        win_avg_r = np.delete(local_avg, rel_tdx, axis=0)
        var_r = np.delete(local_var, rel_tdx, axis=0)

        n_frames, _, _, n_bands = bands_r.shape

        valid_mask = np.delete(is_data, rel_tdx, axis=0) & is_data[rel_tdx, ..., 0].reshape(1, *is_data.shape[1:-1], 1)

        for b_i in range(n_bands):
            local_ssim = []

            for t_j in range(n_frames):
                ssim_ij = self._red_ssim(
                    data_x=bands[rel_tdx, ..., b_i],
                    data_y=bands_r[t_j, ..., b_i],
                    valid_mask=valid_mask[t_j, ..., 0],
                    mu1=local_avg[rel_tdx, ..., b_i],
                    mu2=win_avg_r[t_j, ..., b_i],
                    sigma1_2=local_var[rel_tdx, ..., b_i],
                    sigma2_2=var_r[t_j, ..., b_i],
                    sigma=sigma,
                )

                local_ssim.append(ssim_ij)

            local_ssim = np.ma.array(np.stack(local_ssim), mask=~valid_mask)

            ssim_max[0, ..., b_i] = np.ma.max(local_ssim, axis=0).data
            ssim_mean[0, ..., b_i] = np.ma.mean(local_ssim, axis=0).data
            ssim_std[0, ..., b_i] = np.ma.std(local_ssim, axis=0).data

        return ssim_max, ssim_mean, ssim_std

    def _do_single_temporal_cloud_detection(self, bands: np.ndarray) -> np.ndarray:
        """Performs a cloud detection process on each scene separately"""
        n_times, height, width, n_bands = bands.shape
        img_size = height * width
        mono_proba = np.empty(n_times * img_size)

        for t_i in range(0, n_times, self.max_proc_frames):
            # Extract mono features
            nt_min = t_i
            nt_max = min(t_i + self.max_proc_frames, n_times)

            mono_features = bands[nt_min:nt_max].reshape(-1, n_bands)

            mono_proba[nt_min * img_size : nt_max * img_size] = self._run_prediction(
                self.mono_classifier, mono_features
            )

        return mono_proba[..., None]

    def _do_multi_temporal_cloud_detection(self, bands: np.ndarray, is_data: np.ndarray, sigma: float) -> np.ndarray:
        """Performs a cloud detection process on multiple scenes at once"""
        n_times, height, width, n_bands = bands.shape
        img_size = height * width
        multi_proba = np.empty(n_times * img_size)

        local_avg: np.ndarray | None = None
        local_var: np.ndarray | None = None
        prev_left: int | None = None
        prev_right: int | None = None

        for t_idx in range(n_times):
            # Extract temporal window indices
            left, right = _get_window_indices(n_times, t_idx, self.max_proc_frames)

            bands_slice = bands[left:right]
            is_data_slice = is_data[left:right]
            masked_bands = np.ma.array(bands_slice, mask=~is_data_slice.repeat(n_bands, axis=-1))

            # Calculate the averages/variances for the local (windowed) streaming data
            if local_avg is None or (left, right) != (prev_left, prev_right):
                local_avg, local_var = self._update_batches(local_avg, local_var, bands_slice, is_data_slice, sigma)

            local_var = cast(np.ndarray, local_var)

            # Interweave and concatenate
            multi_features = self._extract_multi_features(
                bands_slice, is_data_slice, local_avg, local_var, t_idx - left, masked_bands, sigma
            )

            multi_proba[t_idx * img_size : (t_idx + 1) * img_size] = self._run_prediction(
                self.multi_classifier, multi_features
            )

            prev_left, prev_right = left, right

        return multi_proba[..., None]

    def _update_batches(
        self,
        local_avg: np.ndarray | None,
        local_var: np.ndarray | None,
        bands: np.ndarray,
        is_data: np.ndarray,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates or updates the window average and variance. The calculation is done per 2D image along the
        temporal and band axes."""
        local_avg_func = partial(self._win_avg, sigma=sigma)
        local_var_func = partial(self._win_prevar, sigma=sigma)

        # take full batch if avg/var don't exist, otherwise take only last index slice
        data = bands if local_avg is None else bands[-1][np.newaxis, ...]
        data_mask = is_data if local_avg is None else is_data[-1][np.newaxis, ...]

        avg_data = _apply_to_spatial_axes(local_avg_func, data, (1, 2))
        avg_data_mask = _apply_to_spatial_axes(local_avg_func, data_mask, (1, 2))
        avg_data_mask[avg_data_mask == 0.0] = 1.0

        var_data = _apply_to_spatial_axes(local_var_func, data, (1, 2))

        if local_avg is None or local_var is None:
            local_avg = cast(np.ndarray, avg_data / avg_data_mask)
            local_var = cast(np.ndarray, var_data - local_avg**2)
            return local_avg, local_var

        # shift back, drop first element
        local_avg[:-1] = local_avg[1:]
        local_var[:-1] = local_var[1:]

        # set new element
        local_avg[-1] = (avg_data / avg_data_mask)[0]
        local_var[-1] = var_data[0] - local_avg[-1] ** 2

        return local_avg, local_var

    def _extract_multi_features(  # pylint: disable=too-many-locals
        self,
        bands: np.ndarray,
        is_data: np.ndarray,
        local_avg: np.ndarray,
        local_var: np.ndarray,
        local_t_idx: int,
        masked_bands: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Extracts features for a batch"""
        # Compute SSIM stats
        ssim_max, ssim_mean, ssim_std = self._ssim_stats(bands, is_data, local_avg, local_var, local_t_idx, sigma)

        # Compute temporal stats
        temp_min = np.ma.min(masked_bands, axis=0).data[np.newaxis, ...]
        temp_mean = np.ma.mean(masked_bands, axis=0).data[np.newaxis, ...]

        # Compute difference stats
        t_all = len(bands)

        diff_max = (masked_bands[local_t_idx][np.newaxis, ...] - temp_min).data
        coef1 = 1.0 + 1.0 / (t_all - 1)
        coef2 = t_all * temp_mean / (t_all - 1)
        diff_mean = (masked_bands[local_t_idx][np.newaxis, ...] * coef1 - coef2).data

        # Interweave
        ssim_interweaved = np.empty((*ssim_max.shape[:-1], 3 * ssim_max.shape[-1]))
        ssim_interweaved[..., 0::3] = ssim_max
        ssim_interweaved[..., 1::3] = ssim_mean
        ssim_interweaved[..., 2::3] = ssim_std

        temp_interweaved = np.empty((*temp_min.shape[:-1], 2 * temp_min.shape[-1]))
        temp_interweaved[..., 0::2] = temp_min
        temp_interweaved[..., 1::2] = temp_mean

        diff_interweaved = np.empty((*diff_max.shape[:-1], 2 * diff_max.shape[-1]))
        diff_interweaved[..., 0::2] = diff_max
        diff_interweaved[..., 1::2] = diff_mean

        # Put it all together
        multi_features = np.concatenate(
            (
                bands[local_t_idx][np.newaxis, ...],
                local_avg[local_t_idx][np.newaxis, ...],
                ssim_interweaved,
                temp_interweaved,
                diff_interweaved,
            ),
            axis=3,
        )

        return multi_features.reshape(-1, multi_features.shape[-1])

    def execute(self, eopatch: EOPatch) -> EOPatch:  # noqa: C901
        """Add selected features (cloud probabilities and masks) to an EOPatch instance.

        :param eopatch: Input `EOPatch` instance
        :return: `EOPatch` with additional features
        """
        bands = eopatch[self.data_feature][..., self.band_indices].astype(np.float32)

        is_data = eopatch[self.is_data_feature].astype(bool)

        image_size = bands.shape[1:-1]
        patch_bbox = eopatch.bbox
        if patch_bbox is None:
            raise ValueError("Cannot run cloud masking on an EOPatch without a BBox.")
        scale_factors, sigma = self._scale_factors(image_size, patch_bbox)

        is_data_sm = is_data
        # Downscale if specified
        if scale_factors is not None:
            bands = resize_images(bands.astype(np.float32), scale_factors=scale_factors)
            is_data_sm = resize_images(is_data.astype(np.uint8), scale_factors=scale_factors).astype(bool)

        mono_proba_feature, mono_mask_feature = self.mono_features
        multi_proba_feature, multi_mask_feature = self.multi_features

        # Run s2cloudless if needed
        if any([self.mask_feature, mono_mask_feature, mono_proba_feature]):
            mono_proba = self._do_single_temporal_cloud_detection(bands)
            mono_proba = mono_proba.reshape(*bands.shape[:-1], 1)

            # Upscale if necessary
            if scale_factors is not None:
                mono_proba = resize_images(mono_proba, new_size=image_size)

            # Average over and threshold
            mono_mask = self._average_all(mono_proba) >= self.mono_threshold

        # Run SSIM-based multi-temporal classifier if needed
        if any([self.mask_feature, multi_mask_feature, multi_proba_feature]):
            multi_proba = self._do_multi_temporal_cloud_detection(bands, is_data_sm, sigma)
            multi_proba = multi_proba.reshape(*bands.shape[:-1], 1)

            # Upscale if necessary
            if scale_factors is not None:
                multi_proba = resize_images(multi_proba, new_size=image_size)

            # Average over and threshold
            multi_mask = self._average_all(multi_proba) >= self.multi_threshold

        if mono_mask_feature is not None:
            mono_mask = self._dilate_all(mono_mask)
            eopatch.mask[mono_mask_feature] = (mono_mask * is_data).astype(bool)

        if multi_mask_feature is not None:
            multi_mask = self._dilate_all(multi_mask)
            eopatch.mask[multi_mask_feature] = (multi_mask * is_data).astype(bool)

        # Intersect
        if self.mask_feature:
            inter_mask = mono_mask & multi_mask
            inter_mask = self._dilate_all(inter_mask)
            eopatch[self.mask_feature] = (inter_mask * is_data).astype(bool)

        if mono_proba_feature is not None:
            eopatch.data[mono_proba_feature] = (mono_proba * is_data).astype(np.float32)

        if multi_proba_feature is not None:
            eopatch.data[multi_proba_feature] = (multi_proba * is_data).astype(np.float32)

        return eopatch


def _get_window_indices(num_of_elements: int, middle_idx: int, window_size: int) -> tuple[int, int]:
    """
    Returns the minimum and maximum indices to be used for indexing, lower inclusive and upper exclusive.
    The window has the following properties:
        1. total size is `window_size` (unless there are not enough frames)
        2. centered around `middle_idx` if possible, otherwise shifted so that the window is contained without reducing
            it's size.
    """
    if window_size >= num_of_elements:
        return 0, num_of_elements

    # Construct window (is not necessarily contained)
    min_frame = middle_idx - window_size // 2
    max_frame = min_frame + window_size

    # Shift window so that it is inside [0, num_all_frames].
    # Only one of the following can happen because `window_size < num_of_elements`
    if min_frame < 0:
        return 0, window_size
    if max_frame >= num_of_elements:
        return num_of_elements - window_size, num_of_elements

    return min_frame, max_frame
