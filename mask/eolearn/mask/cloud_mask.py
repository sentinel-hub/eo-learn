"""
Module for cloud masking

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import logging
import os
from functools import partial
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from lightgbm import Booster
from skimage.morphology import disk
from typing_extensions import Protocol

from sentinelhub import BBox, bbox_to_resolution

from eolearn.core import EOTask, FeatureType, execute_with_mp_lock

from .utils import map_over_axis, resize_images

LOGGER = logging.getLogger(__name__)


class ClassifierType(Protocol):
    """Defines the necessary classifier interface."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=missing-function-docstring,invalid-name
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=missing-function-docstring,invalid-name
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
        data_feature: Tuple[FeatureType, str] = (FeatureType.DATA, "BANDS-S2-L1C"),
        is_data_feature: Tuple[FeatureType, str] = (FeatureType.MASK, "IS_DATA"),
        all_bands: bool = True,
        processing_resolution: Union[None, float, Tuple[float, float]] = None,
        max_proc_frames: int = 11,
        mono_features=None,
        multi_features=None,
        mask_feature: Optional[Tuple[FeatureType, str]] = (FeatureType.MASK, "CLM_INTERSSIM"),
        mono_threshold: float = 0.4,
        multi_threshold: float = 0.5,
        average_over: Optional[int] = 4,
        dilation_size: Optional[int] = 2,
        mono_classifier: Optional[ClassifierType] = None,
        multi_classifier: Optional[ClassifierType] = None,
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
        :type mono_features: (str or None, str or None)
        :param multi_features: Tuple of keys used for storing cloud probabilities and masks of the multi classifier.
            The probabilities are added as a data feature, while masks are added as a mask feature. By default,
            none of them are added.
        :type multi_features: (str or None, str or None)
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
        :type mono_classifier: lightgbm.Booster or sklearn.base.BaseEstimator
        :param multi_classifier: Classifier used for multi-temporal cloud detection.
            Must work on the 90 multi-temporal features:

            - raw reflectance value in the target frame,
            - average value within a spatial window in the target frame,
            - maximum, mean and standard deviation of the structural similarity (SSIM)
            - indices between a spatial window in the target frame and every other,
            - minimum and mean reflectance of all available time frames,
            - maximum and mean difference in reflectances between the target frame and every other.

            Default value: None (SSIM-based model is used)
        :type multi_classifier: lightgbm.Booster or sklearn.base.BaseEstimator
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
    def _parse_resolution_arg(resolution: Union[None, float, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
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

    def _scale_factors(self, reference_shape: Tuple[int, int], bbox: BBox) -> Tuple[Tuple[float, float], float]:
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

    def _red_ssim(self, *, data_x, data_y, valid_mask, mu1, mu2, sigma1_2, sigma2_2, const1=1e-6, const2=1e-5, sigma):
        """Slightly reduced (pre-computed) SSIM computation"""
        # Increase precision and mask invalid regions
        valid_mask = valid_mask.astype(np.float64)
        data_x = data_x.astype(np.float64) * valid_mask
        data_y = data_y.astype(np.float64) * valid_mask

        # Init
        mu1_2 = mu1 * mu1
        mu2_2 = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma12 = cv2.GaussianBlur((data_x * data_y).astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        sigma12 -= mu1_mu2

        # Formula
        tmp1 = 2.0 * mu1_mu2 + const1
        tmp2 = 2.0 * sigma12 + const2
        num = tmp1 * tmp2

        tmp1 = mu1_2 + mu2_2 + const1
        tmp2 = sigma1_2 + sigma2_2 + const2
        den = tmp1 * tmp2

        return np.divide(num, den)

    def _win_avg(self, data, sigma):
        """Spatial window average"""
        return cv2.GaussianBlur(data.astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    def _win_prevar(self, data, sigma):
        """Incomplete spatial window variance"""
        return cv2.GaussianBlur((data * data).astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    def _average(self, data):
        return cv2.filter2D(data.astype(np.float64), -1, self.avg_kernel, borderType=cv2.BORDER_REFLECT)

    def _dilate(self, data):
        return (cv2.dilate(data.astype(np.uint8), self.dil_kernel) > 0).astype(np.uint8)

    @staticmethod
    def _map_sequence(data, func2d):
        """Iterate over time and band dimensions and apply a function to each slice.
        Returns a new array with the combined results.

        :param data: input array
        :type data: array of shape (timestamps, rows, columns, channels)
        :param func2d: Mapping function that is applied on each 2d image slice. All outputs must have the same shape.
        :type func2d: function (rows, columns) -> (new_rows, new_columns)
        """
        # TODO: this should be replaced with `eolearn.features.utils._apply_to_spatial_axes`

        # Map over channel dimension on 3d tensor
        def func3d(dim):
            return map_over_axis(dim, func2d, axis=2)

        # Map over time dimension on 4d tensor
        def func4d(dim):
            return map_over_axis(dim, func3d, axis=0)

        output = func4d(data)

        return output

    def _average_all(self, data):
        """Average over each spatial slice of data"""
        if self.avg_kernel is not None:
            return self._map_sequence(data, self._average)

        return data

    def _dilate_all(self, data):
        """Dilate over each spatial slice of data"""
        if self.dil_kernel is not None:
            return self._map_sequence(data, self._dilate)

        return data

    def _ssim_stats(self, bands, is_data, win_avg, var, nt_rel, sigma):
        """Calculate SSIM stats"""
        ssim_max = np.empty((1, *bands.shape[1:]), dtype=np.float32)
        ssim_mean = np.empty_like(ssim_max)
        ssim_std = np.empty_like(ssim_max)

        bands_r = np.delete(bands, nt_rel, axis=0)
        win_avg_r = np.delete(win_avg, nt_rel, axis=0)
        var_r = np.delete(var, nt_rel, axis=0)

        n_frames = bands_r.shape[0]
        n_bands = bands_r.shape[-1]

        valid_mask = np.delete(is_data, nt_rel, axis=0) & is_data[nt_rel, ..., 0].reshape(1, *is_data.shape[1:-1], 1)

        for b_i in range(n_bands):
            local_ssim = []

            for t_j in range(n_frames):
                ssim_ij = self._red_ssim(
                    data_x=bands[nt_rel, ..., b_i],
                    data_y=bands_r[t_j, ..., b_i],
                    valid_mask=valid_mask[t_j, ..., 0],
                    mu1=win_avg[nt_rel, ..., b_i],
                    mu2=win_avg_r[t_j, ..., b_i],
                    sigma1_2=var[nt_rel, ..., b_i],
                    sigma2_2=var_r[t_j, ..., b_i],
                    sigma=sigma,
                )

                local_ssim.append(ssim_ij)

            local_ssim = np.ma.array(np.stack(local_ssim), mask=~valid_mask)

            ssim_max[0, ..., b_i] = np.ma.max(local_ssim, axis=0).data
            ssim_mean[0, ..., b_i] = np.ma.mean(local_ssim, axis=0).data
            ssim_std[0, ..., b_i] = np.ma.std(local_ssim, axis=0).data

        return ssim_max, ssim_mean, ssim_std

    def _do_single_temporal_cloud_detection(self, bands):
        """Performs a cloud detection process on each scene separately"""
        mono_proba = np.empty(np.prod(bands.shape[:-1]))
        img_size = np.prod(bands.shape[1:-1])

        n_times = bands.shape[0]

        for t_i in range(0, n_times, self.max_proc_frames):
            # Extract mono features
            nt_min = t_i
            nt_max = min(t_i + self.max_proc_frames, n_times)

            bands_t = bands[nt_min:nt_max]

            mono_features = bands_t.reshape(np.prod(bands_t.shape[:-1]), bands_t.shape[-1])

            mono_proba[nt_min * img_size : nt_max * img_size] = self._run_prediction(
                self.mono_classifier, mono_features
            )

        return mono_proba[..., np.newaxis]

    def _do_multi_temporal_cloud_detection(self, bands, is_data, sigma):
        """Performs a cloud detection process on multiple scenes at once"""
        multi_proba = np.empty(np.prod(bands.shape[:-1]))
        img_size = np.prod(bands.shape[1:-1])

        n_times = bands.shape[0]

        loc_mu = None
        loc_var = None

        prev_nt_min = None
        prev_nt_max = None

        for t_i in range(n_times):
            # Extract temporal window indices
            nt_min, nt_max = _get_window_indices(n_times, t_i, self.max_proc_frames)
            rel_t_i = t_i - nt_min

            bands_t = bands[nt_min:nt_max]
            is_data_t = is_data[nt_min:nt_max]

            masked_bands = np.ma.array(bands_t, mask=~is_data_t.repeat(bands_t.shape[-1], axis=-1))

            # Add window averages and variances to local data
            if loc_mu is None or prev_nt_min != nt_min or prev_nt_max != nt_max:
                loc_mu, loc_var = self._update_batches(loc_mu, loc_var, bands_t, is_data_t, sigma)

            # Interweave and concatenate
            multi_features = self._extract_multi_features(
                bands_t, is_data_t, loc_mu, loc_var, rel_t_i, masked_bands, sigma
            )

            multi_proba[t_i * img_size : (t_i + 1) * img_size] = self._run_prediction(
                self.multi_classifier, multi_features
            )

            prev_nt_min = nt_min
            prev_nt_max = nt_max

        return multi_proba[..., np.newaxis]

    def _update_batches(self, loc_mu, loc_var, bands_t, is_data_t, sigma):
        """Updates window variance and mean values for a batch"""
        # Add window averages and variances to local data
        if loc_mu is None:
            win_avg_bands = self._map_sequence(bands_t, partial(self._win_avg, sigma=sigma))
            win_avg_is_data = self._map_sequence(is_data_t, partial(self._win_avg, sigma=sigma))

            win_avg_is_data[win_avg_is_data == 0.0] = 1.0

            loc_mu = win_avg_bands / win_avg_is_data

            win_prevars = self._map_sequence(bands_t, partial(self._win_prevar, sigma=sigma))
            win_prevars -= loc_mu * loc_mu

            loc_var = win_prevars

        else:
            win_avg_bands = self._map_sequence(bands_t[-1][None, ...], partial(self._win_avg, sigma=sigma))
            win_avg_is_data = self._map_sequence(is_data_t[-1][None, ...], partial(self._win_avg, sigma=sigma))

            win_avg_is_data[win_avg_is_data == 0.0] = 1.0

            loc_mu[:-1] = loc_mu[1:]
            loc_mu[-1] = (win_avg_bands / win_avg_is_data)[0]

            win_prevars = self._map_sequence(bands_t[-1][None, ...], partial(self._win_prevar, sigma=sigma))
            win_prevars[0] -= loc_mu[-1] * loc_mu[-1]

            loc_var[:-1] = loc_var[1:]
            loc_var[-1] = win_prevars[0]

        return loc_mu, loc_var

    def _extract_multi_features(self, bands_t, is_data_t, loc_mu, loc_var, nt_rel, masked_bands, sigma):
        """Extracts features for a batch"""
        # Compute SSIM stats
        ssim_max, ssim_mean, ssim_std = self._ssim_stats(bands_t, is_data_t, loc_mu, loc_var, nt_rel, sigma)

        # Compute temporal stats
        temp_min = np.ma.min(masked_bands, axis=0).data[None, ...]
        temp_mean = np.ma.mean(masked_bands, axis=0).data[None, ...]

        # Compute difference stats
        t_all = len(bands_t)

        diff_max = (masked_bands[nt_rel][None, ...] - temp_min).data
        diff_mean = (masked_bands[nt_rel][None, ...] * (1.0 + 1.0 / (t_all - 1)) - t_all * temp_mean / (t_all - 1)).data

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
                bands_t[nt_rel][None, ...],
                loc_mu[nt_rel][None, ...],
                ssim_interweaved,
                temp_interweaved,
                diff_interweaved,
            ),
            axis=3,
        )

        multi_features = multi_features.reshape(np.prod(multi_features.shape[:-1]), multi_features.shape[-1])

        return multi_features

    def execute(self, eopatch):
        """Add selected features (cloud probabilities and masks) to an EOPatch instance.

        :param eopatch: Input `EOPatch` instance
        :return: `EOPatch` with additional features
        """
        bands = eopatch[self.data_feature][..., self.band_indices].astype(np.float32)

        is_data = eopatch[self.is_data_feature].astype(bool)

        original_shape = bands.shape[1:-1]
        scale_factors, sigma = self._scale_factors(original_shape, eopatch.bbox)

        is_data_sm = is_data
        # Downscale if specified
        if scale_factors is not None:
            bands = resize_images(bands.astype(np.float32), scale_factors=scale_factors)
            is_data_sm = resize_images(is_data.astype(np.uint8), scale_factors=scale_factors).astype(bool)

        mono_proba = None
        multi_proba = None
        mono_proba_feature, mono_mask_feature = self.mono_features
        multi_proba_feature, multi_mask_feature = self.multi_features

        # Run s2cloudless if needed
        if any([self.mask_feature, mono_mask_feature, mono_proba_feature]):
            mono_proba = self._do_single_temporal_cloud_detection(bands)
            mono_proba = mono_proba.reshape(*bands.shape[:-1], 1)

            # Upscale if necessary
            if scale_factors is not None:
                mono_proba = resize_images(mono_proba, new_size=original_shape)

            # Average over and threshold
            mono_mask = self._average_all(mono_proba) >= self.mono_threshold

        # Run SSIM-based multi-temporal classifier if needed
        if any([self.mask_feature, multi_mask_feature, multi_proba_feature]):
            multi_proba = self._do_multi_temporal_cloud_detection(bands, is_data_sm, sigma)
            multi_proba = multi_proba.reshape(*bands.shape[:-1], 1)

            # Upscale if necessary
            if scale_factors is not None:
                multi_proba = resize_images(multi_proba, new_size=original_shape)

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


def _get_window_indices(num_of_elements: int, middle_idx: int, window_size: int) -> Tuple[int, int]:
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
