"""
Module for cloud masking

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import logging

import joblib
import numpy as np
import cv2
from skimage.morphology import disk
from s2cloudless import S2PixelCloudDetector, MODEL_EVALSCRIPT
from sentinelhub import WmsRequest, WcsRequest, DataSource, CustomUrlParam, MimeType, ServiceType, bbox_to_resolution

from eolearn.core import EOTask, get_common_timestamps, FeatureType, execute_with_mp_lock
from .utilities import resize_images, map_over_axis


INTERP_METHODS = ['nearest', 'linear']

LOGGER = logging.getLogger(__name__)


class AddCloudMaskTask(EOTask):
    """ Task to add a cloud mask and cloud probability map to an EOPatch

    This task computes a cloud probability map and corresponding cloud binary mask for the input EOPatch. The classifier
    to be used to compute such maps must be provided at declaration. The `data_feature` to be used as input to the
    classifier is also a mandatory argument. If `data_feature` exists already, downscaling to the given (lower) cloud
    mask resolution is performed, the classifier is run, and upsampling returns the cloud maps to the original
    resolution.
    Otherwise, if `data_feature` does not exist, a new OGC request at the given cloud mask resolution is made, the
    classifier is run, and upsampling returns the cloud masks to original resolution. This design should allow faster
    execution of the classifier, and reduce the number of requests. `linear` interpolation is used for resampling of
    the `data_feature` and cloud probability map, while `nearest` interpolation is used to upsample the binary cloud
    mask.

    This implementation should allow usage with any cloud detector implemented for different data sources (S2, L8, ..).
    """
    def __init__(self, classifier, data_feature, cm_size_x=None, cm_size_y=None, cmask_feature='CLM',
                 cprobs_feature=None, instance_id=None, data_source=DataSource.SENTINEL2_L1C,
                 image_format=MimeType.TIFF_d32f, model_evalscript=MODEL_EVALSCRIPT):
        """ Constructor

        If both `cm_size_x` and `cm_size_y` are `None` and `data_feature` exists, cloud detection is computed at same
        resolution of `data_feature`.

        :param classifier: Cloud detector classifier. This object implements a `get_cloud_probability_map` and
                            `get_cloud_masks` functions to generate probability maps and binary masks
        :param data_feature: Name of key in eopatch.data dictionary to be used as input to the classifier. If the
                           `data_feature` does not exist, a new OGC request at the given cloud mask resolution is made
                           with layer name set to `data_feature` parameter.
        :param cm_size_x: Resolution to be used for computation of cloud mask. Allowed values are number of column
                            pixels (WMS-request) or spatial resolution (WCS-request, e.g. '10m'). Default is `None`
        :param cm_size_y: Resolution to be used for computation of cloud mask. Allowed values are number of row
                            pixels (WMS-request) or spatial resolution (WCS-request, e.g. '10m'). Default is `None`
        :param cmask_feature: Name of key to be used for the cloud mask to add. The cloud binary mask is added to the
                            `eopatch.mask` attribute dictionary. Default is `'clm'`.
        :param cprobs_feature: Name of key to be used for the cloud probability map to add. The cloud probability map is
                            added to the `eopatch.data` attribute dictionary. Default is `None`, so no cloud
                            probability map will be computed.
        :param instance_id: Instance ID to be used for OGC request. Default is `None`
        :param data_source: Data source to be requested by OGC service request. Default is `DataSource.SENTINEL2_L1C`
        :param image_format: Image format to be requested by OGC service request. Default is `MimeType.TIFF_d32f`
        :param model_evalscript: CustomUrlParam defining the EVALSCRIPT to be used by OGC request. Should reflect the
                            request necessary for the correct functioning of the classifier. For instance, for the
                            `S2PixelCloudDetector` classifier, `MODEL_EVALSCRIPT` is used as it requests the required 10
                            bands. Default is `MODEL_EVALSCRIPT`
        """
        self.classifier = classifier
        self.data_feature = data_feature
        self.cm_feature = cmask_feature
        self.cm_size_x = cm_size_x
        self.cm_size_y = cm_size_y
        self.cprobs_feature = cprobs_feature
        self.instance_id = instance_id
        self.data_source = data_source
        self.image_format = image_format
        self.model_evalscript = model_evalscript

    def _get_wms_request(self, bbox, time_interval, size_x, size_y, maxcc, time_difference, custom_url_params):
        """
        Returns WMS request.
        """
        return WmsRequest(layer=self.data_feature,
                          bbox=bbox,
                          time=time_interval,
                          width=size_x,
                          height=size_y,
                          maxcc=maxcc,
                          custom_url_params=custom_url_params,
                          time_difference=time_difference,
                          image_format=self.image_format,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _get_wcs_request(self, bbox, time_interval, size_x, size_y, maxcc, time_difference, custom_url_params):
        """
        Returns WCS request.
        """
        return WcsRequest(layer=self.data_feature,
                          bbox=bbox,
                          time=time_interval,
                          resx=size_x, resy=size_y,
                          maxcc=maxcc,
                          custom_url_params=custom_url_params,
                          time_difference=time_difference,
                          image_format=self.image_format,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _get_rescale_factors(self, reference_shape, meta_info):
        """ Compute the resampling factor for height and width of the input array

        :param reference_shape: Tuple specifying height and width in pixels of high-resolution array
        :type reference_shape: tuple of ints
        :param meta_info: Meta-info dictionary of input eopatch. Defines OGC request and parameters used to create the
                            eopatch
        :return: Rescale factor for rows and columns
        :rtype: tuple of floats
        """
        # Figure out resampling size
        height, width = reference_shape

        service_type = ServiceType(meta_info['service_type'])
        rescale = None
        if service_type == ServiceType.WMS:

            if (self.cm_size_x is None) and (self.cm_size_y is not None):
                rescale = (self.cm_size_y / height, self.cm_size_y / height)
            elif (self.cm_size_x is not None) and (self.cm_size_y is None):
                rescale = (self.cm_size_x / width, self.cm_size_x / width)
            else:
                rescale = (self.cm_size_y / height, self.cm_size_x / width)

        elif service_type == ServiceType.WCS:
            # Case where only one resolution for cloud masks is specified in WCS
            if self.cm_size_y is None:
                self.cm_size_y = self.cm_size_x
            elif self.cm_size_x is None:
                self.cm_size_x = self.cm_size_y

            hr_res_x, hr_res_y = int(meta_info['size_x'].strip('m')), int(meta_info['size_y'].strip('m'))
            lr_res_x, lr_res_y = int(self.cm_size_x.strip('m')), int(self.cm_size_y.strip('m'))
            rescale = (hr_res_y / lr_res_y, hr_res_x / lr_res_x)

        return rescale

    def _downscaling(self, hr_array, meta_info, interp='linear', smooth=True):
        """ Downscale existing array to resolution requested by cloud detector

        :param hr_array: High-resolution data array to be downscaled
        :param meta_info: Meta-info of eopatch
        :param interp: Interpolation method to be used in downscaling. Default is `'linear'`
        :param smooth: Apply Gaussian smoothing in spatial directions before downscaling. Sigma of kernel is estimated
                        by rescaling factor. Default is `True`
        :return: Down-scaled array
        """
        # Run cloud mask on full resolution
        if (self.cm_size_y is None) and (self.cm_size_x is None):
            return hr_array, None

        # Rescaling factor in spatial (width, height) dimensions
        rescale = self._get_rescale_factors(hr_array.shape[1:3], meta_info)

        lr_array = resize_images(hr_array,
                                 scale_factors=rescale,
                                 anti_alias=smooth,
                                 interpolation=interp)

        return lr_array, rescale

    @staticmethod
    def _upsampling(lr_array, rescale, reference_shape, interp='linear'):
        """ Upsample the low-resolution array to the original high-resolution grid

        :param lr_array: Low-resolution array to be upsampled
        :param rescale: Rescale factor for rows/columns
        :param reference_shape: Original size of high-resolution eopatch. Tuple with dimension for time, height and
                                width
        :param interp: Interpolation method ot be used in upsampling. Default is `'linear'`
        :return: Upsampled array. The array has 4 dimensions, the last one being of size 1
        """
        lr_shape = lr_array.shape + (1,)

        if rescale is None:
            return lr_array.reshape(lr_shape)

        # Resize to reference shape (height, width)
        output_size = reference_shape[1:3]
        hr_array = resize_images(lr_array.reshape(lr_shape),
                                 new_size=output_size,
                                 interpolation=interp)

        return hr_array

    def _make_request(self, bbox, meta_info, timestamps):
        """ Make OGC request to create input for cloud detector classifier

        :param bbox: Bounding box
        :param meta_info: Meta-info dictionary of input eopatch
        :return: Requested data
        """
        service_type = ServiceType(meta_info['service_type'])

        # Raise error if resolutions are not specified
        if self.cm_size_x is None and self.cm_size_y is None:
            raise ValueError("Specify size_x and size_y for data request")

        # If WCS request, make sure both resolutions are set
        if service_type == ServiceType.WCS:
            if self.cm_size_y is None:
                self.cm_size_y = self.cm_size_x
            elif self.cm_size_x is None:
                self.cm_size_x = self.cm_size_y

        custom_url_params = {CustomUrlParam.SHOWLOGO: False,
                             CustomUrlParam.TRANSPARENT: False,
                             CustomUrlParam.EVALSCRIPT: self.model_evalscript}

        build_request = {ServiceType.WMS: self._get_wms_request,
                         ServiceType.WCS: self._get_wcs_request}[service_type]

        request = build_request(bbox,
                                meta_info['time_interval'],
                                self.cm_size_x,
                                self.cm_size_y,
                                meta_info['maxcc'],
                                meta_info['time_difference'],
                                custom_url_params)

        request_dates = request.get_dates()
        download_frames = get_common_timestamps(request_dates, timestamps)

        request_return = request.get_data(raise_download_errors=False, data_filter=download_frames)
        bad_data = [idx for idx, value in enumerate(request_return) if value is None]
        for idx in reversed(sorted(bad_data)):
            LOGGER.warning('Data from %s could not be downloaded for %s!', str(request_dates[idx]), self.data_feature)
            del request_return[idx]
            del request_dates[idx]

        return np.asarray(request_return), request_dates

    def execute(self, eopatch):
        """ Add cloud binary mask and (optionally) cloud probability map to input eopatch

        :param eopatch: Input `EOPatch` instance
        :return: `EOPatch` with additional cloud maps
        """
        # Downsample or make request
        if not eopatch.data:
            raise ValueError('EOPatch must contain some data feature')
        if self.data_feature in eopatch.data:
            new_data, rescale = self._downscaling(eopatch.data[self.data_feature], eopatch.meta_info)
            reference_shape = eopatch.data[self.data_feature].shape[:3]
        else:
            new_data, new_dates = self._make_request(eopatch.bbox, eopatch.meta_info, eopatch.timestamp)
            removed_frames = eopatch.consolidate_timestamps(new_dates)
            for rm_frame in removed_frames:
                LOGGER.warning('Removed data for frame %s from '
                               'eopatch due to unavailability of %s!', str(rm_frame), self.data_feature)

            # Get reference shape from first item in data dictionary
            if not eopatch.data:
                raise ValueError('Given EOPatch does not have any data feature')

            reference_data_feature = sorted(eopatch.data)[0]
            reference_shape = eopatch.data[reference_data_feature].shape[:3]
            rescale = self._get_rescale_factors(reference_shape[1:3], eopatch.meta_info)

        clf_probs_lr = execute_with_mp_lock(self.classifier.get_cloud_probability_maps, new_data)
        clf_mask_lr = self.classifier.get_mask_from_prob(clf_probs_lr)

        # Add cloud mask as a feature to EOPatch
        clf_mask_hr = self._upsampling(clf_mask_lr, rescale, reference_shape, interp='nearest')
        eopatch.mask[self.cm_feature] = clf_mask_hr.astype(np.bool)

        # If the feature name for cloud probability maps is specified, add as feature
        if self.cprobs_feature is not None:
            clf_probs_hr = self._upsampling(clf_probs_lr, rescale, reference_shape, interp='linear')
            eopatch.data[self.cprobs_feature] = clf_probs_hr.astype(np.float32)

        return eopatch


def get_s2_pixel_cloud_detector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True):
    """ Wrapper function for pixel-based S2 cloud detector `S2PixelCloudDetector`
    """
    return S2PixelCloudDetector(threshold=threshold,
                                average_over=average_over,
                                dilation_size=dilation_size,
                                all_bands=all_bands)


# Twin classifier
MONO_CLASSIFIER_NAME = 'pixel_s2_cloud_detector_lightGBM_v0.2.joblib.dat'
MULTI_CLASSIFIER_NAME = 'ssim_s2_cloud_detector_lightGBM_v0.2.joblib.dat'


class AddMultiCloudMaskTask(EOTask):
    """ This task wraps around s2cloudless and the SSIM-based multi-temporal classifier.
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
    ```python
    # Only output the combined mask
    task1 = AddMultiCloudMaskTask(processing_resolution='120m',
                                  mask_feature='CLM_INTERSSIM',
                                  average_over=16,
                                  dilation_size=8)

    # Only output monotemporal masks. Only monotemporal processing is done.
    task2 = AddMultiCloudMaskTask(processing_resolution='120m',
                                  mono_features=(None, 'CLM_S2C'),
                                  mask_feature=None,
                                  average_over=16,
                                  dilation_size=8)
    ```
    """

    # A temporary fix of too many arguments and class attributes
    # pylint: disable=R0902
    # pylint: disable=R0913

    MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'models')

    def __init__(self,
                 mono_classifier=None,
                 multi_classifier=None,
                 data_feature='BANDS-S2-L1C',
                 is_data_feature='IS_DATA',
                 all_bands=True,
                 processing_resolution=None,
                 max_proc_frames=11,
                 mono_features=None,
                 multi_features=None,
                 mask_feature='CLM_INTERSSIM',
                 mono_threshold=0.4,
                 multi_threshold=0.5,
                 average_over=1,
                 dilation_size=1):
        """
        :param mono_classifier: Classifier used for mono-temporal cloud detection (`s2cloudless` or equivalent).
                                Must work on the 10 selected reflectance bands as features
                                (`B01`, `B02`, `B04`, `B05`, `B08`, `B8A`, `B09`, `B10`, `B11`, `B12`)
                                Default value: None (s2cloudless is used)
        :type mono_classifier: sklearn Estimator
        :param multi_classifier: Classifier used for multi-temporal cloud detection.
                                 Must work on the 90 multi-temporal features:
                                    - raw reflectance value in the target frame,
                                    - average value within a spatial window in the target frame,
                                    - maximum, mean and standard deviation of the structural similarity (SSIM)
                                    - indices between a spatial window in the target frame and every other,
                                    - minimum and mean reflectance of all available time frames,
                                    - maximum and mean difference in reflectances between the target frame
                                      and every other.
                                 Default value: None (SSIM-based model is used)
        :type multi_classifier: sklearn Estimator
        :param data_feature: Name of the key in the `eopatch.data` dictionary, which stores raw reflectance data.
                             Default value:  `'BANDS-S2-L1C'`.
        :type data_feature: str
        :param is_data_feature: Name of the key in the `eopatch.mask` dictionary, which indicates whether data is valid.
                                Default value: `'IS_DATA'`.
        :type is_data_feature: str
        :param all_bands: Flag, which indicates whether images will consist of all 13 Sentinel-2 bands or only
                          the required 10. Default value:  `True`.
        :type all_bands: bool
        :param processing_resolution: Resolution to be used during the computation of cloud probabilities and masks,
                                      expressed in meters. Resolution is given as a pair of x and y resolutions.
                                      If a single value is given, it is used for both dimensions.
                                      Default is `None` (source resolution).
        :type processing_resolution: int or (int, int)
        :param max_proc_frames: Maximum number of frames (including the target, for multi-temporal classification)
                                considered in a single batch iteration (To keep memory usage at agreeable levels,
                                the task operates on smaller batches of time frames). Default value:  `11`.
        :type max_proc_frames: int
        :param mono_features: Tuple of keys to be used for storing cloud probabilities and masks of the mono classifier.
                              The probabilities are added to the `eopatch.data` attribute dictionary, while masks are
                              added to `eopatch.mask`. By default, none of them are added.
        :type mono_features: (str | None, str | None)
        :param multi_features: Tuple of keys used for storing cloud probabilities and masks of the multi classifier.
                               The probabilities are added to the `eopatch.data` attribute dictionary, while masks are
                               added to `eopatch.mask`. By default, none of them are added.
        :type multi_features: (str | None, str | None)
        :param mask_feature: Name of the output intersection feature. The masks are added to the `eopatch.mask`
                             attribute dictionary. Default value: `'CLM_INTERSSIM'`. If None, the intersection
                             feature is not computed.
        :type mask_feature: str | None
        :param mono_threshold: Cloud probability threshold for the mono classifier. Default value: `0.4`.
        :type mono_threshold: float
        :param multi_threshold: Cloud probability threshold for the multi classifier. Default value: `0.5`.
        :type multi_threshold: float
        :param average_over: Size of the pixel neighbourhood used in the averaging post-processing step.
                             A value of `0` or `None` skips this post-processing step. Default value: `1`.
        :type average_over: int or None
        :param dilation_size: Size of the dilation post-processing step. A value of `0` or `None` skips
                              this post-processing step. Default value: `1`.
        :type dilation_size: int or None
        """

        self.proc_resolution = self._parse_resolution_arg(processing_resolution)

        self._mono_classifier = mono_classifier
        self._multi_classifier = multi_classifier

        # Set data info
        self.data_feature = self._parse_features(data_feature, default_feature_type=FeatureType.DATA)
        self.is_data_feature = self._parse_features(is_data_feature, default_feature_type=FeatureType.MASK)
        self.band_indices = (0, 1, 3, 4, 7, 8, 9, 10, 11, 12) if all_bands else tuple(range(10))

        self.sigma = 1.

        # Set max frames for single iteration
        self.max_proc_frames = max_proc_frames

        # Set feature info
        if mono_features is not None and isinstance(mono_features, tuple):
            self.mono_features = mono_features
        else:
            self.mono_features = (None, None)

        if multi_features is not None and isinstance(multi_features, tuple):
            self.multi_features = multi_features
        else:
            self.multi_features = (None, None)

        self.mask_feature = mask_feature

        # Set thresholding and morph. ops. parameters and kernels
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
    def _parse_resolution_arg(res):
        """ Parses initialization resolution argument
        """
        if res is None:
            return None

        if isinstance(res, (int, float, str)):
            res = res, res

        if isinstance(res, tuple) and len(res) == 2:
            return tuple(float(rs.strip('m')) if isinstance(rs, str) else rs for rs in res)

        raise ValueError("Wrong resolution parameter passed as an argument.")

    @property
    def mono_classifier(self):
        """ An instance of pre-trained mono-temporal cloud classifier. It is loaded only the first time it is required.
        """
        if self._mono_classifier is None:
            self._mono_classifier = joblib.load(os.path.join(self.MODELS_FOLDER, MONO_CLASSIFIER_NAME))

        return self._mono_classifier

    @property
    def multi_classifier(self):
        """ An instance of pre-trained multi-temporal cloud classifier. It is loaded only the first time it is required.
        """
        if self._multi_classifier is None:
            self._multi_classifier = joblib.load(os.path.join(self.MODELS_FOLDER, MULTI_CLASSIFIER_NAME))

        return self._multi_classifier

    @staticmethod
    def _get_max(data):
        """Timewise max for masked arrays."""
        return np.ma.max(data, axis=0).data

    @staticmethod
    def _get_min(data):
        """Timewise min for masked arrays."""
        return np.ma.min(data, axis=0).data

    @staticmethod
    def _get_mean(data):
        """Timewise mean for masked arrays."""
        return np.ma.mean(data, axis=0).data

    @staticmethod
    def _get_std(data):
        """Timewise std for masked arrays."""
        return np.ma.std(data, axis=0).data

    def _scale_factors(self, reference_shape, bbox):
        """ Compute the resampling factor for height and width of the input array

        :param reference_shape: Tuple specifying height and width in pixels of high-resolution array
        :type reference_shape: (int, int)
        :param bbox: An EOPatch bounding box
        :type bbox: sentinelhub.BBox
        :return: Rescale factor for rows and columns
        :rtype: tuple of floats
        """
        res_x, res_y = bbox_to_resolution(bbox, width=reference_shape[1], height=reference_shape[0])

        if self.proc_resolution is None:
            pres_x, pres_y = res_x, res_y
        else:
            pres_x, pres_y = self.proc_resolution

        rescale = res_y / pres_y, res_x / pres_x
        sigma = 200 / (pres_x + pres_y)

        return rescale, sigma

    def _frame_indices(self, num_of_frames, target_idx):
        """Returns frame indices within a given time window, with the target index relative to it."""

        # Get reach
        nt_min = target_idx - self.max_proc_frames//2
        nt_max = target_idx + self.max_proc_frames - self.max_proc_frames//2

        # Shift reach
        shift = max(0, -nt_min) - max(0, nt_max-num_of_frames)
        nt_min += shift
        nt_max += shift

        # Get indices within range
        nt_min = max(0, nt_min)
        nt_max = min(num_of_frames, nt_max)
        nt_rel = target_idx - nt_min

        return nt_min, nt_max, nt_rel

    def _red_ssim(self, data_x, data_y, valid_mask, mu1, mu2, sigma1_2, sigma2_2, const1=1e-6, const2=1e-5):
        """Slightly reduced (pre-computed) SSIM computation."""

        # Increase precision and mask invalid regions
        valid_mask = valid_mask.astype(np.float64)
        data_x = data_x.astype(np.float64) * valid_mask
        data_y = data_y.astype(np.float64) * valid_mask

        # Init
        mu1_2 = mu1 * mu1
        mu2_2 = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma12 = cv2.GaussianBlur(
            (data_x*data_y).astype(np.float64), (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT)
        sigma12 -= mu1_mu2

        # Formula
        tmp1 = 2. * mu1_mu2 + const1
        tmp2 = 2. * sigma12 + const2
        num = tmp1 * tmp2

        tmp1 = mu1_2 + mu2_2 + const1
        tmp2 = sigma1_2 + sigma2_2 + const2
        den = tmp1 * tmp2

        return np.divide(num, den)

    def _win_avg(self, data):
        """Spatial window average."""

        return cv2.GaussianBlur(data.astype(np.float64), (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT)

    def _win_prevar(self, data):
        """Incomplete spatial window variance."""

        return cv2.GaussianBlur((data*data).astype(np.float64), (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT)

    def _average(self, data):
        return cv2.filter2D(data.astype(np.float64), -1, self.avg_kernel, borderType=cv2.BORDER_REFLECT)

    def _dilate(self, data):
        return (cv2.dilate(data.astype(np.uint8), self.dil_kernel) > 0).astype(np.uint8)

    @staticmethod
    def _map_sequence(data, func2d):
        """
        Iterate over time and band dimensions and apply a function to each slice.
        Returns a new array with the combined results.

        :param data: input array
        :type data: array of shape (timestamps, rows, columns, channels)
        :param func2d: Mapping function that is applied on each 2d image slice. All outputs must have the same shape.
        :type func2d: function (rows, columns) -> (new_rows, new_columns)
        """

        # Map over channel dimension on 3d tensor
        def func3d(dim):
            return map_over_axis(dim, func2d, axis=2)
        # Map over time dimension on 4d tensor
        def func4d(dim):
            return map_over_axis(dim, func3d, axis=0)

        output = func4d(data)

        return output

    def _average_all(self, data):
        if self.avg_kernel is not None:
            return self._map_sequence(data, self._average)

        return data

    def _dilate_all(self, data):
        if self.dil_kernel is not None:
            return self._map_sequence(data, self._dilate)

        return data

    def _ssim_stats(self, bands, is_data, win_avg, var, nt_rel):

        ssim_max = np.empty((1, *bands.shape[1:]), dtype=np.float32)
        ssim_mean = np.empty_like(ssim_max)
        ssim_std = np.empty_like(ssim_max)

        bands_r = np.delete(bands, nt_rel, axis=0)
        win_avg_r = np.delete(win_avg, nt_rel, axis=0)
        var_r = np.delete(var, nt_rel, axis=0)

        n_frames = bands_r.shape[0]
        n_bands = bands_r.shape[-1]

        valid_mask = np.delete(is_data, nt_rel, axis=0) & is_data[nt_rel, ..., 0].reshape(
            1, *is_data.shape[1:-1], 1)

        for b_i in range(n_bands):
            local_ssim = []

            for t_j in range(n_frames):
                ssim_ij = self._red_ssim(bands[nt_rel, ..., b_i],
                                         bands_r[t_j, ..., b_i],
                                         valid_mask[t_j, ..., 0],
                                         win_avg[nt_rel, ..., b_i],
                                         win_avg_r[t_j, ..., b_i],
                                         var[nt_rel, ..., b_i],
                                         var_r[t_j, ..., b_i]
                                         )

                local_ssim.append(ssim_ij)

            local_ssim = np.ma.array(np.stack(local_ssim), mask=~valid_mask)

            ssim_max[0, ..., b_i] = self._get_max(local_ssim)
            ssim_mean[0, ..., b_i] = self._get_mean(local_ssim)
            ssim_std[0, ..., b_i] = self._get_std(local_ssim)

        return ssim_max, ssim_mean, ssim_std

    def _mono_iterations(self, bands):

        # Init
        mono_proba = np.empty((np.prod(bands.shape[:-1]), 1))
        img_size = np.prod(bands.shape[1:-1])

        n_times = bands.shape[0]

        for t_i in range(0, n_times, self.max_proc_frames):

            # Extract mono features
            nt_min = t_i
            nt_max = min(t_i+self.max_proc_frames, n_times)

            bands_t = bands[nt_min:nt_max]

            mono_features = bands_t.reshape(np.prod(bands_t.shape[:-1]), bands_t.shape[-1])

            # Run mono classifier
            mono_proba[nt_min*img_size:nt_max*img_size] = execute_with_mp_lock(
                self.mono_classifier.predict_proba, mono_features
            )[..., 1:]

        return mono_proba

    def _multi_iterations(self, bands, is_data):

        # Init
        multi_proba = np.empty((np.prod(bands.shape[:-1]), 1))
        img_size = np.prod(bands.shape[1:-1])

        n_times = bands.shape[0]

        loc_mu = None
        loc_var = None

        prev_nt_min = None
        prev_nt_max = None

        for t_i in range(n_times):

            # Extract temporal window indices
            nt_min, nt_max, nt_rel = self._frame_indices(n_times, t_i)

            bands_t = bands[nt_min:nt_max]
            is_data_t = is_data[nt_min:nt_max]

            masked_bands = np.ma.array(bands_t, mask=~is_data_t.repeat(bands_t.shape[-1], axis=-1))

            # Add window averages and variances to local data
            if loc_mu is None or prev_nt_min != nt_min or prev_nt_max != nt_max:
                loc_mu, loc_var = self._update_batches(loc_mu, loc_var, bands_t, is_data_t)

            # Interweave and concatenate
            multi_features = self._extract_multi_features(bands_t, is_data_t, loc_mu, loc_var, nt_rel, masked_bands)

            # Run multi classifier
            multi_proba[t_i*img_size:(t_i+1)*img_size] = execute_with_mp_lock(
                self.multi_classifier.predict_proba, multi_features
            )[..., 1:]

            prev_nt_min = nt_min
            prev_nt_max = nt_max

        return multi_proba

    def _update_batches(self, loc_mu, loc_var, bands_t, is_data_t):
        """Updates window variance and mean values for a batch"""

        # Add window averages and variances to local data
        if loc_mu is None:
            win_avg_bands = self._map_sequence(bands_t, self._win_avg)
            win_avg_is_data = self._map_sequence(is_data_t, self._win_avg)

            win_avg_is_data[win_avg_is_data == 0.] = 1.

            loc_mu = win_avg_bands / win_avg_is_data

            win_prevars = self._map_sequence(bands_t, self._win_prevar)
            win_prevars -= loc_mu*loc_mu

            loc_var = win_prevars

        else:

            win_avg_bands = self._map_sequence(
                bands_t[-1][None, ...], self._win_avg)
            win_avg_is_data = self._map_sequence(
                is_data_t[-1][None, ...], self._win_avg)

            win_avg_is_data[win_avg_is_data == 0.] = 1.

            loc_mu[:-1] = loc_mu[1:]
            loc_mu[-1] = (win_avg_bands / win_avg_is_data)[0]

            win_prevars = self._map_sequence(
                bands_t[-1][None, ...], self._win_prevar)
            win_prevars[0] -= loc_mu[-1]*loc_mu[-1]

            loc_var[:-1] = loc_var[1:]
            loc_var[-1] = win_prevars[0]

        return loc_mu, loc_var

    def _extract_multi_features(self, bands_t, is_data_t, loc_mu, loc_var, nt_rel, masked_bands):
        """Extracts features for a batch."""

        # Compute SSIM stats
        ssim_max, ssim_mean, ssim_std = self._ssim_stats(bands_t, is_data_t, loc_mu, loc_var, nt_rel)

        # Compute temporal stats
        temp_min = self._get_min(masked_bands)[None, ...]
        temp_mean = self._get_mean(masked_bands)[None, ...]

        # Compute difference stats
        t_all = len(bands_t)

        diff_max = (masked_bands[nt_rel][None, ...] - temp_min).data
        diff_mean = (masked_bands[nt_rel][None, ...]*(1. + 1./(t_all-1)) - t_all*temp_mean/(t_all-1)).data

        # Interweave
        ssim_interweaved = np.empty((*ssim_max.shape[:-1], 3*ssim_max.shape[-1]))
        ssim_interweaved[..., 0::3] = ssim_max
        ssim_interweaved[..., 1::3] = ssim_mean
        ssim_interweaved[..., 2::3] = ssim_std

        temp_interweaved = np.empty((*temp_min.shape[:-1], 2*temp_min.shape[-1]))
        temp_interweaved[..., 0::2] = temp_min
        temp_interweaved[..., 1::2] = temp_mean

        diff_interweaved = np.empty((*diff_max.shape[:-1], 2*diff_max.shape[-1]))
        diff_interweaved[..., 0::2] = diff_max
        diff_interweaved[..., 1::2] = diff_mean

        # Put it all together
        multi_features = np.concatenate((bands_t[nt_rel][None, ...],
                                         loc_mu[nt_rel][None, ...],
                                         ssim_interweaved,
                                         temp_interweaved,
                                         diff_interweaved
                                         ),
                                        axis=3
                                        )

        multi_features = multi_features.reshape(np.prod(multi_features.shape[:-1]), multi_features.shape[-1])

        return multi_features

    def execute(self, eopatch):
        """
        Add selected features (cloud probabilities and masks) to an EOPatch instance.

        :param eopatch: Input `EOPatch` instance
        :return: `EOPatch` with additional features
        """

        # Get data and is_data
        feature_type, feature_name = next(self.data_feature(eopatch))
        bands = eopatch[feature_type][feature_name][..., self.band_indices].astype(np.float32)

        feature_type, feature_name = next(self.is_data_feature(eopatch))
        is_data = eopatch[feature_type][feature_name].astype(bool)

        original_shape = bands.shape[1:-1]
        scale_factors, self.sigma = self._scale_factors(original_shape, eopatch.bbox)

        mono_proba_feature, mono_mask_feature = self.mono_features
        multi_proba_feature, multi_mask_feature = self.multi_features

        is_data_sm = is_data
        # Downscale if specified
        if scale_factors is not None:
            bands = resize_images(bands.astype(np.float32), scale_factors=scale_factors)
            is_data_sm = resize_images(is_data.astype(np.uint8), scale_factors=scale_factors).astype(np.bool)


        mono_proba = None
        multi_proba = None

        # Run s2cloudless if needed
        if any(feature is not None for feature in [self.mask_feature, mono_mask_feature, mono_proba_feature]):
            mono_proba = self._mono_iterations(bands)
            mono_proba = mono_proba.reshape(*bands.shape[:-1], 1)

            # Upscale if necessary
            if scale_factors is not None:
                mono_proba = resize_images(mono_proba, new_size=original_shape)

            # Average over and threshold
            mono_mask = self._average_all(mono_proba) >= self.mono_threshold

        # Run SSIM-based multi-temporal classifier if needed
        if any(feature is not None for feature in [self.mask_feature, multi_mask_feature, multi_proba_feature]):
            multi_proba = self._multi_iterations(bands, is_data_sm)
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
        if self.mask_feature is not None:
            inter_mask = mono_mask & multi_mask
            inter_mask = self._dilate_all(inter_mask)
            eopatch.mask[self.mask_feature] = (inter_mask * is_data).astype(bool)

        if mono_proba_feature is not None:
            eopatch.data[mono_proba_feature] = (mono_proba * is_data).astype(np.float32)

        if multi_proba_feature is not None:
            eopatch.data[multi_proba_feature] = (multi_proba * is_data).astype(np.float32)

        return eopatch
