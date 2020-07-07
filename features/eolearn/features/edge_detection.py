"""
Module handling processing of temporal features

Credits:
Copyright (c) 2018-2019 Mark Bogataj, Filip Koprivec (Jožef Stefan Institute)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from enum import Enum

import cv2 as cv
import numpy as np
from eolearn.core import EOTask, FeatureType
from scipy import signal


class AdaptiveThresholdMethod(Enum):
    """The Enum class encapsulating possible adaptive thresholding methods from opencv
    """
    MEAN = cv.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN = cv.ADAPTIVE_THRESH_GAUSSIAN_C


class SimpleThresholdMethod(Enum):
    """The Enum class encapsulating possible simple thresholding methods from opencv
    """
    BINARY = cv.THRESH_BINARY
    BINARY_INV = cv.THRESH_BINARY_INV
    TRUNC = cv.THRESH_TRUNC
    TOZERO = cv.THRESH_TOZERO
    TOZERO_INV = cv.THRESH_TOZERO_INV


class ThresholdType(Enum):
    """The Enum class encapsulating possible thresholding methods from opencv
    """
    BINARY = cv.THRESH_BINARY
    BINARY_INV = cv.THRESH_BINARY_INV


class Thresholding(EOTask):
    """
    Task to compute thresholds of the image using basic and adaptive thresholding methods.
    Depending on the image, we can also use bluring methods that sometimes improve our results.

    With adaptive thresholding we detect edges and with basic thresholding we connect field into
    one area - segmentation.

    The task uses methods from cv2 library.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, feature, rgb_indices, base_image_index=0, correction_factor=3.5,
                 simple_th_value=127, simple_th_max_value=255, adaptive_th=AdaptiveThresholdMethod.MEAN,
                 thresh_type=ThresholdType.BINARY, simple_th=SimpleThresholdMethod.BINARY, block_size=11, base_val=2,
                 mask_th=10, max_value=255, otsu=False):
        """
        :param feature: Input feature
        :type feature: obj
        :param rgb_indices: Indices corresponding to B,G,R data in input feature
        :type rgb_indices: (int, int, int)
        :param base_image_index: Index of image to use for thresholding
        :type base_image_index: int
        :param correction_factor: Correction factor for rgb images
        :type correction_factor: float
        :param adaptive_th: adaptive thresholding method,
            ADAPTIVE_THRESH_MEAN_C=threshold value is the mean of neighbourhood area
            ADAPTIVE_THRESH_GAUSSIAN_C=threshold value is the weighted sum of neighbourhood values
            where weights are a gaussian window
        :type adaptive_th: AdaptiveThresholdMethod
        :param thresh_type: thresholding type used in adaptive thresholding
        :type thresh_type: ThresholdType
        :param simple_th: simple thesholding method
        :type simple_th: SimpleThresholdMethod
        :param block_size: it decides the size of neighbourhood area, must be odd
        :type block_size: int
        :param base_val: a constant which is subtracted from mean or weighted mean calculated
        :type base_val: int
        :param mask_th: which values do we want on our mask
        :type mask_th: int
        :param max_value:
        :type max_value: int
        :param otsu: flag that tells us if we want otsu binarization or not
        :type otsu: bool
        :param simple_th_value: threshold value for simple threshold
        :type simple_th_value: int
        :param simple_th_max_value: maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV
        thresholding types
        :type simple_th_max_value: int
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-arguments
        self.feature = self._parse_features(feature, new_names=True,
                                            default_feature_type=FeatureType.DATA)

        self.feature_type, self.feature_name, self.new_feature_name = next(
            iter(self.feature))

        self.rgb_indices = rgb_indices

        self.base_image_index = base_image_index

        self.correction_factor = correction_factor

        self.adaptive_th = adaptive_th

        self.thresh_type = thresh_type

        self.simple_th = simple_th

        self.block_size = block_size
        if not self.block_size % 2:
            raise ValueError("Block size must be an odd number")

        self.base_val = base_val

        self.mask_th = mask_th
        if self.mask_th < 0 or self.mask_th > 255:
            raise ValueError("Mask threshold must be between 0 and 255")

        self.max_value = max_value
        if self.max_value > 255 or self.max_value < 0:
            raise ValueError("maxValue must be between 0 and 255")

        self.otsu = int(otsu) * cv.THRESH_OTSU

        self.simple_th_value = simple_th_value
        if simple_th_value > 255 or simple_th_value < 0:
            raise ValueError("simple_th_value must be between 0 and 255")

        self.simple_th_max_value = simple_th_max_value
        if simple_th_max_value > 255 or simple_th_max_value < 0:
            raise ValueError("simple_th_maxValue must be between 0 and 255")

    def execute(self, eopatch):
        """Calculates edges and edge mask

        :param eopatch: Input eopatch
        :return: eopatch with calculated edges and calculated edge mask (in _mask)
        """
        # pylint: disable=invalid-name
        img_true_color = eopatch[self.feature_type][self.feature_name][self.base_image_index]
        img_true_color = cv.cvtColor(img_true_color[..., self.rgb_indices].astype('float32'), cv.COLOR_BGR2RGB)
        img_grayscale = (cv.cvtColor(img_true_color.copy(), cv.COLOR_RGB2GRAY) * 255).astype(np.uint8)

        img_true_color = np.clip(img_true_color * self.correction_factor, 0, 1)

        img2 = img_true_color

        th_adaptiv = cv.adaptiveThreshold(img_grayscale, self.max_value, self.adaptive_th.value,
                                          self.thresh_type.value, self.block_size, self.base_val)

        mask = (th_adaptiv < self.mask_th) * 255

        img2[mask != 0] = (255, 0, 0)

        _, result = cv.threshold(img2, self.simple_th.value, self.simple_th_max_value,
                                 self.thresh_type.value + self.otsu)

        eopatch.data_timeless[self.new_feature_name] = result
        n, m, _ = result.shape
        eopatch.data_timeless[self.new_feature_name + "_mask"] = mask.reshape((n, m, 1))
        return eopatch


class BlurringMethod(Enum):
    """The Enum class encapsulating possible blurring methods from opencv
    """
    NONE = 'none'
    MEDIAN_BLUR = 'medianBlurr'
    GAUSSIAN_BLUR = 'GaussianBlurr'
    BILATERAL_FILTER = 'bilateralFilter'


class Blurring:
    """Helper task for blurring methods
    """

    def __init__(self, img, sigma_y, border_type, blur_method=BlurringMethod.NONE,
                 g_ksize=(5, 5), sigma_x=0, m_ksize=5, diameter=9, sigma_color=75,
                 sigma_space=75):
        """
        :param img: a image that will be used
        :type img: 2D array or 3D array
        :param border_type: border mode used to extrapolate pixels outside of the image
        :type border_type: int
        :param blur_method: image blurring (smoothing) methods
        :type blur method: str

        => GaussianBlur params:
        :param g_ksize: Gaussian kernel size. ksize.width and ksize.height can differ but they
        both must be positive and odd. Or, they can be zero's and then they are computed from sigma
        :type g_ksize: Size
        :param sigma_x: Gaussian kernel standard deviation in X direction
        :type sigma_x: double
        :param sigma_y: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is
        set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height
        :type sigma_y: double
        :param border_type: pixel extrapolation method
        :type border_type: int

        => medianBlur params:
        :param m_ksize: aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7...
        :type m_ksize: int

        => bilateralFilter params:
        :param diameter: Diameter of each pixel neighborhood that is used during filtering. If it is
        non-positive, it is computed from sigmaSpace
        :type diameter: int
        :param sigma_color: Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace.
        :type sigma_color: double
        :param sigma_space: 	Filter sigma in the coordinate space. A larger value of the parameter
        means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor).
        When d > 0, it specifies the neighborhood size regardless of sigmaSpace.
        Otherwise, d is proportional to sigmaSpace
        :type sigma_space: double
        """

        self.img = img
        self.blur_method = blur_method
        if self.blur_method not in BlurringMethod:
            raise ValueError("Bluring method must be one of these: {}".format(
                BlurringMethod))

        self.g_ksize = g_ksize
        if self.g_ksize % 2 != 1 | self.g_ksize != 0 | self.g_ksize < 0:
            raise ValueError("gKsize must be odd and positive or 0")

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        self.m_ksize = m_ksize
        if self.m_ksize % 2 != 1 or self.m_ksize <= 1:
            raise ValueError("mKsize must be odd and greater than 1")

        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type

    def _blur(self):
        """
        Blurs the image
        :return: Blurred image
        """
        if self.blur_method is BlurringMethod.BILATERAL_FILTER:
            self.img = cv.bilateralFilter(self.img, self.diameter, self.sigma_color,
                                          self.sigma_space, self.border_type)
        elif self.blur_method is BlurringMethod.MEDIAN_BLUR:
            self.img = cv.medianBlur(self.img, self.m_ksize)
        elif self.blur_method is BlurringMethod.GAUSSIAN_BLUR:
            self.img = cv.GaussianBlur(self.img, self.g_ksize, self.sigma_x,
                                       self.sigma_y, self.border_type)
        return self.img


class OperatorEdgeDetection(EOTask):
    """Operator based edge detection EOTask

    Tasks calculates convolution between operator and image and calculated magnitude and gradient
    of corresponding detected edges.
    """

    def __init__(self, feature, operator, index=0, sub_index=None,
                 to_grayscale=False, grayscale_coef=None, convolve_args=None):
        """
        :param feature: Name of feature to perform edge detection on
        :type feature: object
        :param operator: Tuple of x and y wise derivative operator. Each operator is square matrix
        :type operator (ndarray, ndarray)
        :param sub_index Optional index of data considered in feature (default: 0)
        :param to_grayscale: Flag to indicate if the patch data needs to be converted to grayscale
        :type to_grayscale: bool
        :param grayscale_coef: Coefficients for grayscale transformation
        :type grayscale_coef: None or `np.ndarray`
        :param convolve_args: Additional arguments for convolution
        """
        self.feature = self._parse_features(feature, new_names=True,
                                            default_feature_type=FeatureType.DATA)
        self.index = index
        self.sub_index = sub_index
        self.to_grayscale = to_grayscale
        if grayscale_coef is None:
            self.grayscale_coef = np.array([0.2989, 0.5870, 0.1140])
        else:
            self.grayscale_coef = grayscale_coef
        self.operator = operator

        self.convolve_args = convolve_args or {"mode": "same"}

        self.feature_type, self.feature_name, self.new_feature_name = next(iter(self.feature))

    def execute(self, eopatch):
        """Calculates angle and magnitude of the gradient according to the provided operator

        magnitude = |[G_x * img, G_y * img]|
        gradient = atan2(G_y * img, G_x * img)

        :param eopatch: Input eopatch
        :return: Eopatch with calculated gradient magnitude and angle.
        """
        image = eopatch[self.feature_type][self.feature_name][self.index]

        if self.sub_index is not None:
            image = image[..., self.sub_index]

        # Convert to grayscale
        if self.to_grayscale:
            image = np.dot(image[..., :], self.grayscale_coef)

        squeezed = image.squeeze()

        gradients = [
            signal.convolve2d(squeezed, p_operator, **self.convolve_args) for p_operator in
            self.operator
        ]

        magnitude = np.sqrt(sum(np.square(grad) for grad in gradients))
        magnitude = magnitude / np.max(np.abs(magnitude))
        angle = np.arctan2(gradients[1], gradients[0])

        eopatch.data_timeless[self.new_feature_name] = np.concatenate(
            (np.expand_dims(magnitude, -1), np.expand_dims(angle, -1)), axis=-1)

        return eopatch


class SobelOperator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using Sobel operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class ScharrOperator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using Scharr operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[-3, 0, 3],
                   [-10, 0, 10],
                   [-3, 0, 3]])

    Gy = np.array([[-3, -10, -3],
                   [0, 0, 0],
                   [3, 10, 3]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class ScharrFourierOperator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using Scharr operator
    (using fourier coefficients)
    """
    # pylint: disable=invalid-name

    Gx = np.array([[-47, 0, 47],
                   [-162, 0, 162],
                   [-47, 0, 47]])

    Gy = np.array([[-47, -162, -47],
                   [0, 0, 0],
                   [47, 162, 47]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class Prewitt3Operator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using 3x3 Prewitt operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

    Gy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class Prewitt4Operator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using 4x4 Prewitt operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[3, 1, -1, -3],
                   [3, 1, -1, -3],
                   [3, 1, -1, -3],
                   [3, 1, -1, -3]])

    Gy = np.array([[3, 3, 3, 3],
                   [1, 1, 1, 1],
                   [-1, -1, -1, -1],
                   [-3, -3, -3, -3]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class RobertsCrossOperator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using Robert's cross operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[1, 0],
                   [0, -1]])

    Gy = np.array([[0, 1],
                   [-1, 0]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class KayyaliOperator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using Kayyali operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[6, 0, -6],
                   [0, 0, 0],
                   [-6, 0, 6]])

    Gy = np.array([[-6, 0, 6],
                   [0, 0, 0],
                   [6, 0, -6]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)


class KirschOperator(OperatorEdgeDetection):
    """
    Implements `eolearn.features.OperatorEdgeDetection` using Kirsch operator
    """
    # pylint: disable=invalid-name

    Gx = np.array([[5, 5, 5],
                   [-3, 0, -3],
                   [-3, -3, -3]])

    Gy = np.array([[5, -3, -3],
                   [5, 0, -3],
                   [5, -3, -3]])

    def __init__(self, feature, **kwargs):
        super().__init__(feature, (self.Gx, self.Gy), **kwargs)
