"""
Module for extraction of edge mask in EOPatch
"""
import cv2
import numpy as np
from eolearn.core import EOTask, FeatureType


def normalization(feature):
    """
    Function that normalizes the input to interval [0..1]. It also returns the input as numpy float 32 bit array.
    :param feature: Array to normalize
    :return: Array with same shape, but normalized
    """
    f_min = np.nanmin(feature)
    f_max = np.nanmax(feature)
    return np.asarray((feature - f_min) / (f_max - f_min), np.float32)


class EdgeExtractionTask(EOTask):
    """
    Task computes a timeless mask of of edges from single patch based on multiple features during whole year.

    Mask is computed in several steps:
        - Individual image edge calculation
            Each image is firstly blurred with a Gaussian filter (cv2.GaussianBlur), then edges are computed using
            edge detector (cv2.Canny), finally dilation and erosion are applied for filling potential holes.
        - Individual image weight calculation
            Each edge pixel's contribution is adjusted based on that feature's values in the vicinity. The weights are
            calculated by normalizing and blurring image with a Gaussian filter (cv2.GaussianBlur).
        - Yearly feature mask calculation by joining single weighted edge images for each feature
            Weight mask is calculated by summing all weights for each pixel through the whole year. The pixels where
            do not contain an edge have a weight of 0, those who do, have a weight proportional to the weight
            calculated in the previous step. The idea of this is that we should prioritize edges which were calculated
            during the high vegetation period and ignore edges calculated during the off season eg. winter.
        - Final temporal mask calculation by joining all the yearly feature masks
            Pixels are included only, if total sum of all feature's weights for that pixel exceeds the weight threshold.
    """

    def __init__(self,
                 features,
                 output_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
                 canny_low_threshold=0.15,
                 canny_high_threshold=0.3,
                 canny_blur_size=5,
                 canny_blur_sigma=2,
                 structuring_element=None,
                 dilation_mask_size=3,
                 erosion_mask_size=2,
                 weight_blur_size=7,
                 weight_blur_sigma=4,
                 weight_threshold=0.05,
                 valid_mask=(FeatureType.MASK, 'IS_VALID')):
        """
        :param features: Features used for computation of the mask
        :param output_feature: Name of output feature
        :type output_feature: (FeatureType, str) or str
        :param canny_low_threshold: Low threshold parameter for Canny algorithm on interval [0..1]
        :type canny_low_threshold: float
        :param canny_high_threshold: High threshold parameter for Canny algorithm on interval [0..1]
        :type canny_high_threshold: float
        :param canny_blur_size: Gaussian blur mask size
        :type canny_blur_size: int
        :param canny_blur_sigma: Sigma parameter for gaussian blur
        :type canny_blur_sigma: float
        :param structuring_element: Structuring element for dilation and erosion
        :type structuring_element: 2d array with elements 0 or 1
        :param dilation_mask_size: Dilation mask size
        :type dilation_mask_size: int
        :param erosion_mask_size: Erosion mask size
        :type erosion_mask_size: int
        :param weight_blur_size: Gaussian blur mask size used for weighted sum
        :type weight_blur_size: int
        :param weight_blur_sigma: Gaussian blur mask sigma used for weighted sum
        :type weight_blur_sigma: float
        :param weight_threshold: Threshold for joining all the images together. If the threshold is 0 the entire step
            weight calculation is not done and none of edges are excluded.
        :type weight_threshold: float
        :param valid_mask: A feature used as a mask for valid regions. If left as None the whole patch is used
        :type valid_mask: (FeatureType, str), str or None
        """

        self.features = self._parse_features(features, default_feature_type=FeatureType.DATA)
        self.output_feature = next(
            self._parse_features(output_feature, default_feature_type=FeatureType.MASK_TIMELESS)())
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        if canny_blur_size and canny_blur_sigma:
            self.canny_blur = lambda x: cv2.GaussianBlur(x, (canny_blur_size, canny_blur_size), canny_blur_sigma)
        else:
            self.canny_blur = lambda x: x
        if not structuring_element:
            self.structuring_element = [[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]]
        else:
            self.structuring_element = structuring_element
        self.dilation_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_mask_size, dilation_mask_size))
        self.erosion_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_mask_size, erosion_mask_size))
        if canny_blur_size and canny_blur_sigma:
            self.weight_blur = lambda x: cv2.GaussianBlur(x, (weight_blur_size, weight_blur_size), weight_blur_sigma)
        else:
            self.canny_blur = lambda x: x
        self.weight_threshold = weight_threshold
        self.valid_mask = next(self._parse_features(valid_mask, default_feature_type=FeatureType.MASK)())

    def execute(self, eopatch):
        """
        :param eopatch: Source EOPatch with all necessary features
        :return: EOPatch with added mask of edges
        """
        timestamps, width, height, _ = eopatch[self.valid_mask].shape
        no_feat = None
        all_edges = np.zeros((timestamps, width, height))

        # Iterating over all features
        for i, feature in enumerate(self.features()):
            no_feat = i + 1
            images = eopatch[feature].squeeze()
            images = images * eopatch[self.valid_mask].squeeze()
            images = normalization(images)
            feature_edge = np.zeros((timestamps, width, height))

            # For each feature, calculation of yearly weighted mask
            for individual_time in range(timestamps):
                one_image = images[individual_time]
                smoothed_image = self.canny_blur(one_image)

                # cv2.Canny only works on 8 bit color coding
                smoothed_image *= 255
                one_edge = cv2.Canny(smoothed_image.astype(np.uint8),
                                     int(self.canny_low_threshold * 255),
                                     int(self.canny_high_threshold * 255))
                feature_edge[individual_time] = one_edge > 0

            if self.weight_threshold:
                adjust_weights = np.asarray([self.weight_blur(x) for x in images], np.float32)
                all_edges += feature_edge * adjust_weights
            else:
                all_edges += feature_edge

        all_edges = np.sum(all_edges, 0)
        all_edges = all_edges / (timestamps * no_feat)
        all_edges = all_edges > self.weight_threshold

        all_edges = np.logical_not(all_edges)
        eopatch[self.output_feature] = all_edges[..., np.newaxis]
        return eopatch
