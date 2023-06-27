"""
Module for computing Haralick textures in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import itertools as it
import warnings

import numpy as np
import skimage.feature

from eolearn.core import EOPatch, EOTask
from eolearn.core.exceptions import EOUserWarning
from eolearn.core.types import SingleFeatureSpec


class HaralickTask(EOTask):
    """Task to compute Haralick texture images

    The task compute the grey-level co-occurrence matrix (GLCM) on a sliding window over the input image and extract the
    texture properties.

    The task uses `skimage.feature.greycomatrix` and `skimage.feature.greycoprops` to extract the texture features.
    """

    AVAILABLE_TEXTURES_SKIMAGE = {"contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"}
    AVAILABLE_TEXTURES = {
        "sum_of_square_variance",
        "inverse_difference_moment",
        "sum_average",
        "sum_variance",
        "sum_entropy",
        "difference_variance",
        "difference_entropy",
    }.union(AVAILABLE_TEXTURES_SKIMAGE)

    def __init__(
        self,
        feature: SingleFeatureSpec,
        texture_feature: str = "contrast",
        distance: int = 1,
        angle: float = 0,
        levels: int = 8,
        window_size: int = 3,
        stride: int = 1,
    ):
        """
        :param feature: A feature that will be used and a new feature name where data will be saved, e.g.
          `(FeatureType.DATA, 'bands', 'haralick_values')`
        :param texture_feature: Type of Haralick textural feature to be calculated
        :param distance: Distance between pairs of pixels used for GLCM
        :param angle: Angle between pairs of pixels used for GLCM in radians, e.g. angle=np.pi/4
        :param levels: Number of bins in GLCM
        :param window_size: Size of the moving GLCM window
        :param stride: How much the GLCM window moves each time
        """
        self.feature_parser = self.get_feature_parser(feature)

        self.texture_feature = texture_feature
        if self.texture_feature not in self.AVAILABLE_TEXTURES:
            raise ValueError(f"Haralick texture feature must be one of these: {self.AVAILABLE_TEXTURES}")

        self.distance = distance
        self.angle = angle
        self.levels = levels

        self.window_size = window_size
        if self.window_size % 2 != 1:
            raise ValueError("Window size must be an odd number")

        self.stride = stride
        if self.stride > self.window_size:
            warnings.warn(
                "Haralick stride is larger than window size; some pixel values will be ignored", EOUserWarning
            )

    def _custom_texture(self, glcm: np.ndarray) -> np.ndarray:  # pylint: disable=too-many-return-statements
        if self.texture_feature == "sum_of_square_variance":
            i_raw = np.empty_like(glcm)
            i_raw[...] = np.arange(glcm.shape[0])
            i_raw = np.transpose(i_raw)
            i_minus_mean = (i_raw - glcm.mean()) ** 2
            return np.apply_over_axes(np.sum, i_minus_mean * glcm, axes=(0, 1))[0][0]
        if self.texture_feature == "inverse_difference_moment":
            j_cols = np.empty_like(glcm)
            j_cols[...] = np.arange(glcm.shape[1])
            i_minus_j = ((j_cols - np.transpose(j_cols)) ** 2) + 1
            return np.apply_over_axes(np.sum, glcm / i_minus_j, axes=(0, 1))[0][0]
        if self.texture_feature == "sum_average":
            p_x_y = self._get_pxy(glcm)
            return np.array(p_x_y * np.arange(len(p_x_y))).sum()
        if self.texture_feature == "sum_variance":
            p_x_y = self._get_pxy(glcm)
            sum_average = np.array(p_x_y * np.arange(len(p_x_y))).sum()
            return ((np.arange(len(p_x_y)) - sum_average) ** 2).sum()
        if self.texture_feature == "sum_entropy":
            p_x_y = self._get_pxy(glcm)
            return (p_x_y * np.log(p_x_y + np.finfo(float).eps)).sum() * -1.0
        if self.texture_feature == "difference_variance":
            p_x_y = self._get_pxy_for_diff(glcm)
            sum_average = np.array(p_x_y * np.arange(len(p_x_y))).sum()
            return ((np.arange(len(p_x_y)) - sum_average) ** 2).sum()

        # self.texture_feature == 'difference_entropy':
        p_x_y = self._get_pxy_for_diff(glcm)
        return (p_x_y * np.log(p_x_y + np.finfo(float).eps)).sum() * -1.0

    def _get_pxy(self, glcm: np.ndarray) -> np.ndarray:
        tuple_array = np.array(list(it.product(range(self.levels), range(self.levels))))
        index = [tuple_array[tuple_array.sum(axis=1) == x] for x in range(self.levels)]
        return np.array([glcm[tuple(np.moveaxis(idx, -1, 0))].sum() for idx in index])

    def _get_pxy_for_diff(self, glcm: np.ndarray) -> np.ndarray:
        tuple_array = np.array(list(it.product(range(self.levels), np.asarray(range(self.levels)) * -1)))
        index = [tuple_array[np.abs(tuple_array.sum(axis=1)) == x] for x in range(self.levels)]
        return np.array([glcm[tuple(np.moveaxis(idx, -1, 0))].sum() for idx in index])

    def _calculate_haralick(self, data: np.ndarray) -> np.ndarray:
        result = np.empty(data.shape, dtype=float)
        num_times, _, _, num_bands = data.shape
        # For each date and each band
        for time, band in it.product(range(num_times), range(num_bands)):
            image = data[time, :, :, band]
            image_min, image_max = np.min(image), np.max(image)
            coef = (image_max - image_min) / self.levels
            digitized_image = np.digitize(image, np.array([image_min + k * coef for k in range(self.levels - 1)]))

            # Padding the image to handle borders
            pad = self.window_size // 2
            digitized_image = np.pad(digitized_image, ((pad, pad), (pad, pad)), "edge")
            # Sliding window
            for i, j in it.product(range(0, image.shape[0], self.stride), range(0, image.shape[1], self.stride)):
                window = digitized_image[i : i + self.window_size, j : j + self.window_size]
                glcm = skimage.feature.graycomatrix(
                    window, [self.distance], [self.angle], levels=self.levels, normed=True, symmetric=True
                )

                if self.texture_feature in self.AVAILABLE_TEXTURES_SKIMAGE:
                    result[time, i, j, band] = skimage.feature.graycoprops(glcm, self.texture_feature)[0][0]
                else:
                    result[time, i, j, band] = self._custom_texture(glcm[:, :, 0, 0])

        return result

    def execute(self, eopatch: EOPatch) -> EOPatch:
        for ftype, fname, new_fname in self.feature_parser.get_renamed_features(eopatch):
            eopatch[ftype, new_fname] = self._calculate_haralick(eopatch[ftype, fname])

        return eopatch
