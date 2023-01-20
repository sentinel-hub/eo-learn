"""
Module for computing Haralick textures in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
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
    }

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
        :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
            specified it will be saved with name '<feature_name>_HARALICK'.

            Example: `(FeatureType.DATA, 'bands')` or `(FeatureType.DATA, 'bands', 'haralick_values')`
        :param texture_feature: Type of Haralick textural feature to be calculated
        :param distance: Distance between pairs of pixels used for GLCM
        :param angle: Angle between pairs of pixels used for GLCM in radians, e.g. angle=np.pi/4
        :param levels: Number of bins in GLCM
        :param window_size: Size of the moving GLCM window
        :param stride: How much the GLCM window moves each time
        """
        self.feature_parser = self.get_feature_parser(feature)

        self.texture_feature = texture_feature
        if self.texture_feature not in self.AVAILABLE_TEXTURES.union(self.AVAILABLE_TEXTURES_SKIMAGE):
            raise ValueError(
                "Haralick texture feature must be one of these: "
                f"{self.AVAILABLE_TEXTURES.union(self.AVAILABLE_TEXTURES_SKIMAGE)}"
            )

        self.distance = distance
        self.angle = angle
        self.levels = levels

        self.window_size = window_size
        if self.window_size % 2 != 1:
            raise ValueError("Window size must be an odd number")

        self.stride = stride
        if self.stride >= self.window_size + 1:
            warnings.warn(
                "Haralick stride is superior to the window size; some pixel values will be ignored", EOUserWarning
            )

    def _custom_texture(self, glcm: np.ndarray) -> np.ndarray:
        # Sum of square: Variance
        if self.texture_feature == "sum_of_square_variance":
            i_raw = np.empty_like(glcm)
            i_raw[...] = np.arange(glcm.shape[0])
            i_raw = np.transpose(i_raw)
            i_minus_mean = (i_raw - glcm.mean()) ** 2
            res = np.apply_over_axes(np.sum, i_minus_mean * glcm, axes=(0, 1))[0][0]
        elif self.texture_feature == "inverse_difference_moment":
            # np.meshgrid
            j_cols = np.empty_like(glcm)
            j_cols[...] = np.arange(glcm.shape[1])
            i_minus_j = ((j_cols - np.transpose(j_cols)) ** 2) + 1
            res = np.apply_over_axes(np.sum, glcm / i_minus_j, axes=(0, 1))[0][0]
        elif self.texture_feature == "sum_average":
            # Slow
            tuple_array = np.array(list(it.product(list(range(self.levels)), list(range(self.levels)))), dtype=(int, 2))
            index = np.array([list(map(tuple, tuple_array[tuple_array.sum(axis=1) == x])) for x in range(self.levels)])
            p_x_y = np.array([glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))])
            res = np.array(p_x_y * np.array(range(len(index)))).sum()
        elif self.texture_feature == "sum_variance":
            # Slow
            tuple_array = np.array(list(it.product(list(range(self.levels)), list(range(self.levels)))), dtype=(int, 2))
            index = np.array([list(map(tuple, tuple_array[tuple_array.sum(axis=1) == x])) for x in range(self.levels)])
            p_x_y = np.array([glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))])
            sum_average = np.array(p_x_y * np.array(range(len(index)))).sum()
            res = ((np.array(range(len(index))) - sum_average) ** 2).sum()
        elif self.texture_feature == "sum_entropy":
            # Slow
            tuple_array = np.array(list(it.product(list(range(self.levels)), list(range(self.levels)))), dtype=(int, 2))
            index = np.array([list(map(tuple, tuple_array[tuple_array.sum(axis=1) == x])) for x in range(self.levels)])
            p_x_y = np.array([glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))])
            res = (p_x_y * np.log(p_x_y + np.finfo(float).eps)).sum() * -1.0
        elif self.texture_feature == "difference_variance":
            # Slow
            tuple_array = np.array(
                list(it.product(list(range(self.levels)), list(np.asarray(range(self.levels)) * -1))), dtype=(int, 2)
            )
            index = np.array(
                [list(map(tuple, tuple_array[np.abs(tuple_array.sum(axis=1)) == x])) for x in range(self.levels)]
            )
            p_x_y = np.array([glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))])
            sum_average = np.array(p_x_y * np.array(range(len(index)))).sum()
            res = ((np.array(range(len(index))) - sum_average) ** 2).sum()
        else:
            # self.texture_feature == 'difference_entropy':
            # Slow
            tuple_array = np.array(
                list(it.product(list(range(self.levels)), list(np.asarray(range(self.levels)) * -1))), dtype=(int, 2)
            )
            index = np.array(
                [list(map(tuple, tuple_array[np.abs(tuple_array.sum(axis=1)) == x])) for x in range(self.levels)]
            )
            p_x_y = np.array([glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))])
            res = (p_x_y * np.log(p_x_y + np.finfo(float).eps)).sum() * -1.0
        return res

    def _calculate_haralick(self, data: np.ndarray) -> np.ndarray:
        result = np.empty(data.shape, dtype=float)
        # For each date and each band
        for time in range(data.shape[0]):
            for band in range(data.shape[3]):
                image = data[time, :, :, band]
                image_min, image_max = np.min(image), np.max(image)
                coef = (image_max - image_min) / self.levels
                digitized_image = np.digitize(image, np.array([image_min + k * coef for k in range(self.levels - 1)]))

                # Padding the image to handle borders
                pad = self.window_size // 2
                digitized_image = np.pad(digitized_image, ((pad, pad), (pad, pad)), "edge")
                # Sliding window
                for i in range(0, image.shape[0], self.stride):
                    for j in range(0, image.shape[1], self.stride):
                        window = digitized_image[i : i + self.window_size, j : j + self.window_size]
                        glcm = skimage.feature.graycomatrix(
                            window, [self.distance], [self.angle], levels=self.levels, normed=True, symmetric=True
                        )

                        if self.texture_feature in self.AVAILABLE_TEXTURES_SKIMAGE:
                            res = skimage.feature.graycoprops(glcm, self.texture_feature)[0][0]
                        else:
                            res = self._custom_texture(glcm[:, :, 0, 0])

                        result[time, i, j, band] = res
        return result

    def execute(self, eopatch: EOPatch) -> EOPatch:
        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            eopatch[feature_type, new_feature_name] = self._calculate_haralick(eopatch[feature_type, feature_name])

        return eopatch
