"""
Module for computing blobs in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from math import sqrt
from typing import Any, Callable

import numpy as np
import skimage.feature

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import SingleFeatureSpec


class BlobTask(EOTask):
    """
    Task to compute blobs

    A blob is a region of an image in which some properties are constant or approximately constant; all the points in a
    blob can be considered in some sense to be similar to each other.

    3 methods are implemented: The Laplacian of Gaussian (LoG), the difference of Gaussian approach (DoG) and the
    determinant of the Hessian (DoH).

    The output is a `FeatureType.DATA` where the radius of each blob is stored in his center.
    ie : If blob[date, i, j, 0] = 5 then a blob of radius 5 is present at the coordinate (i, j)

    The task uses skimage.feature.blob_log or skimage.feature.blob_dog or skimage.feature.blob_doh to extract the blobs.

    The input image must be in [-1,1] range.

    :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
                    specified it will be saved with name '<feature_name>_BLOB'

                    Example: (FeatureType.DATA, 'bands') or (FeatureType.DATA, 'bands', 'blob')

    :param blob_object: Name of the blob method to use
    :type blob_object: skimage.feature.blob_*
    :param blob_parameters: List of parameters to be passed to the blob function. Below is a list of such parameters.
    :param min_sigma: The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs
    :type min_sigma: float
    :param max_sigma: The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs
    :type max_sigma: float
    :param threshold: The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored.
                        Reduce this to detect blobs with less intensity
    :type threshold: float
    :param overlap: A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold,
                    the smaller blob is eliminated
    :type overlap: float
    :param num_sigma: For ‘Log’ and ‘DoH’: The number of intermediate values of standard deviations to consider between
                        min_sigma and max_sigma
    :type num_sigma: int
    :param log_scale: For ‘Log’ and ‘DoH’: If set intermediate values of standard deviations are interpolated using a
                        logarithmic scale to the base 10. If not, linear interpolation is used
    :type log_scale: bool
    :param sigma_ratio: For ‘DoG’: The ratio between the standard deviation of Gaussian Kernels used for computing the
                        Difference of Gaussians
    :type sigma_ratio: float
    """

    def __init__(self, feature: SingleFeatureSpec, blob_object: Callable, **blob_parameters: Any):
        self.feature_parser = self.get_feature_parser(feature)

        self.blob_object = blob_object
        self.blob_parameters = blob_parameters

    def _compute_blob(self, data: np.ndarray) -> np.ndarray:
        result = np.zeros(data.shape, dtype=float)
        for time in range(data.shape[0]):
            for band in range(data.shape[-1]):
                image = data[time, :, :, band]
                res = np.asarray(self.blob_object(image, **self.blob_parameters))
                x_coord = res[:, 0].astype(int)
                y_coord = res[:, 1].astype(int)
                radius = res[:, 2] * sqrt(2)
                result[time, x_coord, y_coord, band] = radius
        return result

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute computation of blobs on input eopatch

        :param eopatch: Input eopatch
        :return: EOPatch instance with new key holding the blob image.
        """
        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            eopatch[feature_type][new_feature_name] = self._compute_blob(
                eopatch[feature_type][feature_name].astype(np.float64)
            ).astype(np.float32)

        return eopatch


class DoGBlobTask(BlobTask):
    """Task to compute blobs with Difference of Gaussian (DoG) method"""

    def __init__(
        self,
        feature: SingleFeatureSpec,
        *,
        sigma_ratio: float = 1.6,
        min_sigma: float = 1,
        max_sigma: float = 30,
        threshold: float = 0.1,
        overlap: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            feature,
            skimage.feature.blob_dog,
            sigma_ratio=sigma_ratio,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap,
            **kwargs,
        )


class DoHBlobTask(BlobTask):
    """Task to compute blobs with Determinant of the Hessian (DoH) method"""

    def __init__(
        self,
        feature: SingleFeatureSpec,
        *,
        num_sigma: float = 10,
        log_scale: bool = False,
        min_sigma: float = 1,
        max_sigma: float = 30,
        threshold: float = 0.1,
        overlap: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            feature,
            skimage.feature.blob_doh,
            num_sigma=num_sigma,
            log_scale=log_scale,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap,
            **kwargs,
        )


class LoGBlobTask(BlobTask):
    """Task to compute blobs with Laplacian of Gaussian (LoG) method"""

    def __init__(
        self,
        feature: SingleFeatureSpec,
        *,
        num_sigma: float = 10,
        log_scale: bool = False,
        min_sigma: float = 1,
        max_sigma: float = 30,
        threshold: float = 0.1,
        overlap: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            feature,
            skimage.feature.blob_log,
            num_sigma=num_sigma,
            log_scale=log_scale,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap,
            **kwargs,
        )
