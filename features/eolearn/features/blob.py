"""
Module for computing blobs in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import itertools as it
from math import sqrt
from typing import Any, Callable

import numpy as np
import skimage.feature

from sentinelhub.exceptions import deprecated_class

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.exceptions import EODeprecationWarning
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

    The task uses `skimage.feature.blob_log`, `skimage.feature.blob_dog` or `skimage.feature.blob_doh` for extraction.

    The input image must be in [-1,1] range.

    :param feature: A feature that will be used and a new feature name where data will be saved, e.g.
        `(FeatureType.DATA, 'bands', 'blob')`.
    :param blob_object: Callable that calculates the blob
    :param blob_parameters: Parameters to be passed to the blob function. Consult documentation of `blob_object`
        for available parameters.
    """

    def __init__(self, feature: SingleFeatureSpec, blob_object: Callable, **blob_parameters: Any):
        self.feature_parser = self.get_feature_parser(feature, allowed_feature_types=[FeatureType.DATA])

        self.blob_object = blob_object
        self.blob_parameters = blob_parameters

    def _compute_blob(self, data: np.ndarray) -> np.ndarray:
        result = np.zeros(data.shape, dtype=np.float32)
        num_time, _, _, num_bands = data.shape
        for time_idx, band_idx in it.product(range(num_time), range(num_bands)):
            image = data[time_idx, :, :, band_idx]
            blob = self.blob_object(image, **self.blob_parameters)
            x_coord = blob[:, 0].astype(int)
            y_coord = blob[:, 1].astype(int)
            radius = blob[:, 2] * sqrt(2)
            result[time_idx, x_coord, y_coord, band_idx] = radius
        return result

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute computation of blobs on input eopatch

        :param eopatch: Input eopatch
        :return: EOPatch instance with new key holding the blob image.
        """
        for ftype, fname, new_fname in self.feature_parser.get_renamed_features(eopatch):
            eopatch[ftype, new_fname] = self._compute_blob(eopatch[ftype, fname].astype(np.float64))

        return eopatch


@deprecated_class(EODeprecationWarning, "Use `BlobTask` with `blob_object=skimage.feature.blob_dog`.")
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


@deprecated_class(EODeprecationWarning, "Use `BlobTask` with `blob_object=skimage.feature.blob_doh`.")
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


@deprecated_class(EODeprecationWarning, "Use `BlobTask` with `blob_object=skimage.feature.blob_log`.")
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
