"""
Module for super-pixel segmentation

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import skimage.segmentation

from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.types import SingleFeatureSpec

LOGGER = logging.getLogger(__name__)


class SuperpixelSegmentationTask(EOTask):
    """Super-pixel segmentation task

    Given a raster feature it will segment data into super-pixels. Representation of super-pixels will be returned as
    a mask timeless feature where all pixels with the same value belong to one super-pixel
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        superpixel_feature: SingleFeatureSpec,
        *,
        segmentation_object: Callable = skimage.segmentation.felzenszwalb,
        **segmentation_params: Any,
    ):
        """
        :param feature: Raster feature which will be used in segmentation
        :param superpixel_feature: A new mask timeless feature to hold super-pixel mask
        :param segmentation_object: A function (object) which performs superpixel segmentation, by default that is
            `skimage.segmentation.felzenszwalb`
        :param segmentation_params: Additional parameters which will be passed to segmentation_object function
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=FeatureTypeSet.SPATIAL_TYPES)
        self.superpixel_feature = self.parse_feature(
            superpixel_feature, allowed_feature_types={FeatureType.MASK_TIMELESS}
        )
        self.segmentation_object = segmentation_object
        self.segmentation_params = segmentation_params

    def _create_superpixel_mask(self, data: np.ndarray) -> np.ndarray:
        """Method which performs the segmentation"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return self.segmentation_object(data, **self.segmentation_params)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Main execute method"""
        data = eopatch[self.feature]

        if np.isnan(data).any():
            warnings.warn(
                "There are NaN values in given data, super-pixel segmentation might produce bad results",
                EORuntimeWarning,
            )

        if self.feature[0].is_temporal():
            data = np.moveaxis(data, 0, 2)
            data = data.reshape((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))

        superpixel_mask = np.atleast_3d(self._create_superpixel_mask(data))

        eopatch[self.superpixel_feature] = superpixel_mask

        return eopatch


class FelzenszwalbSegmentationTask(SuperpixelSegmentationTask):
    """Super-pixel segmentation which uses Felzenszwalb's method of segmentation

    Uses segmentation function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
    """

    def __init__(self, feature: SingleFeatureSpec, superpixel_feature: SingleFeatureSpec, **kwargs: Any):
        """Arguments are passed to `SuperpixelSegmentationTask` task"""
        super().__init__(feature, superpixel_feature, segmentation_object=skimage.segmentation.felzenszwalb, **kwargs)


class SlicSegmentationTask(SuperpixelSegmentationTask):
    """Super-pixel segmentation which uses SLIC method of segmentation

    Uses segmentation function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
    """

    def __init__(self, feature: SingleFeatureSpec, superpixel_feature: SingleFeatureSpec, **kwargs: Any):
        """Arguments are passed to `SuperpixelSegmentationTask` task"""
        super().__init__(
            feature, superpixel_feature, segmentation_object=skimage.segmentation.slic, start_label=0, **kwargs
        )

    def _create_superpixel_mask(self, data: np.ndarray) -> np.ndarray:
        """Method which performs the segmentation"""
        if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float64:
            data = data.astype(np.float64)
        return super()._create_superpixel_mask(data)


class MarkSegmentationBoundariesTask(EOTask):
    """Takes super-pixel segmentation mask and creates a new mask where boundaries of super-pixels are marked

    The result is a binary mask with values 0 and 1 and dtype `numpy.uint8`

    Uses `mark_boundaries` function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries
    """

    def __init__(self, feature: SingleFeatureSpec, new_feature: SingleFeatureSpec, **params: Any):
        """
        :param feature: Input feature - super-pixel mask
        :param new_feature: Output feature - a new feature where new mask with boundaries will be put
        :param params: Additional parameters which will be passed to `mark_boundaries`. Supported parameters are `mode`
            and `background_label`
        """
        self.feature = self.parse_feature(feature, allowed_feature_types={FeatureType.MASK_TIMELESS})
        self.new_feature = self.parse_feature(new_feature, allowed_feature_types={FeatureType.MASK_TIMELESS})

        self.params = params

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute method"""
        segmentation_mask = eopatch[self.feature][..., 0]

        bounds_mask = skimage.segmentation.mark_boundaries(
            np.zeros(segmentation_mask.shape[:2], dtype=np.uint8), segmentation_mask, **self.params
        )

        bounds_mask = bounds_mask[..., :1].astype(np.uint8)
        eopatch[self.new_feature[0]][self.new_feature[1]] = bounds_mask
        return eopatch
