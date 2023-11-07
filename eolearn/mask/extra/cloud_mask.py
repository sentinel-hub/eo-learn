"""
Module for cloud masking

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import numpy as np
from s2cloudless import S2PixelCloudDetector

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import Feature

LOGGER = logging.getLogger(__name__)


class CloudMaskTask(EOTask):
    """Cloud masking with the s2cloudless model. Outputs a cloud mask and optionally the cloud probabilities."""

    def __init__(
        self,
        data_feature: Feature,
        valid_data_feature: Feature,
        output_mask_feature: Feature,
        output_proba_feature: Feature | None = None,
        all_bands: bool = True,
        threshold: float = 0.4,
        average_over: int | None = 4,
        dilation_size: int | None = 2,
    ):
        """
        :param data_feature: A data feature which stores raw Sentinel-2 reflectance bands.
        :param valid_data_feature: A mask feature which indicates whether data is valid.
        :param output_mask_feature: The output feature containing cloud masks.
        :param output_proba_feature: The output feature containing cloud probabilities. By default this is not saved.
        :param all_bands: Flag which indicates whether images will consist of all 13 Sentinel-2 L1C bands or only
            the required 10.
        :param threshold: Cloud probability threshold for the classifier.
        :param average_over: Size of the pixel neighbourhood used in the averaging post-processing step. Set to `None`
            to skip this post-processing step.
        :param dilation_size: Size of the dilation post-processing step. Set to `None` to skip this post-processing
            step.
        """
        self.data_feature = self.parse_feature(data_feature)
        self.data_indices = (0, 1, 3, 4, 7, 8, 9, 10, 11, 12) if all_bands else tuple(range(10))
        self.valid_data_feature = self.parse_feature(valid_data_feature)

        self.output_mask_feature = self.parse_feature(output_mask_feature)
        self.output_proba_feature = None
        if output_proba_feature is not None:
            self.output_proba_feature = self.parse_feature(output_proba_feature)

        self.threshold = threshold

        self.classifier = S2PixelCloudDetector(
            threshold=threshold, average_over=average_over, dilation_size=dilation_size, all_bands=all_bands
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Add selected features (cloud probabilities and masks) to an EOPatch instance.

        :param eopatch: Input `EOPatch` instance
        :return: `EOPatch` with additional features
        """
        data = eopatch[self.data_feature].astype(np.float32)
        valid_data = eopatch[self.valid_data_feature].astype(bool)

        patch_bbox = eopatch.bbox
        if patch_bbox is None:
            raise ValueError("Cannot run cloud masking on an EOPatch without a BBox.")

        cloud_proba = self.classifier.get_cloud_probability_maps(data)
        cloud_mask = self.classifier.get_mask_from_prob(cloud_proba, threshold=self.threshold)

        eopatch[self.output_mask_feature] = (cloud_mask[..., np.newaxis] * valid_data).astype(bool)
        if self.output_proba_feature is not None:
            eopatch[self.output_proba_feature] = (cloud_proba[..., np.newaxis] * valid_data).astype(np.float32)

        return eopatch
