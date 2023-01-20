"""
Module for computing the Local Binary Pattern in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import numpy as np
import skimage.feature

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import SingleFeatureSpec


class LocalBinaryPatternTask(EOTask):
    """Task to compute the Local Binary Pattern images

    LBP looks at points surrounding a central point and tests whether the surrounding points are greater than or
    less than the central point

    The task uses skimage.feature.local_binary_pattern to extract the texture features.
    """

    def __init__(self, feature: SingleFeatureSpec, nb_points: int = 24, radius: int = 3):
        """
        :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
            specified it will be saved with name '<feature_name>_LBP'.

            Example: (FeatureType.DATA, 'bands') or (FeatureType.DATA, 'bands', 'lbp')
        :param nb_points: Number of point to use
        :param radius: Radius of the circle of neighbors
        """
        self.feature_parser = self.get_feature_parser(feature)

        self.nb_points = nb_points
        self.radius = radius
        if nb_points < 1 or radius < 1:
            raise ValueError("Local binary pattern task parameters must be positives")

    def _compute_lbp(self, data: np.ndarray) -> np.ndarray:
        result = np.empty(data.shape, dtype=float)
        for time in range(data.shape[0]):
            for band in range(data.shape[-1]):
                image = data[time, :, :, band]
                result[time, :, :, band] = skimage.feature.local_binary_pattern(
                    image, self.nb_points, self.radius, method="uniform"
                )
        return result

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute computation of local binary patterns on input eopatch

        :param eopatch: Input eopatch
        :return: EOPatch instance with new key holding the LBP image.
        """
        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            eopatch[feature_type, new_feature_name] = self._compute_lbp(eopatch[feature_type, feature_name])

        return eopatch
