"""
Module for computing the Histogram of gradient in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import skimage.feature

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import SingleFeatureSpec


class HOGTask(EOTask):
    """Task to compute the histogram of gradient

    Divide the image into small connected regions called cells, and for each cell compute a histogram of gradient
    directions or edge orientations for the pixels within the cell.

    The algorithm stores the result in images where each band is the value of the histogram for a specific angular
    bin. If `visualize` is `True`, it also outputs the images representing the gradients for each orientation.
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (3, 3),
        visualize: bool = True,
        hog_feature_vector: bool = False,
        block_norm: str = "L2-Hys",
        visualize_feature_name: Optional[str] = None,
    ):
        """
        :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
            specified it will be saved with name '<feature_name>_HOG'

            Example: `(FeatureType.DATA, 'bands')` or `(FeatureType.DATA, 'bands', 'hog')`
        :param orientations: Number of direction to use for the oriented gradient
        :param pixels_per_cell: Number of pixels in a cell, provided as a pair of integers.
        :param cells_per_block: Number of cells in a block, provided as a pair of integers.
        :param visualize: Produce a visualization for the HOG in an image
        :param visualize_feature_name: Name of the visualization feature to be added to the eopatch (if empty and
            visualize is True, it becomes “new_name”_VIZU)
        """
        self.feature_parser = self.get_feature_parser(feature, allowed_feature_types=[FeatureType.DATA])

        self.n_orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.block_norm = block_norm
        self.hog_feature_vector = hog_feature_vector
        self.visualize_name = visualize_feature_name

    def _compute_hog(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        results_im = np.empty(
            (
                data.shape[0],
                (int(data.shape[1] // self.pixels_per_cell[0]) - self.cells_per_block[0] + 1) * self.cells_per_block[0],
                (int(data.shape[2] // self.pixels_per_cell[1]) - self.cells_per_block[1] + 1) * self.cells_per_block[1],
                self.n_orientations,
            ),
            dtype=float,
        )
        if self.visualize:
            im_visu = np.empty(data.shape[0:3] + (1,))
        for time in range(data.shape[0]):
            is_multichannel = data.shape[-1] != 1
            image = data[time] if is_multichannel else data[time, :, :, 0]
            res, image = skimage.feature.hog(
                image,
                orientations=self.n_orientations,
                pixels_per_cell=self.pixels_per_cell,
                visualize=self.visualize,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                feature_vector=self.hog_feature_vector,
                channel_axis=-1 if is_multichannel else None,
            )
            if self.visualize:
                im_visu[time, :, :, 0] = image
            for block_row in range(res.shape[0]):
                for block_col in range(res.shape[1]):
                    for cell_row in range(res.shape[2]):
                        for cell_col in range(res.shape[3]):
                            row = block_row * self.cells_per_block[0] + cell_row
                            col = block_col * self.cells_per_block[1] + cell_col
                            for angle in range(res.shape[4]):
                                results_im[time, row, col, angle] = res[block_row, block_col, cell_row, cell_col, angle]
        return results_im, im_visu

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute computation of HoG features on input eopatch

        :param eopatch: Input eopatch
        :return: EOPatch instance with new keys holding the HoG features and HoG image for visualisation.
        """
        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            result_im, im_visu = self._compute_hog(eopatch[feature_type, feature_name])
            eopatch[feature_type, new_feature_name] = result_im
            if self.visualize:
                visualize_name = self.visualize_name or f"{new_feature_name}_VISU"
                eopatch[feature_type, visualize_name] = im_visu

        return eopatch
