"""
Module for computing the Histogram of gradient in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import itertools as it

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
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (3, 3),
        visualize: bool = True,
        hog_feature_vector: bool = False,
        block_norm: str = "L2-Hys",
        visualize_feature_name: str | None = None,
    ):
        """
        :param feature: A feature that will be used and a new feature name where data will be saved, e.g.
            `(FeatureType.DATA, 'bands', 'hog')`.
        :param orientations: Number of direction to use for the oriented gradient
        :param pixels_per_cell: Number of pixels in a cell, provided as a pair of integers.
        :param cells_per_block: Number of cells in a block, provided as a pair of integers.
        :param visualize: Produce a visualization for the HOG in an image
        :param visualize_feature_name: Name of the visualization feature to be added to the eopatch (if empty and
            visualize is True, it becomes “new_name”_VISU)
        """
        self.feature_parser = self.get_feature_parser(feature, allowed_feature_types=[FeatureType.DATA])

        self.n_orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.block_norm = block_norm
        self.hog_feature_vector = hog_feature_vector
        self.visualize_name = visualize_feature_name

    def _compute_hog(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pylint: disable=too-many-locals
        num_times, height, width, num_bands = data.shape
        is_multichannel = num_bands != 1
        hog_result = np.empty(
            (
                num_times,
                ((height // self.pixels_per_cell[0]) - self.cells_per_block[0] + 1) * self.cells_per_block[0],
                ((width // self.pixels_per_cell[1]) - self.cells_per_block[1] + 1) * self.cells_per_block[1],
                self.n_orientations,
            ),
            dtype=np.float32,
        )
        if self.visualize:
            hog_visualization = np.empty((num_times, height, width, 1))
        for time in range(num_times):
            output, image = skimage.feature.hog(
                data[time] if is_multichannel else data[time, :, :, 0],
                orientations=self.n_orientations,
                pixels_per_cell=self.pixels_per_cell,
                visualize=self.visualize,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                feature_vector=self.hog_feature_vector,
                channel_axis=-1 if is_multichannel else None,
            )

            block_rows, block_cols, cell_rows, cell_cols, angles = output.shape
            for block_row, block_col in it.product(range(block_rows), range(block_cols)):
                for cell_row, cell_col in it.product(range(cell_rows), range(cell_cols)):
                    row = block_row * self.cells_per_block[0] + cell_row
                    col = block_col * self.cells_per_block[1] + cell_col
                    for angle in range(angles):
                        hog_result[time, row, col, angle] = output[block_row, block_col, cell_row, cell_col, angle]

            if self.visualize:
                hog_visualization[time, :, :, 0] = image

        return hog_result, hog_visualization

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute computation of HoG features on input eopatch

        :param eopatch: Input eopatch
        :return: EOPatch instance with new keys holding the HoG features and HoG image for visualization.
        """
        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            hog_result, hog_visualization = self._compute_hog(eopatch[feature_type, feature_name])
            eopatch[feature_type, new_feature_name] = hog_result
            if self.visualize:
                visualize_name = self.visualize_name or f"{new_feature_name}_VISU"
                eopatch[feature_type, visualize_name] = hog_visualization

        return eopatch
