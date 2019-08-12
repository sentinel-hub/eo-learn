"""
Module for computing the Histogram of gradient in EOPatch

Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2019 Matej Aleksandrov, Devis Peresutti (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import skimage.feature
import numpy as np

from eolearn.core import EOTask, FeatureType


class HOGTask(EOTask):
    """ Task to compute the histogram of gradient

        Divide the image into small connected regions called cells, and for each cell compute a histogram of gradient
        directions or edge orientations for the pixels within the cell.

        The algorithm stores the result in images where each band is the value of the histogram for a specific angular
        bin. If the visualize is True, it also output the images representing the gradients for each orientation.

        :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
                        specified it will be saved with name '<feature_name>_HOG'

                        Example: (FeatureType.DATA, 'bands') or (FeatureType.DATA, 'bands', 'hog')
        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param orientations: Number of direction to use for the oriented gradient
        :type orientations: int
        :param pixels_per_cell: Number of pixels in a cell
        :type pixels_per_cell: (int, int)
        :param cells_per_block: Number of cells in a block
        :type cells_per_block: (int, int)
        :param visualize: Produce a visualization for the HOG in an image
        :type visualize: bool
        :param visualize_feature_name: Name of the visualization feature to be added to the eopatch (if empty and
        visualize is True, the become “new_name”_VIZU
        :type visualize_feature_name: str
    """
    def __init__(self, feature, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                 visualize=True, hog_feature_vector=False, block_norm='L2-Hys', visualize_feature_name=''):
        self.feature = self._parse_features(feature, default_feature_type=FeatureType.DATA, new_names=True,
                                            rename_function='{}_HOG'.format)

        self.n_orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.block_norm = block_norm
        self.hog_feature_vector = hog_feature_vector
        self.visualize_name = visualize_feature_name
        if self.visualize_name == '':
            for _, _, new_feature_name in self.feature:
                self.visualize_name = new_feature_name + '_VISU'

    def _compute_hog(self, data):
        results_im = np.empty((data.shape[0],
                               (int(data.shape[1] // self.pixels_per_cell[0]) - self.cells_per_block[0] + 1) *
                               self.cells_per_block[0],
                               (int(data.shape[2] // self.pixels_per_cell[1]) - self.cells_per_block[1] + 1) *
                               self.cells_per_block[1], self.n_orientations), dtype=np.float)
        if self.visualize:
            im_visu = np.empty(data.shape[0:3] + (1,))
        for time in range(data.shape[0]):
            multi_channel = data.shape[-1] != 1
            image = data[time] if multi_channel else data[time, :, :, 0]
            res, image = skimage.feature.hog(image, orientations=self.n_orientations,
                                             pixels_per_cell=self.pixels_per_cell, visualize=self.visualize,
                                             cells_per_block=self.cells_per_block, block_norm=self.block_norm,
                                             feature_vector=self.hog_feature_vector, multichannel=multi_channel)
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

    def execute(self, eopatch):
        """ Execute computation of HoG features on input eopatch

            :param eopatch: Input eopatch
            :type eopatch: eolearn.core.EOPatch
            :return: EOPatch instance with new keys holding the HoG features and HoG image for visualisation.
            :rtype: eolearn.core.EOPatch
        """
        for feature_type, feature_name, new_feature_name in self.feature:
            result = self._compute_hog(eopatch[feature_type][feature_name])
            eopatch[feature_type][new_feature_name] = result[0]
            if self.visualize:
                eopatch[feature_type][self.visualize_name] = result[1]

        return eopatch
