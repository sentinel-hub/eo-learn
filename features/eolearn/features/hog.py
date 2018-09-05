""" Module for computing the Local Binary Pattern in EOPatch """

import numpy as np
from skimage.feature import hog
from skimage import exposure
from eolearn.core import EOTask, FeatureType
import math

class AddHOGTask(EOTask):
    """
    """
    def __init__(self, feature_name, new_name, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                 visualize=True, visualize_feature_name=''):
        self.feature_name = feature_name
        self.new_name = new_name

        self.n_orientations=orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.visualize_name = visualize_feature_name
        if self.visualize_name == '':
            self.visualize_name = self.new_name + '_VISU'

    def execute(self, eopatch):
        if self.feature_name not in eopatch.features[FeatureType.DATA].keys():
            raise ValueError('Feature {} not found in eopatch.data.'.format(self.feature_name))

        data_in = eopatch.get_feature(FeatureType.DATA, self.feature_name)
        results_im = np.empty((data_in.shape[0], (int(data_in.shape[1] // self.pixels_per_cell[0]) - self.cells_per_block[0] + 1) * self.cells_per_block[0],
                               (int(data_in.shape[2] // self.pixels_per_cell[1]) - self.cells_per_block[1] + 1) * self.cells_per_block[1], self.n_orientations), dtype=np.float)
        if self.visualize:
            im_visu = np.empty(data_in.shape[0:3] + (1,))
        for time in range(data_in.shape[0]):
            multi_channel = True
            if data_in.shape[3] == 1:
                multi_channel = False,
            image = data_in[time, :, :, :]
            res, im = hog(image, orientations=self.n_orientations, pixels_per_cell=self.pixels_per_cell, visualize=self.visualize,
                          cells_per_block=self.cells_per_block, feature_vector=False, block_norm='L2-Hys', multichannel=multi_channel)
            if self.visualize:
                # im = exposure.rescale_intensity(im, in_range=(0, 1))
                im_visu[time, :, :, 0] = im
            for block_row in range(res.shape[0]):
                for block_col in range(res.shape[1]):
                    for cell_row in range(res.shape[2]):
                        for cell_col in range(res.shape[3]):
                            row = block_row * self.cells_per_block[0] + cell_row
                            col = block_col * self.cells_per_block[1] + cell_col
                            for angle in range(res.shape[4]):
                                results_im[time, row, col, angle] = res[block_row, block_col, cell_row, cell_col, angle]

        eopatch.add_feature(FeatureType.DATA, self.new_name, results_im)
        if self.visualize:
            eopatch.add_feature(FeatureType.DATA, self.visualize_name, im_visu)
        return eopatch
