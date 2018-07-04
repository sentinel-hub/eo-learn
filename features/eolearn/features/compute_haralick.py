""" Module for computing Haralick textures in EOPatch """

import numpy as np
from skimage.feature import greycomatrix, greycoprops
from eolearn.core import EOTask, FeatureType


class AddHaralickTask(EOTask):
    """
    Task to compute Haralick texture images

    The task compute the grey-level co-occurrence matrix on a sliding window over the input image and extract the
    texture properties.

    The task uses skimage.feature.greycomatrix and skimage.feature.greycoprops to extract the texture features.
    """
    def __init__(self, feature_name, new_name, texture_feature='contrast', distance=1, angle=0, levels=8, window_size=3,
                 stride=1):
        self.feature_name = feature_name
        self.available_texture = [
            'contrast',
            'dissimilarity',
            'homogeneity',
            'ASM',
            'energy',
            'correlation'
        ]
        self.new_name = new_name
        self.texture_feature = texture_feature
        if self.texture_feature not in self.available_texture:
            raise ValueError('Error : Haralick texture feature must be one these : ' + str(self.available_texture))
        self.distance = distance
        self.angle = angle
        self.levels = levels
        if self.levels <= 255:
            self.type = np.uint8
        else:
            self.type = np.uint16
        self.window_size = window_size
        if self.window_size % 2 != 1:
            raise ValueError('Error : Window size must be an odd number')
        self.stride = stride
        if self.stride >= self.window_size + 1:
            print("WARNING : Haralick stride is superior to the window size; some pixel values will be ignored")

    def execute(self, eopatch):
        if self.feature_name not in eopatch.features[FeatureType.DATA].keys():
            raise ValueError('Feature {} not found in eopatch.data.'.format(self.feature_name))

        data_in = eopatch.get_feature(FeatureType.DATA, self.feature_name)
        old_type = data_in.dtype
        # For each date :
        res = None
        for ind, image in enumerate(data_in):
            # Normalize image to [0, levels]
            image /= image.max()
            image *= (self.levels - 1)
            if image.shape[2] == 1:
                image = np.reshape(image, image.shape[0:2]).astype(self.type)
            else:
                raise ValueError("Error : Data must be 1 band only")
            # Window_size must be an odd number
            shape = image.shape
            # Padding the image to handle borders
            pad = int(self.window_size / 2)
            image = np.pad(image, ((pad, pad), (pad, pad)), 'edge')
            # Sliding window
            range_i = range(0, shape[0], self.stride)
            range_j = range(0, shape[1], self.stride)
            out = np.zeros((len(range_i), len(range_j)))
            for new_i, i in enumerate(range_i):
                for new_j, j in enumerate(range_j):
                    window = image[i:(i+self.window_size), j:(j+self.window_size)]
                    g = greycomatrix(window, [self.distance], [self.angle], levels=self.levels, normed=True,
                                     symmetric=True)
                    g = greycoprops(g, self.texture_feature)
                    out[new_i, new_j] = g[0][0]
            if ind == 0:
                res = np.array([np.reshape(out, (out.shape[0], out.shape[1], 1)).astype(old_type)])
            else:
                res = np.append(res, np.reshape(out, (out.shape[0], out.shape[1], 1)).astype(old_type), axis=0)
        eopatch.add_feature(FeatureType.DATA, self.new_name, res)
        return eopatch
