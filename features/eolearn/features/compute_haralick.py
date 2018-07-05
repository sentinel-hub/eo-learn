""" Module for computing Haralick textures in EOPatch """

import numpy as np

from warnings import warn
from skimage.feature import greycomatrix, greycoprops

from eolearn.core import EOTask, FeatureType


class AddHaralickTask(EOTask):
    """
    Task to compute Haralick texture images

    The task compute the grey-level co-occurrence matrix on a sliding window over the input image and extract the
    texture properties.

    The task uses skimage.feature.greycomatrix and skimage.feature.greycoprops to extract the texture features.
    """
    AVAILABLE_TEXTURES = {
        'contrast',
        'dissimilarity',
        'homogeneity',
        'ASM',
        'energy',
        'correlation'
    }

    def __init__(self, feature_name, new_name, texture_feature='contrast', distance=1, angle=0, levels=8, window_size=3,
                 stride=1):
        self.feature_name = feature_name
        self.new_name = new_name

        self.texture_feature = texture_feature
        if self.texture_feature not in self.AVAILABLE_TEXTURES:
            raise ValueError('Haralick texture feature must be one of these : {}'.format(self.AVAILABLE_TEXTURES))

        self.distance = distance
        self.angle = angle
        self.levels = levels

        self.window_size = window_size
        if self.window_size % 2 != 1:
            raise ValueError('Window size must be an odd number')

        self.stride = stride
        if self.stride >= self.window_size + 1:
            warn('Haralick stride is superior to the window size; some pixel values will be ignored')

    def execute(self, eopatch):
        if self.feature_name not in eopatch.features[FeatureType.DATA].keys():
            raise ValueError('Feature {} not found in eopatch.data.'.format(self.feature_name))

        data_in = eopatch.get_feature(FeatureType.DATA, self.feature_name)

        result = np.empty(data_in.shape, dtype=np.float)
        # For each date and each band
        for time in range(data_in.shape[0]):
            for band in range(data_in.shape[3]):
                image = data_in[time, :, :, band]
                image_min, image_max = np.min(image), np.max(image)
                coef = (image_max - image_min) / self.levels
                digitized_image = np.digitize(image, np.array([image_min + k * coef for k in range(self.levels - 1)]))

                # Padding the image to handle borders
                pad = self.window_size // 2
                digitized_image = np.pad(digitized_image, ((pad, pad), (pad, pad)), 'edge')
                # Sliding window
                for i in range(0, image.shape[0], self.stride):
                    for j in range(0, image.shape[1], self.stride):
                        window = digitized_image[i: i + self.window_size, j: j + self.window_size]
                        glcm = greycomatrix(window, [self.distance], [self.angle], levels=self.levels, normed=True,
                                            symmetric=True)
                        result[time, i, j, band] = greycoprops(glcm, self.texture_feature)[0][0]

        eopatch.add_feature(FeatureType.DATA, self.new_name, result)
        return eopatch
