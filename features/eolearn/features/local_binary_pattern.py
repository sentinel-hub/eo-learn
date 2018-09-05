""" Module for computing the Local Binary Pattern in EOPatch """

import numpy as np
from skimage.feature import local_binary_pattern
from eolearn.core import EOTask, FeatureType


class AddLocalBinaryPatternTask(EOTask):
    """
    Task to compute the Local Binary Pattern images

    LBP looks at points surrounding a central point and tests whether the surrounding points are greater than or less
    than the central point

    The task uses skimage.feature.local_binary_pattern to extract the texture features.
    """
    def __init__(self, feature_name, new_name, nb_points=24, radius=3):
        self.feature_name = feature_name
        self.new_name = new_name

        self.nb_points = nb_points
        self.radius = radius
        if nb_points < 1 or radius < 1:
            raise ValueError('Local binary pattern task parameters must be positives')

    def execute(self, eopatch):
        if self.feature_name not in eopatch.features[FeatureType.DATA].keys():
            raise ValueError('Feature {} not found in eopatch.data.'.format(self.feature_name))

        data_in = eopatch.get_feature(FeatureType.DATA, self.feature_name)
        result = np.empty(data_in.shape, dtype=np.float)

        for time in range(data_in.shape[0]):
            for band in range(data_in.shape[3]):
                image = data_in[time, :, :, band]
                result[time, :, :, band] = local_binary_pattern(image, self.nb_points, self.radius, method='uniform')

        eopatch.add_feature(FeatureType.DATA, self.new_name, result)
        return eopatch
