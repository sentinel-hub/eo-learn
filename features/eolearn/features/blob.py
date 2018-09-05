""" Module for computing the Local Binary Pattern in EOPatch """

import numpy as np
from math import sqrt
from eolearn.core import EOTask, FeatureType

class AddBlobTask(EOTask):
    """
    """
    AVAILABLE_METHODS = {
        'LoG',
        'DoG',
        'DoH'
    }

    def __init__(self, feature_name, new_name, method, min_sigma=1, max_sigma=30, threshold=0.1, overlap=0.5, **kwargs):
        self.feature_name = feature_name
        self.new_name = new_name

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.blob_method = None
        self.method = method
        if self.method == 'LoG':
            from skimage.feature import blob_log
            self.blob_method = blob_log
            self.num_sigma = 10
            self.log_scale = False
            for key, value in kwargs.items():
                if key == 'num_sigma':
                    self.num_sigma = value
                elif key == 'log_scale':
                    self.log_scale = value
            self.param = {
                'min_sigma': self.min_sigma,
                'max_sigma': self.max_sigma,
                'num_sigma': self.num_sigma,
                'threshold': self.threshold,
                'overlap': self.overlap,
                'log_scale': self.log_scale
            }

        elif self.method == 'DoG':
            from skimage.feature import blob_dog
            self.blob_method = blob_dog
            self.sigma_ratio = 1.6
            for key, value in kwargs.items():
                if key == 'sigma_ratio':
                    self.sigma_ratio = value
            self.param = {
                'min_sigma': self.min_sigma,
                'max_sigma': self.max_sigma,
                'sigma_ratio': self.sigma_ratio,
                'threshold': self.threshold,
                'overlap': self.overlap
            }

        elif self.method == 'DoH':
            from skimage.feature import blob_doh
            self.blob_method = blob_doh
            self.num_sigma = 10
            self.log_scale = False
            for key, value in kwargs.items():
                if key == 'num_sigma':
                    self.num_sigma = value
                elif key == 'log_scale':
                    self.log_scale = value
            self.param = {
                'min_sigma': self.min_sigma,
                'max_sigma': self.max_sigma,
                'num_sigma': self.num_sigma,
                'threshold': self.threshold,
                'overlap': self.overlap,
                'log_scale': self.log_scale
            }
            self.param = {
                'min_sigma': self.min_sigma,
                'max_sigma': self.max_sigma,
                'num_sigma': self.num_sigma,
                'threshold': self.threshold,
                'overlap': self.overlap,
                'log_scale': self.log_scale
            }
        else:
            raise ValueError('Blob method must be one of these : {}'.format(self.AVAILABLE_METHODS))

    def execute(self, eopatch):
        if self.feature_name not in eopatch.features[FeatureType.DATA].keys():
            raise ValueError('Feature {} not found in eopatch.data.'.format(self.feature_name))

        data_in = eopatch.get_feature(FeatureType.DATA, self.feature_name)
        result = np.zeros(data_in.shape, dtype=np.float)
        for time in range(data_in.shape[0]):
            for band in range(data_in.shape[3]):
                image = data_in[time, :, :, band]
                res = np.asarray(self.blob_method(image, **self.param))
                x_coord = res[:, 0].astype(np.int)
                y_coord = res[:, 1].astype(np.int)
                radius = res[:, 2] * sqrt(2)
                for i in range(radius.shape[0]):
                    result[time, x_coord[i], y_coord[i], band] = radius[i]
        eopatch.add_feature(FeatureType.DATA, self.new_name, result)
        return eopatch
