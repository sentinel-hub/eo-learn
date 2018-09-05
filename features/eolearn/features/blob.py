""" Module for computing blobs in EOPatch """

import numpy as np
from math import sqrt
from eolearn.core import EOTask, FeatureType


class BlobTask(EOTask):
    """
    Task to compute blobs

    A blob is a region of an image in which some properties are constant or approximately constant; all the points in a
    blob can be considered in some sense to be similar to each other.

    3 methods are implemented: The Laplacian of Gaussian (LoG), the difference of Gaussian approach (DoG) and the
    determinant of the Hessian (DoH).

    The output is a FeatureType.DATA where the radius of each blob is stored in his center.
    ie : If blob[date, i, j, 0] = 5 then a blob of radius 5 is present at he coordinate (i, j)

    The task uses skimage.feature.blob_log or skimage.feature.blob_dog or skimage.feature.blob_doh to extract the blobs.
    """
    AVAILABLE_METHODS = {
        'LoG',
        'DoG',
        'DoH'
    }

    def __init__(self, feature, method, min_sigma=1, max_sigma=30, threshold=0.1, overlap=0.5, **kwargs):
        """
        :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
        specified it will be saved with name '<feature_name>_HARALICK'

        Example: (FeatureType.DATA, 'bands') or (FeatureType.DATA, 'bands', 'blob')

        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param method: Name of the method to use : Must be one of these : ‘Log’, ‘DoG’ or ‘DoH’
        :type method: str
        :param min_sigma: The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs
        :type min_sigma: float
        :param max_sigma: The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs
        :type float
        :param threshold: The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored.
        Reduce this to detect blobs with less intensity
        :type threshold: float
        :param overlap: A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold,
        the smaller blob is eliminated
        :type overlap: float
        :param kwargs: Depending of the method other arguments are available:
        For ‘Log’ and ‘DoH’:
        :param num_sigma: The number of intermediate values of standard deviations to consider between min_sigma and
        max_sigma
        :type num_sigma: int
        :param log_scale: If set intermediate values of standard deviations are interpolated using a logarithmic scale
        to the base 10. If not, linear interpolation is used
        :type log_scale: bool
        For ‘DoG’:
        :param sigma_ratio: The ratio between the standard deviation of Gaussian Kernels used for computing the
        Difference of Gaussians
        :type max_sigma: float
        """
        self.feature = self._parse_features(feature, default_feature_type=FeatureType.DATA, new_names=True,
                                            rename_function='{}_BLOB'.format)

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.blob_method = None
        self.method = method

        # Read arguments depending on the chosen method
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

    def _compute_blob(self, data):
        result = np.zeros(data.shape, dtype=np.float)
        for time in range(data.shape[0]):
            for band in range(data.shape[3]):
                image = data[time, :, :, band]
                res = np.asarray(self.blob_method(image, **self.param))
                x_coord = res[:, 0].astype(np.int)
                y_coord = res[:, 1].astype(np.int)
                radius = res[:, 2] * sqrt(2)
                for i in range(radius.shape[0]):
                    result[time, x_coord[i], y_coord[i], band] = radius[i]
        return result

    def execute(self, eopatch):

        for feature_type, feature_name, new_feature_name in self.feature:
            eopatch[feature_type][new_feature_name] = self._compute_blob(eopatch[feature_type][feature_name])

        return eopatch
