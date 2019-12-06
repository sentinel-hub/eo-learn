"""
Module for super-pixel segmentation

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
import warnings

import skimage.segmentation
import numpy as np

from eolearn.core import EOTask, FeatureType, FeatureTypeSet

LOGGER = logging.getLogger(__name__)


class SuperpixelSegmentation(EOTask):
    """ Super-pixel segmentation task

    Given a raster feature it will segment data into super-pixels. Representation of super-pixels will be returned as
    a mask timeless feature where all pixels with the same value belong to one super-pixel
    """
    def __init__(self, feature, superpixel_feature, *, segmentation_object=skimage.segmentation.felzenszwalb,
                 **segmentation_params):
        """
        :param feature: Raster feature which will be used in segmentation
        :param superpixel_feature: A new mask timeless feature to hold super-pixel mask
        :param segmentation_object: A function (object) which performs superpixel segmentation, by default that is
            `skimage.segmentation.felzenszwalb`
        :param segmentation_params: Additional parameters which will be passed to segmentation_object function
        """
        self.feature_checker = self._parse_features(feature, allowed_feature_types=FeatureTypeSet.SPATIAL_TYPES)
        self.superpixel_feature = next(self._parse_features(superpixel_feature,
                                                            allowed_feature_types={FeatureType.MASK_TIMELESS})())
        self.segmentation_object = segmentation_object
        self.segmentation_params = segmentation_params

    def _create_superpixel_mask(self, data):
        """ Method which performs the segmentation
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module=skimage.segmentation.__name__)
            return self.segmentation_object(data, **self.segmentation_params)

    def execute(self, eopatch):
        """ Main execute method
        """
        feature_type, feature_name = next(self.feature_checker(eopatch))

        data = eopatch[feature_type][feature_name]

        if np.isnan(data).any():
            warnings.warn('There are NaN values in given data, super-pixel segmentation might produce bad results',
                          RuntimeWarning)

        if feature_type.is_time_dependent():
            data = np.moveaxis(data, 0, 2)
            data = data.reshape((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))

        superpixel_mask = np.atleast_3d(self._create_superpixel_mask(data))

        new_feature_type, new_feature_name = self.superpixel_feature
        eopatch[new_feature_type][new_feature_name] = superpixel_mask

        return eopatch


class FelzenszwalbSegmentation(SuperpixelSegmentation):
    """ Super-pixel segmentation which uses Felzenszwalb's method of segmentation

    Uses segmentation function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
    """
    def __init__(self, feature, superpixel_feature, **kwargs):
        """ Arguments are passed to `SuperpixelSegmentation` task
        """
        super().__init__(feature, superpixel_feature, segmentation_object=skimage.segmentation.felzenszwalb, **kwargs)


class SlicSegmentation(SuperpixelSegmentation):
    """ Super-pixel segmentation which uses SLIC method of segmentation

    Uses segmentation function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
    """
    def __init__(self, feature, superpixel_feature, **kwargs):
        """ Arguments are passed to `SuperpixelSegmentation` task
        """
        super().__init__(feature, superpixel_feature, segmentation_object=skimage.segmentation.slic, **kwargs)

    def _create_superpixel_mask(self, data):
        """ Method which performs the segmentation
        """
        if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float64:
            data = data.astype(np.float64)
        return super()._create_superpixel_mask(data)


class MarkSegmentationBoundaries(EOTask):
    """ Takes super-pixel segmentation mask and creates a new mask where boundaries of super-pixels are marked

    The result is a binary mask with values 0 and 1 and dtype `numpy.uint8`

    Uses `mark_boundaries` function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries
    """
    def __init__(self, feature, new_feature, **params):
        """
        :param feature: Input feature - super-pixel mask
        :type feature: (FeatureType, str)
        :param new_feature: Output feature - a new feature where new mask with boundaries will be put
        :type new_feature: (FeatureType, str)
        :param params: Additional parameters which will be passed to `mark_boundaries`. Supported parameters are `mode`
            and `background_label`
        """
        self.feature_checker = self._parse_features(feature, allowed_feature_types={FeatureType.MASK_TIMELESS})
        self.new_feature = next(self._parse_features(new_feature, allowed_feature_types={FeatureType.MASK_TIMELESS})())

        self.params = params

    def execute(self, eopatch):
        """ Execute method
        """
        feature_type, feature_name = next(self.feature_checker(eopatch))
        segmentation_mask = eopatch[feature_type][feature_name][..., 0]

        bounds_mask = skimage.segmentation.mark_boundaries(np.zeros(segmentation_mask.shape[:2], dtype=np.uint8),
                                                           segmentation_mask, **self.params)

        bounds_mask = bounds_mask[..., :1].astype(np.uint8)
        eopatch[self.new_feature[0]][self.new_feature[1]] = bounds_mask
        return eopatch
