"""
Utilities for cloud masking

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja(Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np
import cv2


def map_over_axis(data, func, axis=0):
    """Map function func over each slice along axis.
    If func changes the number of dimensions, mapping axis is moved to the front.

    Returns a new array with the combined results of mapping.



    :param data: input array
    :type data: np.array
    :param func: Mapping function that is applied on each slice. Outputs must have the same shape for every slice.
    :type func: function np.array -> np.array
    :param axis: Axis over which to map the function.
    :type axis: int

    :example:

    >>> data = np.ones((5,10,10))
    >>> func = lambda x: np.zeros((7,20))
    >>> res = map_over_axis(data,func,axis=0)
    >>> res.shape
    (5, 7, 20)
    """

    # Move axis to front
    data = np.moveaxis(data, axis, 0)

    res_mapped = [func(slice) for slice in data]
    res = np.stack(res_mapped)

    # Move axis back if number of dimensions stays the same
    if data.ndim == res.ndim:
        res = np.moveaxis(res, 0, axis)

    return res


def resize_images(data, new_size=None, scale_factors=None, anti_alias=True, interpolation='linear'):
    """Resizes the image(s) acording to given size or scale factors.

    To specify the new scale use one of `new_size` or `scale_factors` parameters.

    :param data: input image array
    :type data: numpy array with shape (timestamps, height, width, channels),
                (height, width, channels), or (height, width)
    :param new_size: New size of the data (height, width)
    :type new_size: (int, int)
    :param scale_factors: Factors (fy,fx) by which to resize the image
    :type scale_factors: (float, float)
    :param anti_alias: Use anti aliasing smoothing operation when downsampling. Default is True.
    :type anti_alias: bool
    :param interpolation: Interpolation method used for resampling.
                          One of 'nearest', 'linear', 'cubic'. Default is 'linear'.
    :type interpolation: string
    """

    inter_methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }

    # Number of dimensions of input data
    ndims = data.ndim

    # Height and width axis indices for dimensionality
    height_width_axis = {2: (0, 1), 3: (0, 1), 4: (1, 2)}

    # Old height and width
    old_size = [data.shape[axis] for axis in height_width_axis[ndims]]

    if new_size is not None and scale_factors is None:
        if len(new_size) != 2 or any(not isinstance(value, int) for value in new_size):
            raise ValueError('new_size must be a pair of integers (height, width).')
        scale_factors = [new/old for old, new in zip(old_size, new_size)]
    elif scale_factors is not None and new_size is None:
        new_size = [int(size * factor) for size, factor in zip(old_size, scale_factors)]
    else:
        raise ValueError('Exactly one of the arguments new_size, scale_factors must be given.')

    if interpolation not in inter_methods:
        raise ValueError('Invalid interpolation method: %s' % interpolation)

    interpolation_method = inter_methods[interpolation]
    downscaling = scale_factors[0] < 1 or scale_factors[1] < 1

    def _resize2d(image):
        # Perform anti-alias smoothing if downscaling
        if downscaling and anti_alias:
            # Sigma computation based on skimage resize implementation
            sigmas = [((1/s) - 1)/2 for s in scale_factors]

            # Limit sigma values above 0
            sigma_y, sigma_x = [max(1e-8, sigma) for sigma in sigmas]
            image = cv2.GaussianBlur(image, (0, 0),
                                     sigmaX=sigma_x,
                                     sigmaY=sigma_y,
                                     borderType=cv2.BORDER_REFLECT)

        height, width = new_size
        resized = cv2.resize(image, (width, height), interpolation=interpolation_method)

        return resized

    _resize3d = lambda x: map_over_axis(x, _resize2d, axis=2) # Map over channel dimension
    _resize4d = lambda x: map_over_axis(x, _resize3d, axis=0) # Map over time dimension

    # Choose a resize method based on number of dimensions
    resize_methods = {2: _resize2d, 3: _resize3d, 4: _resize4d}

    resize_method = resize_methods[ndims]

    return resize_method(data)
