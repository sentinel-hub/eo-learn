"""
Utilities for cloud masking

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from typing import Callable

import cv2
import numpy as np


def map_over_axis(data: np.ndarray, func: Callable[[np.ndarray], np.ndarray], axis: int = 0) -> np.ndarray:
    """Map function func over each slice along axis.
    If func changes the number of dimensions, mapping axis is moved to the front.

    Returns a new array with the combined results of mapping.

    :param data: input array
    :param func: Mapping function that is applied on each slice. Outputs must have the same shape for every slice.
    :param axis: Axis over which to map the function.

    :example:

    >>> data = np.ones((5,10,10))
    >>> func = lambda x: np.zeros((7,20))
    >>> res = map_over_axis(data,func,axis=0)
    >>> res.shape
    (5, 7, 20)
    """
    # Move axis to front
    data = np.moveaxis(data, axis, 0)
    mapped_data = np.stack([func(data_slice) for data_slice in data])

    # Move axis back if number of dimensions stays the same
    if data.ndim == mapped_data.ndim:
        mapped_data = np.moveaxis(mapped_data, 0, axis)

    return mapped_data


def resize_images(
    data,
    new_size=None,
    scale_factors=None,
    anti_alias=True,
    interpolation="linear",
):
    """DEPRECATED, please use `eolearn.features.utils.spatially_resize_image` instead.

    Resizes the image(s) according to given size or scale factors.

    To specify the new scale use one of `new_size` or `scale_factors` parameters.

    :param data: input image array
    :param new_size: New size of the data (height, width)
    :param scale_factors: Factors (fy,fx) by which to resize the image
    :param anti_alias: Use anti aliasing smoothing operation when downsampling. Default is True.
    :param interpolation: Interpolation method used for resampling.
                          One of 'nearest', 'linear', 'cubic'. Default is 'linear'.
    """

    inter_methods = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

    # Number of dimensions of input data
    ndims = data.ndim

    height_width_axis = {2: (0, 1), 3: (0, 1), 4: (1, 2)}

    # Old height and width
    old_size = tuple(data.shape[axis] for axis in height_width_axis[ndims])

    if new_size is not None and scale_factors is None:
        scale_factors = tuple(new / old for old, new in zip(old_size, new_size))
    elif scale_factors is not None and new_size is None:
        new_size = tuple(int(size * factor) for size, factor in zip(old_size, scale_factors))
    else:
        raise ValueError("Exactly one of the arguments new_size, scale_factors must be given.")

    if interpolation not in inter_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation}")

    interpolation_method = inter_methods[interpolation]
    downscaling = scale_factors[0] < 1 or scale_factors[1] < 1

    def _resize2d(image: np.ndarray) -> np.ndarray:
        if downscaling and anti_alias:
            # Sigma computation based on skimage resize implementation
            sigmas = tuple(((1 / s) - 1) / 2 for s in scale_factors)

            # Limit sigma values above 0
            sigma_y, sigma_x = tuple(max(1e-8, sigma) for sigma in sigmas)
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)

        height, width = new_size
        resized = cv2.resize(image, (width, height), interpolation=interpolation_method)

        return resized

    _resize3d = lambda x: map_over_axis(x, _resize2d, axis=2)  # pylint: disable=unnecessary-lambda-assignment # noqa
    _resize4d = lambda x: map_over_axis(x, _resize3d, axis=0)  # pylint: disable=unnecessary-lambda-assignment # noqa

    # Choose a resize method based on number of dimensions
    resize_methods = {2: _resize2d, 3: _resize3d, 4: _resize4d}

    resize_method = resize_methods[ndims]

    return resize_method(data)
