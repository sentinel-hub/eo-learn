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
import warnings
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from eolearn.core.exceptions import EODeprecationWarning


class ResizeMethod(Enum):
    """Methods available for spatial resizing of data."""

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"

    def get_cv2_method(self, dtype: Union[np.dtype, type]) -> int:
        """Obtain the constant specifying the interpolation method for the CV2 library."""
        number_dtype = np.dtype(dtype)
        if np.issubdtype(number_dtype, np.floating):
            choices = {
                ResizeMethod.NEAREST: cv2.INTER_NEAREST,
                ResizeMethod.LINEAR: cv2.INTER_LINEAR,
                ResizeMethod.CUBIC: cv2.INTER_CUBIC,
            }
            return choices[self]
        if np.issubdtype(number_dtype, np.integer):
            choices = {
                ResizeMethod.NEAREST: cv2.INTER_NEAREST_EXACT,
                ResizeMethod.LINEAR: cv2.INTER_LINEAR_EXACT,
            }
            if self not in choices:
                raise ValueError(
                    f"The {self.value} interpolation method cannot be used for integers with the CV2 backend."
                )
            return choices[self]
        raise ValueError(f"The dtype {dtype} cannot be processed with the CV2 backend.")

    def get_pil_method(self) -> Image.Resampling:
        """Obtain the constant specifying the interpolation method for the PIL library."""
        choices = {
            ResizeMethod.NEAREST: Image.Resampling.NEAREST,
            ResizeMethod.LINEAR: Image.Resampling.BILINEAR,
            ResizeMethod.CUBIC: Image.Resampling.BICUBIC,
        }
        return choices[self]


class ResizeLib(Enum):
    """Backends available for spatial resizing of data."""

    PIL = "PIL"
    CV2 = "cv2"


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


def spatially_resize_image(
    data: np.ndarray,
    new_size: Optional[Tuple[int, int]] = None,
    scale_factors: Optional[Tuple[float, float]] = None,
    spatial_axes: Optional[Tuple[int, int]] = None,
    resize_method: ResizeMethod = ResizeMethod.LINEAR,
    resize_library: ResizeLib = ResizeLib.PIL,
) -> np.ndarray:
    """Resizes the image(s) according to given size or scale factors.

    To specify the new scale use one of `new_size` or `scale_factors` parameters.

    :param data: input image array
    :param new_size: New size of the data (height, width)
    :param scale_factors: Factors (f_height, f_width) by which to resize the image
    :param spatial_axes: Which two axes of input data represent height and width. If left as `None` they are selected
        according to standards of eo-learn features.
    :param resize_method: Interpolation method used for resizing.
    :param resize_library: Which Pyhon library to use for resizing. Default is PIL, as it supports all dtypes and
        features anti-aliasing. For cases where execution speed is crucial one can use CV2.
    """

    resize_method = ResizeMethod(resize_method)
    resize_library = ResizeLib(resize_library)

    if spatial_axes is None:
        spatial_axes = (1, 2) if data.ndim == 4 else (0, 1)
    elif len(spatial_axes) != 2 or spatial_axes[0] >= spatial_axes[1]:
        raise ValueError(
            "Spatial axes should be given as a pair of axis indices. We support only the case where the first axis is"
            " lower than the second one, i.e. (0, 1) is ok but (1, 0) is not supported."
        )

    if new_size is not None and scale_factors is None:
        height, width = new_size
    elif scale_factors is not None and new_size is None:
        f_y, f_x = scale_factors
        old_height, old_width = data.shape[spatial_axes[0]], data.shape[spatial_axes[1]]
        height, width = np.round(f_y * old_height), np.round(f_x * old_width)
    else:
        raise ValueError("Exactly one of the arguments new_size or scale_factors must be given.")

    size = (width, height)
    if resize_library is ResizeLib.CV2:
        resize_function = partial(cv2.resize, dsize=size, interpolation=resize_method.get_cv2_method(data.dtype))
    else:
        resize_function = partial(_pil_resize_ndarray, size=size, method=resize_method.get_pil_method())

    return _apply_to_spatial_axes(resize_function, data, spatial_axes)


def _pil_resize_ndarray(image: np.ndarray, size: Tuple[int, int], method: Image.Resampling) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, method))


def _apply_to_spatial_axes(
    resize_function: Callable[[np.ndarray], np.ndarray], data: np.ndarray, spatial_axes: Tuple[int, int]
) -> np.ndarray:
    """Helper function for applying resizing to spatial axes

    Recursively slices data into smaller-dimensional ones, until only the spatial axes remain. The indices of spatial
    axes have to be adjusted if the recursion-axis is smaller than either one, e.g. spatial axes (1, 2) become (0, 1)
    after splitting the 3D data along axis 0 into 2D arrays.

    After achieving 2D data slices the resizing function is applied. The data is then reconstructed into original form.
    """

    if data.ndim <= 2:
        return resize_function(data)

    axis = next(i for i in range(data.ndim) if i not in spatial_axes)
    data = np.moveaxis(data, axis, 0)

    ax1, ax2 = (ax if axis > ax else ax - 1 for ax in spatial_axes)

    mapped_slices = [_apply_to_spatial_axes(resize_function, data_slice, (ax1, ax2)) for data_slice in data]
    return np.moveaxis(np.stack(mapped_slices), 0, axis)


def resize_images(
    data: np.ndarray,
    new_size: Optional[Tuple[int, int]] = None,
    scale_factors: Optional[Tuple[float, float]] = None,
    anti_alias: bool = True,
    interpolation: str = "linear",
) -> np.ndarray:
    """Resizes the image(s) according to given size or scale factors.

    To specify the new scale use one of `new_size` or `scale_factors` parameters.

    :param data: input image array
    :param new_size: New size of the data (height, width)
    :param scale_factors: Factors (fy,fx) by which to resize the image
    :param anti_alias: Use anti aliasing smoothing operation when downsampling. Default is True.
    :param interpolation: Interpolation method used for resampling.
                          One of 'nearest', 'linear', 'cubic'. Default is 'linear'.
    """
    warnings.warn(
        "The function `resize_images` is deprecated and will be removed. Please switch to `spatially_resize_image`.",
        EODeprecationWarning,
    )
    inter_methods = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

    # Number of dimensions of input data
    ndims = data.ndim

    height_width_axis = {2: (0, 1), 3: (0, 1), 4: (1, 2)}

    # Old height and width
    old_size = [data.shape[axis] for axis in height_width_axis[ndims]]

    if new_size is not None and scale_factors is None:
        scale_factors = [new / old for old, new in zip(old_size, new_size)]
    elif scale_factors is not None and new_size is None:
        new_size = [int(size * factor) for size, factor in zip(old_size, scale_factors)]
    else:
        raise ValueError("Exactly one of the arguments new_size, scale_factors must be given.")

    if interpolation not in inter_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation}")

    interpolation_method = inter_methods[interpolation]
    downscaling = scale_factors[0] < 1 or scale_factors[1] < 1

    def _resize2d(image):
        if downscaling and anti_alias:
            # Sigma computation based on skimage resize implementation
            sigmas = [((1 / s) - 1) / 2 for s in scale_factors]

            # Limit sigma values above 0
            sigma_y, sigma_x = [max(1e-8, sigma) for sigma in sigmas]
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)

        height, width = new_size
        resized = cv2.resize(image, (width, height), interpolation=interpolation_method)

        return resized

    _resize3d = lambda x: map_over_axis(x, _resize2d, axis=2)  # pylint: disable=unnecessary-lambda-assignment
    _resize4d = lambda x: map_over_axis(x, _resize3d, axis=0)  # pylint: disable=unnecessary-lambda-assignment

    # Choose a resize method based on number of dimensions
    resize_methods = {2: _resize2d, 3: _resize3d, 4: _resize4d}

    resize_method = resize_methods[ndims]

    return resize_method(data)
