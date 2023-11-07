"""
Utilities for EOPatch feature modification

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import warnings
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

import cv2
import numpy as np

from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.utils.common import _apply_to_spatial_axes

_PIL_IMPORT_MESSAGE = "The PIL backend is not installed by default. Please install the `pillow` package."

if TYPE_CHECKING:
    from PIL import Image


class ResizeParam(Enum):
    """Descriptors of spatial-resizing parameter options."""

    NEW_SIZE = "new_size"
    SCALE_FACTORS = "scale_factors"
    RESOLUTION = "resolution"


class ResizeMethod(Enum):
    """Methods available for spatial resizing of data."""

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"

    def get_cv2_method(self, dtype: np.dtype | type) -> int:
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
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel
        except ImportError as exception:
            raise ImportError(_PIL_IMPORT_MESSAGE) from exception
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

    def get_compatible_dtype(self, dtype: np.dtype | type) -> np.dtype:
        """Returns a suitable dtype with which the library can work. Warns if information loss could occur."""
        if self is ResizeLib.CV2:
            lossless = {bool: np.uint8, np.float16: np.float32}
            infoloss = {x: np.int32 for x in (np.uint32, np.int64, np.uint64, int)}
        if self is ResizeLib.PIL:
            lossless = {np.float16: np.float32}
            infoloss = {x: np.int32 for x in (np.uint16, np.uint32, np.int64, np.uint64, int)}

        lossless_casts = {np.dtype(k): np.dtype(v) for k, v in lossless.items()}
        infoloss_casts = {np.dtype(k): np.dtype(v) for k, v in infoloss.items()}
        return self._extract_compatible_dtype(dtype, lossless_casts, infoloss_casts)

    @staticmethod
    def _extract_compatible_dtype(
        dtype: np.dtype | type, lossless_casts: dict[np.dtype, np.dtype], infoloss_casts: dict[np.dtype, np.dtype]
    ) -> np.dtype:
        """Searches the dictionaries and extract the appropriate dtype. Warns of data loss if it could occur."""
        dtype = np.dtype(dtype)
        if dtype in infoloss_casts:
            cast_dtype = infoloss_casts[dtype]
            warnings.warn(
                f"Data of type {dtype} will be processed as {cast_dtype}, possible information loss during procedure.",
                EORuntimeWarning,
            )
            return cast_dtype
        return lossless_casts.get(dtype, dtype)


def spatially_resize_image(
    data: np.ndarray,
    new_size: tuple[int, int] | None = None,
    scale_factors: tuple[float, float] | None = None,
    spatial_axes: tuple[int, int] | None = None,
    resize_method: ResizeMethod = ResizeMethod.LINEAR,
    resize_library: ResizeLib = ResizeLib.CV2,
) -> np.ndarray:
    """Resizes the image(s) according to given size or scale factors.

    To specify the new scale use one of `new_size` or `scale_factors` parameters.

    :param data: input image array
    :param new_size: New size of the data (height, width)
    :param scale_factors: Factors (f_height, f_width) by which to resize the image
    :param spatial_axes: Which two axes of input data represent height and width. If left as `None` they are selected
        according to standards of eo-learn features.
    :param resize_method: Interpolation method used for resizing.
    :param resize_library: Which Python library to use for resizing. Default is CV2 because it is faster, but one can
        use PIL, which features anti-aliasing.
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
        height, width = round(f_y * old_height), round(f_x * old_width)
    else:
        raise ValueError("Exactly one of the arguments new_size or scale_factors must be given.")

    size = (width, height)
    old_dtype, new_dtype = data.dtype, resize_library.get_compatible_dtype(data.dtype)
    data = data.astype(new_dtype)

    if resize_library is ResizeLib.CV2:
        resize_function = partial(cv2.resize, dsize=size, interpolation=resize_method.get_cv2_method(data.dtype))
    else:
        resize_function = partial(_pil_resize_ndarray, size=size, method=resize_method.get_pil_method())

    resized_data = _apply_to_spatial_axes(resize_function, data, spatial_axes)

    if np.issubdtype(old_dtype, np.unsignedinteger):
        # remove negatives so that there are no underflows when casting back
        resized_data = resized_data.clip(min=0)
    return resized_data.astype(old_dtype)


def _pil_resize_ndarray(image: np.ndarray, size: tuple[int, int], method: Image.Resampling) -> np.ndarray:
    try:
        from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError as exception:
        raise ImportError(_PIL_IMPORT_MESSAGE) from exception
    return np.array(Image.fromarray(image).resize(size, method))
