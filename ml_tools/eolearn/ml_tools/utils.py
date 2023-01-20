"""
The utilities module is a collection of methods used across the eolearn package, such as checking whether two objects
are deeply equal, padding of an image, etc.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from typing import Any, Optional

import numpy as np

from sentinelhub.exceptions import deprecated_function

from eolearn.core.exceptions import EODeprecationWarning


# This code was copied from https://gist.github.com/seberg/3866040
@deprecated_function(
    category=EODeprecationWarning, message_suffix="Please use `numpy.lib.stride_tricks.sliding_window_view` instead."
)
def rolling_window(
    array: np.ndarray,
    window: Any = (0,),
    asteps: Optional[Any] = None,
    wsteps: Optional[Any] = None,
    axes: Optional[Any] = None,
    toend: bool = True,
) -> np.ndarray:
    """Create a view of `array` which for every point gives the n-dimensional neighbourhood of size window. New
    dimensions are added at the end of `array` or after the corresponding original dimension.

    Examples:

    .. code-block:: python

        >>> a = np.arange(9).reshape(3,3)
        >>> rolling_window(a, (2,2))
        array([[[[0, 1],
                 [3, 4]],
                [[1, 2],
                 [4, 5]]],
               [[[3, 4],
                 [6, 7]],
                [[4, 5],
                 [7, 8]]]])

    Or to create non-overlapping windows, but only along the first dimension:

    .. code-block:: python

        >>> rolling_window(a, (2,0), asteps=(2,1))
        array([[[0, 3],
                [1, 4],
                [2, 5]]])

    Note that the `0` is discared, so that the output dimension is `3`:

    .. code-block:: python

        >>> rolling_window(a, (2,0), asteps=(2,1)).shape
        (1, 3, 2)

    This is useful for example to calculate the maximum in all (overlapping) 2x2 submatrixes:

    .. code-block:: python

        >>> rolling_window(a, (2,2)).max((2,3))
        array([[4, 5],
               [7, 8]])

    Or delay embedding (3D embedding with delay 2):

    .. code-block:: python

        >>> x = np.arange(10)
        >>> rolling_window(x, 3, wsteps=2)
        array([[0, 2, 4],
               [1, 3, 5],
               [2, 4, 6],
               [3, 5, 7],
               [4, 6, 8],
               [5, 7, 9]])

    :param array: Array to which the rolling window is applied
    :param window: Either a single integer to create a window of only the last axis or a tuple to create it for
        the last len(window) axes. 0 can be used to ignore a dimension in the window.
    :param asteps: Aligned at the last axis, new steps for the original array, i.e. for creation of non-overlapping
        windows. (Equivalent to slicing result)
    :param wsteps: Steps for the added window dimensions. These can be 0 to repeat values along the axis.
    :param axes: If given, must have the same size as window. In this case window is interpreted as the size in the
        dimension given by axes. E.g. a window of (2, 1) is equivalent to window=2 and axis=-2.
    :param toend: If False, the new dimensions are right after the corresponding original dimension, instead of at
        the end of the array. Adding the new axes at the end makes it easier to get the neighborhood, however
        toend=False will give a more intuitive result if you view the whole array.
    :returns: A view on `array` which is smaller to fit the windows and has windows added dimensions (0s not counting),
        i.e. every point of `array` is an array of size window.
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        new_window = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            new_window[axis] = size
        window = new_window

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps) :] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger than the original:
    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window) :] = window
        _window = _.copy()
        _[-len(window) :] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)
