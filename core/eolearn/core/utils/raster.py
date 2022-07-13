"""
Useful utilities for working with raster data, typically `numpy` arrays.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Domagoj Korais, Matic Lubej, Žiga Lukšič (Sinergise)
Copyright (c) 2017-2022 Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import numpy as np


def fast_nanpercentile(data: np.ndarray, percentile: float, *, method: str = "linear") -> np.ndarray:
    """This is an alternative implementation of `numpy.nanpercentile`. For cases where the size of the first dimension
    is relatively small compared to the size of the entire array it works much faster than the original.

    The algorithm divides pixel data over the first axis into groups by how many NaN values they have. In each group
    NaN values are removed and `numpy.percentile` function is applied. If any series contains only NaN values also any
    percentile of that series will be NaN.

    This function differs from `numpy` implementations in the following:

        - In case the size of the first dimension of `data` is `0` this method will still return an output array with
        all NaN values. This matches with `numpy.nanpercentile` while `numpy.percentile` raises an error.
        - The output dtype of this method will be always the same as the input dtype while `numpy` implementations
        in many cases use `float64` as the output dtype.

    :param data: An array for which percentiles will be calculated along the first axis.
    :param percentile: A percentile number.
    :param method: A method for estimating the percentile. This parameter is propagated to `numpy.percentile`.
    :return: An array of percentiles and a shape equal to the shape of `data` array without the first dimension.
    """
    combined_data = np.zeros(data.shape[1:], dtype=data.dtype)

    no_data_counts = np.count_nonzero(np.isnan(data), axis=0)
    for no_data_num in np.unique(no_data_counts):
        mask = no_data_counts == no_data_num

        chunk = data[..., mask]
        time_size, sample_size = chunk.shape

        if time_size == no_data_num:
            result = np.full(sample_size, np.nan, dtype=chunk.dtype)
        else:
            chunk = chunk.flatten(order="F")
            chunk = chunk[~np.isnan(chunk)]
            chunk = chunk.reshape((time_size - no_data_num, sample_size), order="F")

            result = np.percentile(chunk, q=percentile, axis=0, method=method)

        combined_data[mask] = result

    return combined_data
