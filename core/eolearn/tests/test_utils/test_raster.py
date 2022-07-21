"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Domagoj Korais, Matic Lubej, Žiga Lukšič (Sinergise)
Copyright (c) 2017-2022 Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core.utils.raster import fast_nanpercentile


@pytest.mark.parametrize("size", [0, 5])
@pytest.mark.parametrize("percentile", [0, 1.5, 50, 80.99, 100])
@pytest.mark.parametrize("nan_ratio", [0, 0.05, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16])
@pytest.mark.parametrize("method", ["linear", "normal_unbiased"])
def test_fast_nanpercentile(size: int, percentile: float, nan_ratio: float, dtype: type, method: str):
    data_shape = (size, 3, 2, 4)
    data = np.random.rand(*data_shape)
    data[data < nan_ratio] = np.nan

    if np.issubdtype(dtype, np.integer):
        data *= 1000
    data = data.astype(dtype)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        expected_result = np.nanpercentile(data, q=percentile, axis=0, method=method).astype(data.dtype)

    result = fast_nanpercentile(data, percentile, method=method)

    assert_array_equal(result, expected_result)
