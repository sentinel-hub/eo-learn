"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
from eolearn.features import LocalBinaryPatternTask


@pytest.mark.parametrize(
    "task, expected_min, expected_max, expected_mean, expected_median",
    ([LocalBinaryPatternTask((FeatureType.DATA, "NDVI", "lbp"), nb_points=24, radius=3), 0.0, 25.0, 15.8313, 21.0],),
)
def test_local_binary_pattern(small_ndvi_eopatch, task, expected_min, expected_max, expected_mean, expected_median):
    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in small_ndvi_eopatch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4

    haralick = eopatch.data["lbp"]
    assert np.min(haralick) == approx(expected_min, abs=delta)
    assert np.max(haralick) == approx(expected_max, abs=delta)
    assert np.mean(haralick) == approx(expected_mean, abs=delta)
    assert np.median(haralick) == approx(expected_median, abs=delta)
