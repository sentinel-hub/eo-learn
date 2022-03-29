"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy

import numpy as np
from numpy.testing import assert_array_equal
from pytest import approx

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
from eolearn.features import HOGTask


def test_hog(small_ndvi_eopatch):
    task = HOGTask(
        (FeatureType.DATA, "NDVI", "hog"),
        orientations=9,
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2),
        visualize=True,
        visualize_feature_name="hog_visu",
    )

    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in small_ndvi_eopatch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    delta = 1e-4
    for feature, expected_min, expected_max, expected_mean, expected_median in [
        ("hog", 0.0, 0.5567, 0.0931, 0.0),
        ("hog_visu", 0.0, 0.3241, 0.0105, 0.0),
    ]:
        hog = eopatch.data[feature]
        assert np.min(hog) == approx(expected_min, abs=delta)
        assert np.max(hog) == approx(expected_max, abs=delta)
        assert np.mean(hog) == approx(expected_mean, abs=delta)
        assert np.median(hog) == approx(expected_median, abs=delta)
