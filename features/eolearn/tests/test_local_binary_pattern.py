"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import copy

import pytest

from sentinelhub.testing_utils import test_numpy_data

from eolearn.core import FeatureType
from eolearn.features import LocalBinaryPatternTask

LBP_FEATURE = (FeatureType.DATA, "NDVI", "lbp")
OUTPUT_FEATURE = (FeatureType.DATA, "lbp")


@pytest.mark.parametrize(
    "task, expected_statistics",
    (
        [
            LocalBinaryPatternTask(LBP_FEATURE, nb_points=24, radius=3),
            {"exp_min": 0.0, "exp_max": 25.0, "exp_mean": 15.8313, "exp_median": 21.0},
        ],
    ),
)
def test_local_binary_pattern(small_ndvi_eopatch, task, expected_statistics):
    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    test_numpy_data(eopatch[OUTPUT_FEATURE], exp_shape=(10, 20, 20, 1), **expected_statistics, delta=1e-4)

    del eopatch[OUTPUT_FEATURE]
    assert small_ndvi_eopatch == eopatch, "Other features of the EOPatch were affected."
