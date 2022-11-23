"""
Credits:
Copyright (c) 2018-2019 Hugo Fournier (Magellium)
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy

import numpy as np
import pytest

from sentinelhub.testing_utils import test_numpy_data

from eolearn.core import FeatureType
from eolearn.features import HaralickTask

FEATURE = (FeatureType.DATA, "NDVI", "haralick")
OUTPUT_FEATURE = (FeatureType.DATA, "haralick")


@pytest.mark.parametrize(
    "task, expected_statistics",
    (
        [
            HaralickTask(FEATURE, texture_feature="contrast", angle=0, levels=255, window_size=3),
            {"exp_min": 3.5, "exp_max": 9079.0, "exp_mean": 965.8295, "exp_median": 628.5833},
        ],
        [
            HaralickTask(FEATURE, texture_feature="sum_of_square_variance", angle=np.pi / 2, levels=8, window_size=5),
            {"exp_min": 0.96899, "exp_max": 48.7815, "exp_mean": 23.0229, "exp_median": 23.8987},
        ],
        [
            HaralickTask(FEATURE, texture_feature="sum_entropy", angle=-np.pi / 2, levels=8, window_size=7),
            {"exp_min": 0, "exp_max": 1.7463, "exp_mean": 0.5657, "exp_median": 0.50558},
        ],
    ),
)
def test_haralick(small_ndvi_eopatch, task, expected_statistics):
    eopatch = copy.deepcopy(small_ndvi_eopatch)
    task.execute(eopatch)

    test_numpy_data(eopatch[OUTPUT_FEATURE], exp_shape=(10, 20, 20, 1), **expected_statistics, delta=1e-4)

    del eopatch[OUTPUT_FEATURE]
    assert small_ndvi_eopatch == eopatch, "Other features of the EOPatch were affected."
