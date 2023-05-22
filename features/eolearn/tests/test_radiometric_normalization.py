"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import copy
from datetime import datetime

import numpy as np
import pytest

from sentinelhub.testing_utils import assert_statistics_match

from eolearn.core import FeatureType
from eolearn.features import (
    BlueCompositingTask,
    HistogramMatchingTask,
    HOTCompositingTask,
    MaxNDVICompositingTask,
    MaxNDWICompositingTask,
    MaxRatioCompositingTask,
    ReferenceScenesTask,
)
from eolearn.mask import MaskFeatureTask

# ruff: noqa: NPY002


@pytest.fixture(name="eopatch")
def eopatch_fixture(example_eopatch):
    np.random.seed(0)
    example_eopatch.mask["SCL"] = np.random.randint(0, 11, example_eopatch.data["BANDS-S2-L1C"].shape, np.uint8)
    blue = BlueCompositingTask(
        (FeatureType.DATA, "REFERENCE_SCENES"),
        (FeatureType.DATA_TIMELESS, "REFERENCE_COMPOSITE"),
        blue_idx=0,
        interpolation="geoville",
    )
    blue.execute(example_eopatch)
    return example_eopatch


DATA_TEST_FEATURE = FeatureType.DATA, "TEST"
DATA_TIMELESS_TEST_FEATURE = FeatureType.DATA_TIMELESS, "TEST"


@pytest.mark.parametrize(
    ("task", "test_feature", "expected_statistics"),
    [
        (
            MaskFeatureTask(
                (FeatureType.DATA, "BANDS-S2-L1C", "TEST"),
                (FeatureType.MASK, "SCL"),
                mask_values=[0, 1, 2, 3, 8, 9, 10, 11],
            ),
            DATA_TEST_FEATURE,
            {"exp_min": 0.0002, "exp_max": 1.4244, "exp_mean": 0.21167801, "exp_median": 0.1422},
        ),
        (
            ReferenceScenesTask(
                (FeatureType.DATA, "BANDS-S2-L1C", "TEST"), (FeatureType.SCALAR, "CLOUD_COVERAGE"), max_scene_number=5
            ),
            DATA_TEST_FEATURE,
            {"exp_min": 0.0005, "exp_max": 0.5318, "exp_mean": 0.16823094, "exp_median": 0.1404},
        ),
        (
            BlueCompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                blue_idx=0,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            {"exp_min": 0.0005, "exp_max": 0.5075, "exp_mean": 0.11658352, "exp_median": 0.0833},
        ),
        (
            HOTCompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                blue_idx=0,
                red_idx=2,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            {"exp_min": 0.0005, "exp_max": 0.5075, "exp_mean": 0.117758796, "exp_median": 0.0846},
        ),
        (
            MaxNDVICompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                red_idx=2,
                nir_idx=7,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            {"exp_min": 0.0005, "exp_max": 0.5075, "exp_mean": 0.13430128, "exp_median": 0.0941},
        ),
        (
            MaxNDWICompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                nir_idx=6,
                swir1_idx=8,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            {"exp_min": 0.0005, "exp_max": 0.5318, "exp_mean": 0.2580135, "exp_median": 0.2888},
        ),
        (
            MaxRatioCompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                blue_idx=0,
                nir_idx=6,
                swir1_idx=8,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            {"exp_min": 0.0006, "exp_max": 0.5075, "exp_mean": 0.13513365, "exp_median": 0.0958},
        ),
        (
            HistogramMatchingTask(
                (FeatureType.DATA, "BANDS-S2-L1C", "TEST"), (FeatureType.DATA_TIMELESS, "REFERENCE_COMPOSITE")
            ),
            DATA_TEST_FEATURE,
            {"exp_min": -0.049050678, "exp_max": 0.68174845, "exp_mean": 0.1165936, "exp_median": 0.08370649},
        ),
    ],
)
def test_radiometric_normalization(eopatch, task, test_feature, expected_statistics):
    initial_patch = copy.deepcopy(eopatch)
    eopatch = task.execute(eopatch)

    assert isinstance(eopatch.timestamps, list), "Expected a list of timestamps"
    assert isinstance(eopatch.timestamps[0], datetime), "Expected timestamps of type datetime.datetime"

    assert_statistics_match(eopatch[test_feature], **expected_statistics, abs_delta=1e-3)

    del eopatch[test_feature]
    assert initial_patch == eopatch, "Other features of the EOPatch were affected."
