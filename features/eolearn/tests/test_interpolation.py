"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import dataclasses
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pytest

from sentinelhub.testing_utils import assert_statistics_match

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.features import (
    AkimaInterpolationTask,
    BSplineInterpolationTask,
    CubicInterpolationTask,
    CubicResamplingTask,
    KrigingInterpolationTask,
    LinearInterpolationTask,
    LinearResamplingTask,
    NearestResamplingTask,
    SplineInterpolationTask,
)


@pytest.fixture(name="test_patch")
def small_test_patch_fixture(example_eopatch):
    test_patch = EOPatch(bbox=example_eopatch.bbox, timestamps=example_eopatch.timestamps)
    required_features = (
        (FeatureType.DATA, "NDVI"),
        (FeatureType.MASK, "IS_VALID"),
        (FeatureType.MASK_TIMELESS, "LULC"),
        (FeatureType.MASK, "IS_VALID"),
        (FeatureType.MASK_TIMELESS, "RANDOM_UINT8"),
        (FeatureType.DATA, "BANDS-S2-L1C"),
    )
    for feature in required_features:
        test_patch[feature] = example_eopatch[feature][..., :20, :20, :]
    test_patch.label["RANDOM_DIGIT"] = example_eopatch.label["RANDOM_DIGIT"]
    return test_patch


@dataclasses.dataclass
class InterpolationTestCase:
    name: str
    task: EOTask
    result_len: int
    expected_statistics: Dict[str, float]
    nan_replace: Optional[float] = None

    def execute(self, eopatch):
        feature_type, feature_name, _ = self.task.renamed_feature

        result = self.task.execute(eopatch)

        if self.nan_replace is not None:
            data = result[feature_type, feature_name]
            data[np.isnan(data)] = self.nan_replace
            result[feature_type, feature_name] = data

        return result


# Some of these might be very randomly slow, but that is due to the JIT of numba
# It is hard to trigger it before the tests reliably.
INTERPOLATION_TEST_CASES = [
    InterpolationTestCase(
        "linear",
        LinearInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(0.0, 1.0),
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            unknown_value=10,
        ),
        result_len=68,
        expected_statistics=dict(exp_min=0.0, exp_max=0.82836, exp_mean=0.51187, exp_median=0.57889),
    ),
    InterpolationTestCase(
        "linear-p",
        LinearInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(0.0, 1.0),
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            unknown_value=10,
            interpolate_pixel_wise=True,
        ),
        result_len=68,
        expected_statistics=dict(exp_min=0.0, exp_max=0.82836, exp_mean=0.51187, exp_median=0.57889),
    ),
    InterpolationTestCase(
        "linear_change_timescale",
        LinearInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(0.0, 1.0),
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            unknown_value=10,
            scale_time=1,
        ),
        result_len=68,
        expected_statistics=dict(exp_min=0.0, exp_max=0.82836, exp_mean=0.51187, exp_median=0.57889),
    ),
    InterpolationTestCase(
        "cubic",
        CubicInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(0.0, 1.0),
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            resample_range=("2015-01-01", "2018-01-01", 16),
            unknown_value=5,
            bounds_error=False,
        ),
        result_len=69,
        expected_statistics=dict(exp_min=0.0, exp_max=5.0, exp_mean=1.3532, exp_median=0.638732),
    ),
    InterpolationTestCase(
        "spline",
        SplineInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(-0.3, 1.0),
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            resample_range=("2016-01-01", "2018-01-01", 5),
            spline_degree=3,
            smoothing_factor=0,
            unknown_value=0,
        ),
        result_len=147,
        expected_statistics=dict(exp_min=-0.1781458, exp_max=1.0, exp_mean=0.49738, exp_median=0.556853),
    ),
    InterpolationTestCase(
        "bspline",
        BSplineInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            unknown_value=-3,
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            resample_range=("2017-01-01", "2017-02-01", 50),
            spline_degree=5,
        ),
        result_len=1,
        expected_statistics=dict(exp_min=-0.0162962, exp_max=0.62323, exp_mean=0.319117, exp_median=0.3258836),
    ),
    InterpolationTestCase(
        "bspline-p",
        BSplineInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            unknown_value=-3,
            mask_feature=(FeatureType.MASK, "IS_VALID"),
            spline_degree=5,
            resample_range=("2017-01-01", "2017-02-01", 50),
            interpolate_pixel_wise=True,
        ),
        result_len=1,
        expected_statistics=dict(exp_min=-0.0162962, exp_max=0.62323, exp_mean=0.319117, exp_median=0.3258836),
    ),
    InterpolationTestCase(
        "akima",
        AkimaInterpolationTask(
            (FeatureType.DATA, "NDVI"), unknown_value=0, mask_feature=(FeatureType.MASK, "IS_VALID")
        ),
        result_len=68,
        expected_statistics=dict(exp_min=-0.091035, exp_max=0.8283603, exp_mean=0.51427454, exp_median=0.59095883),
    ),
    InterpolationTestCase(
        "kriging interpolation",
        KrigingInterpolationTask(
            (FeatureType.DATA, "NDVI"), result_interval=(-10, 10), resample_range=("2017-01-01", "2018-01-01", 10)
        ),
        result_len=37,
        expected_statistics=dict(exp_min=-0.183885, exp_max=0.5995388, exp_mean=0.35485545, exp_median=0.37279952),
    ),
    InterpolationTestCase(
        "nearest resample",
        NearestResamplingTask(
            (FeatureType.DATA, "NDVI"), result_interval=(0.0, 1.0), resample_range=("2016-01-01", "2018-01-01", 5)
        ),
        result_len=147,
        expected_statistics=dict(exp_min=-0.2, exp_max=0.8283603, exp_mean=0.32318678, exp_median=0.2794411),
        nan_replace=-0.2,
    ),
    InterpolationTestCase(
        "linear resample",
        LinearResamplingTask(
            (FeatureType.DATA, "NDVI"), result_interval=(0.0, 1.0), resample_range=("2016-01-01", "2018-01-01", 5)
        ),
        result_len=147,
        expected_statistics=dict(exp_min=-0.2, exp_max=0.82643485, exp_mean=0.32218185, exp_median=0.29093677),
        nan_replace=-0.2,
    ),
    InterpolationTestCase(
        "cubic resample",
        CubicResamplingTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(-0.2, 1.0),
            resample_range=("2015-01-01", "2018-01-01", 16),
            unknown_value=5,
        ),
        result_len=69,
        expected_statistics=dict(exp_min=-0.2, exp_max=5.0, exp_mean=1.209852, exp_median=0.40995836),
        nan_replace=-0.2,
    ),
    InterpolationTestCase(
        "linear custom list",
        LinearInterpolationTask(
            (FeatureType.DATA, "NDVI"),
            result_interval=(-0.2, 1.0),
            unknown_value=-2,
            parallel=True,
            resample_range=("2015-09-01", "2016-01-01", "2016-07-01", "2017-01-01", "2017-07-01"),
        ),
        result_len=5,
        expected_statistics=dict(exp_min=-0.0252167, exp_max=0.816656, exp_mean=0.49966, exp_median=0.533415),
    ),
    InterpolationTestCase(
        "linear with bands and multiple masks",
        LinearInterpolationTask(
            (FeatureType.DATA, "BANDS-S2-L1C"),
            result_interval=(0.0, 1.0),
            unknown_value=10,
            mask_feature=[
                (FeatureType.MASK, "IS_VALID"),
                (FeatureType.MASK_TIMELESS, "RANDOM_UINT8"),
                (FeatureType.LABEL, "RANDOM_DIGIT"),
            ],
        ),
        result_len=68,
        expected_statistics=dict(exp_min=0.0003, exp_max=10.0, exp_mean=0.132176, exp_median=0.086),
    ),
]


COPY_FEATURE_CASES = [
    (
        InterpolationTestCase(
            "cubic_copy_success",
            CubicInterpolationTask(
                (FeatureType.DATA, "NDVI"),
                result_interval=(0.0, 1.0),
                mask_feature=(FeatureType.MASK, "IS_VALID"),
                resample_range=("2015-01-01", "2018-01-01", 16),
                unknown_value=5,
                bounds_error=False,
                copy_features=[
                    (FeatureType.MASK, "IS_VALID"),
                    (FeatureType.DATA, "NDVI", "NDVI_OLD"),
                    (FeatureType.MASK_TIMELESS, "LULC"),
                ],
            ),
            result_len=69,
            expected_statistics={},
        ),
        True,
    ),
    (
        InterpolationTestCase(
            "cubic_copy_fail",
            CubicInterpolationTask(
                (FeatureType.DATA, "NDVI"),
                result_interval=(0.0, 1.0),
                mask_feature=(FeatureType.MASK, "IS_VALID"),
                resample_range=("2015-01-01", "2018-01-01", 16),
                unknown_value=5,
                bounds_error=False,
                copy_features=[
                    (FeatureType.MASK, "IS_VALID"),
                    (FeatureType.DATA, "NDVI"),
                    (FeatureType.MASK_TIMELESS, "LULC"),
                ],
            ),
            result_len=69,
            expected_statistics={},
        ),
        False,
    ),
]


@pytest.mark.parametrize("test_case", INTERPOLATION_TEST_CASES, ids=lambda x: x.name)
def test_interpolation(test_case: InterpolationTestCase, test_patch):
    eopatch = test_case.execute(test_patch)
    delta = 1e-4 if isinstance(test_case.task, KrigingInterpolationTask) else 1e-5

    # Check types and shapes
    assert isinstance(eopatch.timestamps, list), "Expected a list of timestamps"
    assert isinstance(eopatch.timestamps[0], datetime), "Expected timestamps of type datetime.datetime"
    assert len(eopatch.timestamps) == test_case.result_len
    assert eopatch.data["NDVI"].shape == (test_case.result_len, 20, 20, 1)

    # Check results
    feature_type, feature_name, _ = test_case.task.renamed_feature
    data = eopatch[feature_type, feature_name]
    assert_statistics_match(data, **test_case.expected_statistics, abs_delta=delta)


@pytest.mark.parametrize(("test_case", "passes"), COPY_FEATURE_CASES)
def test_copied_fields(test_case, passes, test_patch):
    if passes:
        eopatch = test_case.execute(test_patch)

        copied_features = [
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.DATA, "NDVI_OLD"),
            (FeatureType.MASK_TIMELESS, "LULC"),
        ]
        for feature in copied_features:
            assert feature in eopatch, f"Expected feature `{feature}` is not present in EOPatch"
    else:
        # Fails due to name duplication
        with pytest.raises(ValueError):
            test_case.execute(test_patch)
