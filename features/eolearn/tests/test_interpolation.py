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
            # parallel=True,  # commented out for speed while refactoring
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
        expected_statistics=dict(exp_min=0.0, exp_max=5.0, exp_mean=1.3592644, exp_median=0.6174331),
    ),
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
        expected_statistics=dict(exp_min=0.0, exp_max=5.0, exp_mean=1.3592644, exp_median=0.6174331),
    ),
]


@pytest.mark.parametrize("test_case", INTERPOLATION_TEST_CASES)
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


@pytest.mark.parametrize("test_case", COPY_FEATURE_CASES)
def test_copied_fields(test_case, test_patch):
    try:
        eopatch = test_case.execute(test_patch)
    except ValueError:
        eopatch = None

    if eopatch is not None:
        copied_features = [
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.DATA, "NDVI_OLD"),
            (FeatureType.MASK_TIMELESS, "LULC"),
        ]
        for feature in copied_features:
            assert feature in eopatch, f"Expected feature `{feature}` is not present in EOPatch"


def test_interpolation_exact1(test_patch):
    task = INTERPOLATION_TEST_CASES[0].task
    eopatch = task.execute(test_patch)

    # Check types and shapes
    data = eopatch.data["NDVI"]
    assert data.shape == (68, 20, 20, 1)
    assert np.sum(data) == pytest.approx(13922.76564)
    expected_data = np.array(
        [0.79299557, 0.75465041, 0.71630526, 0.69713271, 0.70360482, 0.66328233, 0.62295991, 0.34070268, 0.34070268]
        + [0.30038023, 0.31299213, 0.11978609, 0.1555274, 0.44927531, 0.53047365, 0.5507732, 0.61167198, 0.63197154]
        + [0.56646109, 0.75504524, 0.72438276, 0.69372028, 0.6630578, 0.73763263, 0.76249093, 0.7608695, 0.72831064]
        + [0.66319299, 0.64944649, 0.50559729, 0.26584867, 0.32763132, 0.38941398, 0.23915902, 0.14852071, 0.25489235]
        + [0.36126396, 0.57400721, 0.53905129, 0.64215463, 0.67197317, 0.7316103, 0.73692691, 0.74224353, 0.74756014]
        + [0.75796175, 0.72173917, 0.57328522, 0.70037711, 0.57889175, 0.64796793, 0.71704417, 0.71936429, 0.72632456]
        + [0.71699739, 0.63335538, 0.54971331, 0.50789231, 0.31887206, 0.57065219, 0.54396003, 0.48462179, 0.26372707]
        + [0.21954814, 0.13119026, 0.2547307, 0.19220223, 0.16093799]
    )
    np.testing.assert_array_almost_equal(data[..., 3, 5, 0], expected_data)


def test_interpolation_exact2(test_patch):
    task = INTERPOLATION_TEST_CASES[3].task
    eopatch = task.execute(test_patch)

    # Check types and shapes
    data = eopatch.data["NDVI"]
    assert data.shape == (69, 20, 20, 1)
    assert np.sum(data) == pytest.approx(37348.08984375)
    expected_data = np.array(
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        + [0.78979117, 0.7408045, 0.70526695, 0.696489, 0.6977065, 0.63019097, 0.51606977, 0.39103627, 0.2907841]
        + [0.25100663, 0.30731565, 0.17120162, 0.19468854, 0.43449947, 0.64237213, 0.7847737, 0.85486186, 0.8457996]
        + [0.7507502, 0.57317805, 0.7716818, 0.78446645, 0.6473747, 0.665229, 0.7470188, 0.75878334, 0.7028236]
        + [0.6593944, 0.6102387, 0.48117316, 0.33040872, 0.2214989, 0.21799724, 0.3616935, 0.32517183, 0.10130903]
        + [0.056695, 0.18499486, 0.42027038, 0.57471883, 0.5433612, 0.70258653, 0.73490167, 0.7338911, 0.7462087]
        + [0.7496303, 0.6170033, 0.6663743, 0.7670803, 0.7346426, 0.6474287, 0.48783618, 0.4460752, 0.2079632]
        + [0.09893011, 0.2601632, 5.0]
    )
    np.testing.assert_array_almost_equal(data[..., 3, 5, 0], expected_data)
