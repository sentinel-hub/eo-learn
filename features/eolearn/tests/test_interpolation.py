"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)
Copyright (c) 2018-2019 Filip Koprivec (Jožef Stefan Institute)
Copyright (c) 2018-2019 William Ouellette

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import dataclasses
from datetime import datetime
from typing import Optional

import numpy as np
import pytest
from pytest import approx

from eolearn.core import EOTask, FeatureType
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


@dataclasses.dataclass
class InterpolationTestCase:
    name: str
    task: EOTask
    result_len: int
    img_min: float
    img_max: float
    img_mean: float
    img_median: float
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
        img_min=0.0,
        img_max=10.0,
        img_mean=0.720405,
        img_median=0.59765935,
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
        img_min=0.0,
        img_max=10.0,
        img_mean=0.720405,
        img_median=0.59765935,
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
        img_min=0.0,
        img_max=10.0,
        img_mean=0.7204042,
        img_median=0.59765697,
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
        img_min=0.0,
        img_max=5.0,
        img_mean=1.3592644,
        img_median=0.6174331,
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
        img_min=-0.3,
        img_max=1.0,
        img_mean=0.492752,
        img_median=0.53776133,
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
        img_min=-0.032482587,
        img_max=0.701796,
        img_mean=0.42080238,
        img_median=0.42889267,
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
        img_min=-0.032482587,
        img_max=0.701796,
        img_mean=0.42080238,
        img_median=0.42889267,
    ),
    InterpolationTestCase(
        "akima",
        AkimaInterpolationTask(
            (FeatureType.DATA, "NDVI"), unknown_value=0, mask_feature=(FeatureType.MASK, "IS_VALID")
        ),
        result_len=68,
        img_min=-0.13793105,
        img_max=0.860242,
        img_mean=0.53159297,
        img_median=0.59087014,
    ),
    InterpolationTestCase(
        "kriging interpolation",
        KrigingInterpolationTask(
            (FeatureType.DATA, "NDVI"), result_interval=(-10, 10), resample_range=("2017-01-01", "2018-01-01", 10)
        ),
        result_len=37,
        img_min=-0.19972801,
        img_max=0.6591711,
        img_mean=0.3773447,
        img_median=0.3993981,
    ),
    InterpolationTestCase(
        "nearest resample",
        NearestResamplingTask(
            (FeatureType.DATA, "NDVI"), result_interval=(0.0, 1.0), resample_range=("2016-01-01", "2018-01-01", 5)
        ),
        result_len=147,
        img_min=-0.2,
        img_max=0.860242,
        img_mean=0.35143828,
        img_median=0.37481314,
        nan_replace=-0.2,
    ),
    InterpolationTestCase(
        "linear resample",
        LinearResamplingTask(
            (FeatureType.DATA, "NDVI"), result_interval=(0.0, 1.0), resample_range=("2016-01-01", "2018-01-01", 5)
        ),
        result_len=147,
        img_min=-0.2,
        img_max=0.8480114,
        img_mean=0.350186,
        img_median=0.3393997,
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
        img_min=-0.2,
        img_max=5.0,
        img_mean=1.234881997,
        img_median=0.465670556,
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
        img_min=-0.032482587,
        img_max=0.8427637,
        img_mean=0.5108417,
        img_median=0.5042224,
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
        img_min=0.000200,
        img_max=10.0,
        img_mean=0.3487376,
        img_median=0.10036667,
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
        img_min=0.0,
        img_max=5.0,
        img_mean=1.3592644,
        img_median=0.6174331,
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
        img_min=0.0,
        img_max=5.0,
        img_mean=1.3592644,
        img_median=0.6174331,
    ),
]


@pytest.mark.parametrize("test_case", INTERPOLATION_TEST_CASES)
def test_interpolation(test_case, example_eopatch):
    eopatch = test_case.execute(example_eopatch)

    # Check types and shapes
    assert isinstance(eopatch.timestamp, list), "Expected a list of timestamps"
    assert isinstance(eopatch.timestamp[0], datetime), "Expected timestamps of type datetime.datetime"
    assert len(eopatch.timestamp) == test_case.result_len
    assert eopatch.data["NDVI"].shape == (test_case.result_len, 101, 100, 1)

    # Check results
    delta = 1e-5  # Can't be higher accuracy because of Kriging interpolation
    feature_type, feature_name, _ = test_case.task.renamed_feature
    data = eopatch[feature_type, feature_name]

    assert np.min(data) == approx(test_case.img_min, abs=delta)
    assert np.max(data) == approx(test_case.img_max, abs=delta)
    assert np.mean(data) == approx(test_case.img_mean, abs=delta)
    assert np.median(data) == approx(test_case.img_median, abs=delta)


@pytest.mark.parametrize("test_case", COPY_FEATURE_CASES)
def test_copied_fields(test_case, example_eopatch):
    try:
        eopatch = test_case.execute(example_eopatch)
    except ValueError:
        eopatch = None

    if eopatch is not None:
        copied_features = [
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.DATA, "NDVI_OLD"),
            (FeatureType.MASK_TIMELESS, "LULC"),
        ]
        for feature in copied_features:
            assert feature in eopatch.get_feature_list(), f"Expected feature `{feature}` is not present in EOPatch"
