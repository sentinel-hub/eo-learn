"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureType
from eolearn.core.types import Feature
from eolearn.features import FilterTimeSeriesTask, SimpleFilterTask
from eolearn.features.feature_manipulation import SpatialResizeTask
from eolearn.features.utils import ResizeParam

DUMMY_BBOX = BBox((0, 0, 1, 1), CRS(3857))
# ruff: noqa: NPY002


@pytest.mark.parametrize("feature", [(FeatureType.DATA, "BANDS-S2-L1C"), (FeatureType.LABEL, "IS_CLOUDLESS")])
def test_simple_filter_task_filter_all(example_eopatch: EOPatch, feature):
    filter_all_task = SimpleFilterTask(feature, filter_func=lambda _: False)
    filtered_eopatch = filter_all_task.execute(example_eopatch)

    assert filtered_eopatch is not example_eopatch
    assert filtered_eopatch.data["CLP"].shape == (0, 101, 100, 1)
    assert filtered_eopatch.scalar["CLOUD_COVERAGE"].shape == (0, 1)
    assert len(filtered_eopatch.vector["CLM_VECTOR"]) == 0
    assert np.array_equal(filtered_eopatch.mask_timeless["LULC"], example_eopatch.mask_timeless["LULC"])
    assert filtered_eopatch.timestamps == []


@pytest.mark.parametrize("feature", [(FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "CLOUD_COVERAGE")])
def test_simple_filter_task_filter_nothing(example_eopatch: EOPatch, feature):
    filter_all_task = SimpleFilterTask(feature, filter_func=lambda _: True)
    filtered_eopatch = filter_all_task.execute(example_eopatch)

    assert filtered_eopatch is not example_eopatch
    assert filtered_eopatch == example_eopatch


@pytest.mark.parametrize(
    "invalid_feature",
    [(FeatureType.VECTOR, "foo"), (FeatureType.VECTOR_TIMELESS, "bar"), (FeatureType.MASK_TIMELESS, "foobar")],
)
def test_simple_filter_invalid_feature(invalid_feature: Feature):
    with pytest.raises(ValueError):
        SimpleFilterTask(invalid_feature, filter_func=lambda _: True)


def test_content_after_time_filter():
    timestamps = [
        datetime.datetime(2017, 1, 1, 10, 4, 7),
        datetime.datetime(2017, 1, 4, 10, 14, 5),
        datetime.datetime(2017, 1, 11, 10, 3, 51),
        datetime.datetime(2017, 1, 14, 10, 13, 46),
        datetime.datetime(2017, 1, 24, 10, 14, 7),
        datetime.datetime(2017, 2, 10, 10, 1, 32),
        datetime.datetime(2017, 2, 20, 10, 6, 35),
        datetime.datetime(2017, 3, 2, 10, 0, 20),
        datetime.datetime(2017, 3, 12, 10, 7, 6),
        datetime.datetime(2017, 3, 15, 10, 12, 14),
    ]
    data = np.random.rand(10, 100, 100, 3)

    new_start, new_end = 4, -3

    eop = EOPatch(bbox=DUMMY_BBOX, timestamps=timestamps, data={"data": data})

    filter_task = FilterTimeSeriesTask(start_date=timestamps[new_start], end_date=timestamps[new_end])
    filtered_eop = filter_task.execute(eop)

    assert filtered_eop is not eop
    assert filtered_eop.timestamps == timestamps[new_start : new_end + 1]
    assert np.array_equal(filtered_eop.data["data"], data[new_start : new_end + 1, ...])


@pytest.mark.parametrize(
    ("resize_type", "height_param", "width_param", "features_call", "features_check", "outputs"),
    [
        (ResizeParam.NEW_SIZE, 50, 70, ("data", "CLP"), ("data", "CLP"), (68, 50, 70, 1)),
        (ResizeParam.NEW_SIZE, 50, 70, ("data", "CLP"), ("mask", "CLM"), (68, 101, 100, 1)),
        (ResizeParam.NEW_SIZE, 50, 70, ..., ("data", "CLP"), (68, 50, 70, 1)),
        (ResizeParam.NEW_SIZE, 50, 70, ..., ("mask", "CLM"), (68, 50, 70, 1)),
        (ResizeParam.NEW_SIZE, 50, 70, ("data", "CLP", "CLP_small"), ("data", "CLP_small"), (68, 50, 70, 1)),
        (ResizeParam.NEW_SIZE, 50, 70, ("data", "CLP", "CLP_small"), ("data", "CLP"), (68, 101, 100, 1)),
        (ResizeParam.SCALE_FACTORS, 2, 2, ("data", "CLP"), ("data", "CLP"), (68, 202, 200, 1)),
        (ResizeParam.SCALE_FACTORS, 0.5, 2, ("data", "CLP"), ("data", "CLP"), (68, 50, 200, 1)),
        (ResizeParam.SCALE_FACTORS, 0.1, 0.1, ("data", "CLP"), ("data", "CLP"), (68, 10, 10, 1)),
        (ResizeParam.RESOLUTION, 5, 5, ("data", "CLP"), ("data", "CLP"), (68, 202, 200, 1)),
        (ResizeParam.RESOLUTION, 20, 20, ("data", "CLP"), ("data", "CLP"), (68, 50, 50, 1)),
        (ResizeParam.RESOLUTION, 5, 20, ("data", "CLP"), ("data", "CLP"), (68, 202, 50, 1)),
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_spatial_resize_task(
    example_eopatch, resize_type, height_param, width_param, features_call, features_check, outputs
):
    # Warnings occur due to lossy casting in the downsampling procedure

    resize = SpatialResizeTask(
        resize_type=resize_type, height_param=height_param, width_param=width_param, features=features_call
    )
    assert resize(example_eopatch)[features_check].shape == outputs


def test_spatial_resize_task_exception():
    with pytest.raises(ValueError):
        SpatialResizeTask(features=("mask", "CLM"), resize_type="blabla", height_param=20, width_param=20)
