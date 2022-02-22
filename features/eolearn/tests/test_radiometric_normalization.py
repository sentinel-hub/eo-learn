"""
Credits:
Copyright (c) 2018-2019 Johannes Schmid (GeoVille)
Copyright (c) 2017-2022 Matej Aleksandrov, Matic Lubej, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy
from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from eolearn.core import FeatureType
from eolearn.core.eodata_io import FeatureIO
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
    "task, test_feature, expected_min, expected_max, expected_mean, expected_median",
    (
        [
            MaskFeatureTask(
                (FeatureType.DATA, "BANDS-S2-L1C", "TEST"),
                (FeatureType.MASK, "SCL"),
                mask_values=[0, 1, 2, 3, 8, 9, 10, 11],
            ),
            DATA_TEST_FEATURE,
            0.0002,
            1.4244,
            0.21167801,
            0.142,
        ],
        [
            ReferenceScenesTask(
                (FeatureType.DATA, "BANDS-S2-L1C", "TEST"), (FeatureType.SCALAR, "CLOUD_COVERAGE"), max_scene_number=5
            ),
            DATA_TEST_FEATURE,
            0.0005,
            0.5318,
            0.16823094,
            0.1404,
        ],
        [
            BlueCompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                blue_idx=0,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            0.0005,
            0.5075,
            0.11658352,
            0.0833,
        ],
        [
            HOTCompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                blue_idx=0,
                red_idx=2,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            0.0005,
            0.5075,
            0.117758796,
            0.0846,
        ],
        [
            MaxNDVICompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                red_idx=2,
                nir_idx=7,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            0.0005,
            0.5075,
            0.13430128,
            0.0941,
        ],
        [
            MaxNDWICompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                nir_idx=6,
                swir1_idx=8,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            0.0005,
            0.5318,
            0.2580135,
            0.2888,
        ],
        [
            MaxRatioCompositingTask(
                (FeatureType.DATA, "REFERENCE_SCENES"),
                (FeatureType.DATA_TIMELESS, "TEST"),
                blue_idx=0,
                nir_idx=6,
                swir1_idx=8,
                interpolation="geoville",
            ),
            DATA_TIMELESS_TEST_FEATURE,
            0.0006,
            0.5075,
            0.13513365,
            0.0958,
        ],
        [
            HistogramMatchingTask(
                (FeatureType.DATA, "BANDS-S2-L1C", "TEST"), (FeatureType.DATA_TIMELESS, "REFERENCE_COMPOSITE")
            ),
            DATA_TEST_FEATURE,
            -0.049050678,
            0.68174845,
            0.1165936,
            0.08370649,
        ],
    ),
)
def test_haralick(eopatch, task, test_feature, expected_min, expected_max, expected_mean, expected_median):
    initial_patch = copy.deepcopy(eopatch)
    eopatch = task.execute(eopatch)

    # Test that no other features were modified
    for feature, value in initial_patch.data.items():
        if isinstance(value, FeatureIO):
            value = value.load()
        assert_array_equal(value, eopatch.data[feature], err_msg=f"EOPatch data feature '{feature}' has changed")

    assert isinstance(eopatch.timestamp, list), "Expected a list of timestamps"
    assert isinstance(eopatch.timestamp[0], datetime), "Expected timestamps of type datetime.datetime"

    delta = 1e-3
    result = eopatch[test_feature]
    assert np.nanmin(result) == approx(expected_min, abs=delta)
    assert np.nanmax(result) == approx(expected_max, abs=delta)
    assert np.nanmean(result) == approx(expected_mean, abs=delta)
    assert np.nanmedian(result) == approx(expected_median, abs=delta)
