import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pytest

from sentinelhub import CRS, BBox
from sentinelhub.testing_utils import assert_statistics_match

from eolearn.core.constants import FeatureType
from eolearn.core.types import FeatureSpec, FeaturesSpecification
from eolearn.core.utils.parsing import parse_features
from eolearn.core.utils.testing import PatchGeneratorConfig, generate_eopatch


def test_generate_eopatch_set_bbox_timestamps() -> None:
    bbox = BBox((0, 0, 10, 10), crs=CRS("EPSG:32633"))
    timestamps = [dt.datetime(2019, 1, 1)]
    patch = generate_eopatch((FeatureType.DATA, "bands"), bbox=bbox, timestamps=timestamps)

    assert patch.bbox == bbox
    assert patch.timestamps == timestamps

    assert patch[(FeatureType.DATA, "bands")].shape[0] == len(timestamps)


@pytest.mark.parametrize(
    "config",
    [
        {"num_timestamps": 3, "max_integer_value": 1, "raster_shape": (1, 1), "depth_range": (1, 2)},
        {"num_timestamps": 7, "max_integer_value": 333, "raster_shape": (3, 27), "depth_range": (5, 15)},
    ],
)
def test_generate_eopatch_config(config: Dict[str, Any]) -> None:
    mask_feature = (FeatureType.MASK, "mask")

    patch = generate_eopatch(mask_feature, config=PatchGeneratorConfig(**config))

    time, raster_y, raster_x, depth = patch[mask_feature].shape
    assert time == config["num_timestamps"]
    assert (raster_y, raster_x) == config["raster_shape"]
    assert config["depth_range"][0] <= depth <= config["depth_range"][1]

    assert np.max(patch[mask_feature]) < config["max_integer_value"]


@pytest.mark.parametrize("seed", [0, 1, 42, 100])
@pytest.mark.parametrize(
    "features",
    [
        {},
        {FeatureType.DATA: ["bands, CLP"]},
        {FeatureType.DATA: ["bands, CLP"], FeatureType.MASK_TIMELESS: "LULC"},
        {
            FeatureType.DATA: "data",
            FeatureType.MASK: "mask",
            FeatureType.SCALAR: "scalar",
            FeatureType.LABEL: "label",
            FeatureType.DATA_TIMELESS: "data_timeless",
            FeatureType.MASK_TIMELESS: "mask_timeless",
            FeatureType.SCALAR_TIMELESS: "scalar_timeless",
            FeatureType.LABEL_TIMELESS: "label_timeless",
            FeatureType.META_INFO: "meta_info",
        },
    ],
)
def test_generate_eopatch_seed(seed: int, features: FeaturesSpecification) -> None:
    patch1 = generate_eopatch(features, seed=seed)
    patch2 = generate_eopatch(features, seed=seed)
    assert patch1 == patch2


@dataclass
class GenerateTestCase:
    data_features: FeaturesSpecification
    seed: int
    expected_statistics: Dict[FeatureSpec, Dict[str, Any]]


@pytest.mark.parametrize(
    "test_case",
    [
        GenerateTestCase(
            seed=3,
            data_features=(FeatureType.MASK, "CLM"),
            expected_statistics={
                (FeatureType.MASK, "CLM"): {
                    "exp_shape": (5, 98, 151, 2),
                    "exp_min": 0,
                    "exp_max": 255,
                    "exp_mean": 127.389,
                }
            },
        ),
        GenerateTestCase(
            seed=1,
            data_features={FeatureType.DATA: ["data"], FeatureType.MASK_TIMELESS: ["LULC", "IS_VALUE"]},
            expected_statistics={
                (FeatureType.DATA, "data"): {
                    "exp_shape": (5, 98, 151, 1),
                    "exp_min": -4.030404,
                    "exp_max": 4.406353,
                    "exp_mean": -0.005156786,
                },
                (FeatureType.MASK_TIMELESS, "LULC"): {
                    "exp_shape": (98, 151, 2),
                    "exp_min": 0,
                    "exp_max": 255,
                    "exp_mean": 127.0385,
                },
                (FeatureType.MASK_TIMELESS, "IS_VALUE"): {
                    "exp_shape": (98, 151, 2),
                    "exp_min": 0,
                    "exp_max": 255,
                    "exp_mean": 127.3140,
                },
            },
        ),
        GenerateTestCase(
            seed=100,
            data_features=[(FeatureType.SCALAR, "scalar"), (FeatureType.SCALAR_TIMELESS, "scalar_timeless")],
            expected_statistics={
                (FeatureType.SCALAR, "scalar"): {
                    "exp_shape": (5, 2),
                    "exp_min": -0.9613,
                    "exp_max": 2.24297,
                    "exp_mean": 0.7223,
                },
                (FeatureType.SCALAR_TIMELESS, "scalar_timeless"): {
                    "exp_shape": (2,),
                    "exp_min": -0.611493,
                    "exp_max": 0.0472111,
                    "exp_mean": -0.28214,
                },
            },
        ),
    ],
)
def test_generate_eopatch_data(test_case: GenerateTestCase) -> None:
    patch = generate_eopatch(test_case.data_features, seed=test_case.seed)
    for feature in parse_features(test_case.data_features):
        assert_statistics_match(patch[feature], **test_case.expected_statistics[feature], rel_delta=0.0001)


@pytest.mark.parametrize(
    "feature",
    [
        (FeatureType.VECTOR, "vector"),
        (FeatureType.VECTOR_TIMELESS, "vector_timeless"),
        {FeatureType.VECTOR_TIMELESS: ["vector_timeless"], FeatureType.META_INFO: ["test_meta"]},
    ],
)
def test_generate_eopatch_fails(feature: FeaturesSpecification) -> None:
    with pytest.raises(ValueError):
        generate_eopatch(feature)


def test_generate_meta_data() -> None:
    patch = generate_eopatch((FeatureType.META_INFO, "test_meta"))
    assert isinstance(patch.meta_info["test_meta"], str)
