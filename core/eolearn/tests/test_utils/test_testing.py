import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict

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
    "feature",
    [
        (FeatureType.META_INFO, "meta_info"),
        (FeatureType.VECTOR, "vector"),
        (FeatureType.VECTOR_TIMELESS, "vector_timeless"),
    ],
)
def test_generate_eopatch_fails(feature: FeatureSpec) -> None:
    with pytest.raises(ValueError):
        generate_eopatch(feature)


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
        },
    ],
)
def test_generate_eopatch_seed(seed: int, features: FeaturesSpecification) -> None:
    patch1 = generate_eopatch(features, seed=seed)
    patch2 = generate_eopatch(features, seed=seed)
    assert patch1 == patch2


@dataclass
class ConfigTestCase:
    config: PatchGeneratorConfig
    data_features: FeaturesSpecification
    expected_statistics: Dict[FeatureSpec, Dict[str, Any]]


@pytest.mark.parametrize(
    "test_case",
    [
        ConfigTestCase(
            config=PatchGeneratorConfig(
                num_timestamps=3,
                timestamps_range=(dt.datetime(2022, 1, 1), dt.datetime(2022, 12, 31)),
                max_integer_value=1,
                raster_shape=(1, 1),
                depth_range=(1, 2),
            ),
            data_features=(FeatureType.MASK, "CLM"),
            expected_statistics={
                (FeatureType.MASK, "CLM"): {
                    "exp_shape": (3, 1, 1),
                    "exp_min": 0,
                    "exp_max": 0,
                    "exp_mean": 0,
                    "rel_delta": 0.1,
                }
            },
        ),
        ConfigTestCase(
            config=PatchGeneratorConfig(
                max_integer_value=7,
                raster_shape=(3, 4),
                depth_range=(1, 4),
            ),
            data_features={FeatureType.MASK: ["CLM"], FeatureType.MASK_TIMELESS: ["LULC", "IS_VALUE"]},
            expected_statistics={
                (FeatureType.MASK, "CLM"): {
                    "exp_shape": (5, 3, 4),
                    "exp_min": 0,
                    "exp_max": 6,
                    "exp_mean": 3,
                    "abs_delta": 1,
                },
                (FeatureType.MASK_TIMELESS, "LULC"): {
                    "exp_shape": (3, 4),
                    "exp_min": 0,
                    "exp_max": 6,
                    "exp_mean": 3,
                    "abs_delta": 1,
                },
                (FeatureType.MASK_TIMELESS, "IS_VALUE"): {
                    "exp_shape": (3, 4),
                    "exp_min": 0,
                    "exp_max": 6,
                    "exp_mean": 3,
                    "abs_delta": 1,
                },
            },
        ),
    ],
)
def test_generate_eopatch_config(test_case: ConfigTestCase) -> None:
    patch = generate_eopatch(test_case.data_features, config=test_case.config)
    for feature in parse_features(test_case.data_features):
        assert test_case.config.depth_range[0] <= patch[feature].shape[-1] <= test_case.config.depth_range[1]
        test_case.expected_statistics[feature]["exp_shape"] = (
            *test_case.expected_statistics[feature]["exp_shape"],
            patch[feature].shape[-1],
        )

        assert_statistics_match(patch[feature], **test_case.expected_statistics[feature])
