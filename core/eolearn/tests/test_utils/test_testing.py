import datetime as dt

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sentinelhub import CRS, BBox

from eolearn.core.constants import FeatureType
from eolearn.core.types import FeatureSpec, FeaturesSpecification
from eolearn.core.utils.testing import PatchGeneratorConfig, generate_eopatch


def test_generate_eopatch_set_bbox_timestamps() -> None:
    bbox = BBox((0, 0, 10, 10), crs=CRS("EPSG:32633"))
    timestamps = [dt.datetime(2019, 1, 1)]
    patch = generate_eopatch((FeatureType.DATA, "bands"), bbox=bbox, timestamps=timestamps)

    assert patch.bbox == bbox
    assert patch.timestamps == timestamps

    assert patch[(FeatureType.DATA, "bands")][0] == len(timestamps)


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
    [{}, {FeatureType.DATA: ["bands, CLP"]}, {FeatureType.DATA: ["bands, CLP"], FeatureType.MASK_TIMELESS: "LULC"}],
)
def test_generate_eopatch_seed(seed: int, features: FeaturesSpecification) -> None:
    patch1 = generate_eopatch(features, seed=seed)
    patch2 = generate_eopatch(features, seed=seed)
    assert patch1 == patch2


def test_generate_eopatch_config() -> None:
    test_config = {
        "num_timestamps": 3,
        "timestamps_range": (dt.datetime(2022, 1, 1), dt.datetime(2022, 12, 31)),
        "max_integer_value": 1,
        "raster_shape": (1, 1),
        "depth_range": (1, 2),
    }
    patch = generate_eopatch((FeatureType.MASK, "CLM"), config=PatchGeneratorConfig(**test_config))
    assert_array_equal(patch[(FeatureType.MASK, "CLM")], np.zeros((3, 1, 1, 1)))
