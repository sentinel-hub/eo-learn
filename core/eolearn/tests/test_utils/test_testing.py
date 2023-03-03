import datetime as dt

import pytest

from sentinelhub import CRS, BBox

from eolearn.core.constants import FeatureType
from eolearn.core.types import FeatureSpec, FeaturesSpecification
from eolearn.core.utils.testing import generate_eopatch


def test_patch_generator_set_bbox_timestamps() -> None:
    bbox = BBox((0, 0, 10, 10), crs=CRS("EPSG:32633"))
    timestamps = [dt.datetime(2019, 1, 1)]
    patch = generate_eopatch(bbox=bbox, timestamps=timestamps)

    assert patch.bbox == bbox
    assert patch.timestamps == timestamps


@pytest.mark.parametrize(
    "feature",
    [
        (FeatureType.META_INFO, "meta_info"),
        (FeatureType.VECTOR, "vector"),
        (FeatureType.VECTOR_TIMELESS, "vector_timeless"),
    ],
)
def test_patch_generator_fails(feature: FeatureSpec) -> None:
    with pytest.raises(ValueError):
        # fails because it is not `raster` and only raster features are supported
        generate_eopatch(feature)


@pytest.mark.parametrize("seed", [0, 1, 42, 100])
@pytest.mark.parametrize(
    "features",
    [{}, {FeatureType.DATA: ["bands, CLP"]}, {FeatureType.DATA: ["bands, CLP"], FeatureType.MASK_TIMELESS: "LULC"}],
)
def test_patch_generator_seed(seed: int, features: FeaturesSpecification) -> None:
    patch1 = generate_eopatch(features)
    patch2 = generate_eopatch(features)
    assert patch1 == patch2
