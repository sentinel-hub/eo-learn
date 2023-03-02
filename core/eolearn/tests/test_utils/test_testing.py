import datetime as dt

import pytest

from sentinelhub import CRS, BBox

from eolearn.core.constants import FeatureType
from eolearn.core.types import FeatureSpec
from eolearn.core.utils.testing import patch_generator


def test_patch_generator_default() -> None:
    patch = patch_generator()
    assert hasattr(patch, "bbox")
    assert hasattr(patch, "timestamps")


def test_patch_generator_not_default() -> None:
    bbox = BBox((0, 0, 10, 10), crs=CRS("EPSG:32633"))
    timestamps = [dt.datetime(2019, 1, 1)]
    patch = patch_generator(bbox=bbox, timestamps=timestamps)

    assert patch.bbox == bbox
    assert patch.timestamps == timestamps


@pytest.mark.parametrize(
    "feature",
    [
        (FeatureType.DATA, "data"),
        (FeatureType.MASK, "mask"),
        (FeatureType.SCALAR, "scalar"),
        (FeatureType.LABEL, "label"),
        (FeatureType.DATA_TIMELESS, "data_timeless"),
        (FeatureType.MASK_TIMELESS, "mask_timeless"),
        (FeatureType.SCALAR_TIMELESS, "scalar_timeless"),
        (FeatureType.LABEL_TIMELESS, "label_timeless"),
    ],
)
def test_patch_generator_dim(feature: FeatureSpec) -> None:
    patch = patch_generator(feature)
    assert patch[feature].ndim == feature[0].ndim()
