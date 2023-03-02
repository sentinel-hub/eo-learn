import pytest

from eolearn.core.constants import FeatureType
from eolearn.core.types import FeatureSpec
from eolearn.core.utils.testing import patch_generator


@pytest.mark.parametrize(
    "feature",
    [
        (FeatureType.DATA, "neki"),
        (FeatureType.MASK, "neki"),
        (FeatureType.SCALAR, "neki"),
        (FeatureType.LABEL, "neki"),
        (FeatureType.DATA_TIMELESS, "neki"),
        (FeatureType.MASK_TIMELESS, "neki"),
        (FeatureType.SCALAR_TIMELESS, "neki"),
        (FeatureType.LABEL_TIMELESS, "neki"),
    ],
)
def test_patch_generator_dim(feature: FeatureSpec) -> None:
    patch = patch_generator(feature)
    assert patch[feature].ndim == feature[0].ndim()
