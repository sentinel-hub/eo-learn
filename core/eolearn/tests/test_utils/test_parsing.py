import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureParser, FeatureType
from eolearn.core.utils.parsing import FeatureRenameSpec, FeatureSpec, FeaturesSpecification
from eolearn.core.utils.types import EllipsisType


@dataclass
class TestCase:
    input: FeaturesSpecification
    features: List[FeatureSpec]
    renaming: List[FeatureRenameSpec]
    specifications: Optional[List[Tuple[FeatureType, Union[str, EllipsisType]]]]


TEST_CASES_EMPTY = [
    TestCase(input=(), features=[], renaming=[], specifications=[]),
]

SPECIAL_CASES = [
    TestCase(
        input=((FeatureType.BBOX, None),),
        features=[(FeatureType.BBOX, None)],
        renaming=[(FeatureType.BBOX, None, None)],
        specifications=[(FeatureType.BBOX, Ellipsis)],
    ),
    TestCase(
        input=((FeatureType.TIMESTAMP, None),),
        features=[(FeatureType.TIMESTAMP, None)],
        renaming=[(FeatureType.TIMESTAMP, None, None)],
        specifications=[(FeatureType.TIMESTAMP, Ellipsis)],
    ),
]


TEST_CASES = [
    TestCase(
        input=(
            {
                FeatureType.DATA: (("bands", "new_bands"),),
                FeatureType.MASK: (("bands", "new_bands"),),
            }
        ),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.MASK, "bands"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.MASK, "bands", "new_bands"),
        ],
        specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.MASK, "bands"),
        ],
    ),
    TestCase(
        input=(
            {
                FeatureType.DATA: [("bands", "new_bands"), ("CLP", "new_CLP")],
                FeatureType.MASK: [("IS_VALID", "new_IS_VALID"), ("CLM", "new_CLM")],
            }
        ),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
            (FeatureType.MASK, "CLM", "new_CLM"),
        ],
        specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
    ),
    TestCase(
        input=({FeatureType.DATA: [("bands", "new_bands"), ("CLP", "new_CLP")], FeatureType.BBOX: ...}),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.BBOX, None),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.BBOX, None, None),
        ],
        specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.BBOX, Ellipsis),
        ],
    ),
    TestCase(
        input=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.BBOX, ...),
        ],
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.BBOX, None),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.BBOX, None, None),
        ],
        specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.BBOX, Ellipsis),
        ],
    ),
]


TEST_CASES_ELLIPSIS = [
    TestCase(
        input=(
            {
                FeatureType.DATA: ...,
                FeatureType.BBOX: ...,
                FeatureType.TIMESTAMP: ...,
                FeatureType.MASK: [("IS_VALID", "new_IS_VALID"), ("CLM", "new_CLM")],
            }
        ),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.BBOX, None),
            (FeatureType.TIMESTAMP, None),
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "bands"),
            (FeatureType.DATA, "CLP", "CLP"),
            (FeatureType.BBOX, None, None),
            (FeatureType.TIMESTAMP, None, None),
            (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
            (FeatureType.MASK, "CLM", "new_CLM"),
        ],
        specifications=[],
    ),
    TestCase(
        input=(
            (FeatureType.DATA, ...),
            (FeatureType.BBOX, ...),
            (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
            (FeatureType.MASK, "CLM", "new_CLM"),
        ),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.BBOX, None),
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "bands"),
            (FeatureType.DATA, "CLP", "CLP"),
            (FeatureType.BBOX, None, None),
            (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
            (FeatureType.MASK, "CLM", "new_CLM"),
        ],
        specifications=[],
    ),
]


@pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + SPECIAL_CASES + TEST_CASES)
def test_FeatureParser(test_case: TestCase):
    parser = FeatureParser(test_case.input)
    assert parser.get_features() == test_case.features
    assert parser.get_renamed_features() == test_case.renaming
    assert parser.get_feature_specifications() == test_case.specifications


@pytest.mark.parametrize("test_case", TEST_CASES_ELLIPSIS)
def test_FeatureParser_Error(test_case: TestCase):
    """Test failing when test_case.input contains ... because it has no meaning without EOPatch."""
    parser = FeatureParser(test_case.input)
    with pytest.raises(ValueError):
        parser.get_features()
    with pytest.raises(ValueError):
        parser.get_renamed_features()


@pytest.fixture(name="eopatch")
def eopatch_fixture():
    eopatch = EOPatch()
    eopatch.data["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.data["CLP"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.bbox = BBox((1, 2, 3, 4), CRS.WGS84)
    eopatch.timestamp = [dt.datetime(2020, 5, 1), dt.datetime(2020, 5, 25)]
    eopatch.mask["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["IS_VALID"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["CLM"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    return eopatch


@pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + SPECIAL_CASES + TEST_CASES + TEST_CASES_ELLIPSIS)
def test_FeatureParser_EOPatch(test_case: TestCase, eopatch: EOPatch):
    parser = FeatureParser(test_case.input)
    assert parser.get_features(eopatch) == test_case.features
    assert parser.get_renamed_features(eopatch) == test_case.renaming


@pytest.fixture(name="empty_subset_eopatch")
def empty_eopatch_fixture():
    eopatch = EOPatch()
    eopatch.data["CLP_S2C"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["CLM_S2C"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    return eopatch


@pytest.mark.parametrize("test_case", TEST_CASES + TEST_CASES_ELLIPSIS)
def test_FeatureParser_EOPatch_Error(test_case: TestCase, empty_subset_eopatch: EOPatch):
    """Test failing when the test_case.input is not subset of EOPatch attributes."""
    parser = FeatureParser(test_case.input)
    with pytest.raises(ValueError):
        parser.get_features(empty_subset_eopatch)
    with pytest.raises(ValueError):
        parser.get_renamed_features(empty_subset_eopatch)


def test_FeatureParser_allowed_Error():
    """Test failing when some features are not allowed."""
    with pytest.raises(ValueError):
        FeatureParser(
            features=(
                (FeatureType.DATA, "bands", "new_bands"),
                (FeatureType.MASK, "bands", "new_bands"),
                (FeatureType.MASK, "CLP", "new_CLP"),
            ),
            allowed_feature_types=(FeatureType.MASK,),
        )
