from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureParser, FeatureType
from eolearn.core.utils.parsing import FeatureRenameSpec, FeatureSpec, FeaturesSpecification
from eolearn.core.utils.types import EllipsisType


@dataclass
class TestClass:
    input: FeaturesSpecification
    features: List[FeatureSpec]
    renaming: List[FeatureRenameSpec]
    specifications: List[Tuple[FeatureType, Union[str, EllipsisType]]]


TEST_CASES_EMPTY = [
    TestClass(input=(), features=[], renaming=[], specifications=[]),
]

SPECIAL_CASES = [
    TestClass(
        input=((FeatureType.BBOX, None),),
        features=[(FeatureType.BBOX, None)],
        renaming=[(FeatureType.BBOX, None, None)],
        specifications=[(FeatureType.BBOX, Ellipsis)],
    ),
]


TEST_CASES = [
    TestClass(
        input=((FeatureType.DATA, "bands", "new_bands"),),
        features=[(FeatureType.DATA, "bands")],
        renaming=[(FeatureType.DATA, "bands", "new_bands")],
        specifications=[(FeatureType.DATA, "bands")],
    ),
    TestClass(
        input=([(FeatureType.DATA, "bands"), (FeatureType.MASK, "bands", "new_bands")]),
        features=[(FeatureType.DATA, "bands"), (FeatureType.MASK, "bands")],
        renaming=[(FeatureType.DATA, "bands", "bands"), (FeatureType.MASK, "bands", "new_bands")],
        specifications=[(FeatureType.DATA, "bands"), (FeatureType.MASK, "bands")],
    ),
    TestClass(
        input=(
            {
                FeatureType.DATA: [("IS_VALID", "new_IS_VALID"), ("CLM", "new_CLM")],
                FeatureType.MASK: [("IS_VALID", "new_IS_VALID"), ("CLM", "new_CLM")],
            }
        ),
        features=[
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
        renaming=[
            (FeatureType.DATA, "IS_VALID", "new_IS_VALID"),
            (FeatureType.DATA, "CLM", "new_CLM"),
            (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
            (FeatureType.MASK, "CLM", "new_CLM"),
        ],
        specifications=[
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
    ),
    TestClass(
        input=(
            {
                FeatureType.DATA: [("IS_VALID", "new_IS_VALID"), ("CLM", "new_CLM"), ("BBOX", "BBOX")],
            }
        ),
        features=[
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
        ],
        renaming=[
            (FeatureType.DATA, "IS_VALID", "new_IS_VALID"),
            (FeatureType.DATA, "CLM", "new_CLM"),
            (FeatureType.DATA, "BBOX", "BBOX"),
        ],
        specifications=[
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
        ],
    ),
    TestClass(
        input=[
            (FeatureType.DATA, "IS_VALID", "new_IS_VALID"),
            (FeatureType.DATA, "CLM", "new_CLM"),
            (FeatureType.DATA, "BBOX", "BBOX"),
        ],
        features=[
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
        ],
        renaming=[
            (FeatureType.DATA, "IS_VALID", "new_IS_VALID"),
            (FeatureType.DATA, "CLM", "new_CLM"),
            (FeatureType.DATA, "BBOX", "BBOX"),
        ],
        specifications=[
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
        ],
    ),
]


TEST_CASES_ELLIPSIS = [
    TestClass(
        input=({FeatureType.DATA: ..., FeatureType.MASK: [("bands", "new_bands"), ("CLP", "new_CLP")]}),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "bands"),
            (FeatureType.DATA, "IS_VALID", "IS_VALID"),
            (FeatureType.DATA, "CLP", "CLP"),
            (FeatureType.DATA, "CLM", "CLM"),
            (FeatureType.DATA, "BBOX", "BBOX"),
            (FeatureType.MASK, "bands", "new_bands"),
            (FeatureType.MASK, "CLP", "new_CLP"),
        ],
        specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
    ),
    TestClass(
        input=((FeatureType.DATA, ...), (FeatureType.MASK, "bands", "new_bands"), (FeatureType.MASK, "CLP", "new_CLP")),
        features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "bands"),
            (FeatureType.DATA, "IS_VALID", "IS_VALID"),
            (FeatureType.DATA, "CLP", "CLP"),
            (FeatureType.DATA, "CLM", "CLM"),
            (FeatureType.DATA, "BBOX", "BBOX"),
            (FeatureType.MASK, "bands", "new_bands"),
            (FeatureType.MASK, "CLP", "new_CLP"),
        ],
        specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "IS_VALID"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.DATA, "CLM"),
            (FeatureType.DATA, "BBOX"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
    ),
]


@pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + SPECIAL_CASES + TEST_CASES)
def test_FeatureParser(test_case: TestClass):
    parser = FeatureParser(test_case.input)
    assert parser.get_features() == test_case.features
    assert parser.get_renamed_features() == test_case.renaming
    assert parser.get_feature_specifications() == test_case.specifications


@pytest.mark.parametrize("test_case", TEST_CASES_ELLIPSIS)
def test_FeatureParser_Error(test_case: TestClass):
    """failing by design because ... does not have meaning without EOPatch"""
    parser = FeatureParser(test_case.input)
    with pytest.raises(ValueError):
        parser.get_features()
    with pytest.raises(ValueError):
        parser.get_renamed_features()


@pytest.fixture(name="eopatch")
def eopatch_fixture():
    eopatch = EOPatch()
    eopatch.data["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.data["IS_VALID"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.data["CLP"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.data["CLM"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.data["BBOX"] = np.ones((5, 4, 5, 2))
    eopatch.mask["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["IS_VALID"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["CLP"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["CLM"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    return eopatch


@pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + TEST_CASES + SPECIAL_CASES + TEST_CASES_ELLIPSIS)
def test_FeatureParser_EOPatch(test_case: TestClass, eopatch: EOPatch):
    parser = FeatureParser(test_case.input)
    assert parser.get_features(eopatch) == test_case.features
    assert parser.get_renamed_features(eopatch) == test_case.renaming


@pytest.fixture(name="empty_eopatch")
def empty_eopatch_fixture():
    eopatch = EOPatch()
    eopatch.data["CLP"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    return eopatch


@pytest.mark.parametrize("test_case", TEST_CASES + TEST_CASES_ELLIPSIS)
def test_FeatureParser_EOPatch_Error(test_case: TestClass, empty_eopatch: EOPatch):
    """failing by design because the test_case.input is not subset of EOPatch attributes"""
    parser = FeatureParser(test_case.input)
    with pytest.raises(ValueError):
        parser.get_features(empty_eopatch)
    with pytest.raises(ValueError):
        parser.get_renamed_features(empty_eopatch)
