from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureParser, FeatureType
from eolearn.core.utils.parsing import FeatureRenameSpec, FeatureSpec, FeaturesSpecification
from eolearn.core.utils.types import EllipsisType


@dataclass
class ParsingTestClass:
    input: FeaturesSpecification
    output_get_features: List[FeatureSpec]
    output_get_renamed_features: List[FeatureRenameSpec]
    output_get_feature_specifications: List[Tuple[FeatureType, Union[str, EllipsisType]]]


TEST_CASES_EMPTY = [
    ParsingTestClass(
        input=(), output_get_features=[], output_get_renamed_features=[], output_get_feature_specifications=[]
    ),
]


TEST_CASES = [
    ParsingTestClass(
        input=((FeatureType.DATA, "bands", "new_bands"),),
        output_get_features=[(FeatureType.DATA, "bands")],
        output_get_renamed_features=[(FeatureType.DATA, "bands", "new_bands")],
        output_get_feature_specifications=[(FeatureType.DATA, "bands")],
    ),
    ParsingTestClass(
        input=([(FeatureType.DATA, "bands"), (FeatureType.MASK, "bands", "new_bands")]),
        output_get_features=[(FeatureType.DATA, "bands"), (FeatureType.MASK, "bands")],
        output_get_renamed_features=[(FeatureType.DATA, "bands", "bands"), (FeatureType.MASK, "bands", "new_bands")],
        output_get_feature_specifications=[(FeatureType.DATA, "bands"), (FeatureType.MASK, "bands")],
    ),
    ParsingTestClass(
        input=(
            {
                FeatureType.DATA: [("bands", "new_bands"), ("CLP", "new_CLP")],
                FeatureType.MASK: [("bands", "new_bands"), ("CLP", "new_CLP")],
            }
        ),
        output_get_features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
        output_get_renamed_features=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.MASK, "bands", "new_bands"),
            (FeatureType.MASK, "CLP", "new_CLP"),
        ],
        output_get_feature_specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
    ),
]


TEST_CASES_ELLIPSIS = [
    ParsingTestClass(
        input=({FeatureType.DATA: ..., FeatureType.MASK: [("bands", "new_bands"), ("CLP", "new_CLP")]}),
        output_get_features=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
        output_get_renamed_features=[
            (FeatureType.DATA, "bands", "bands"),
            (FeatureType.DATA, "CLP", "CLP"),
            (FeatureType.MASK, "bands", "new_bands"),
            (FeatureType.MASK, "CLP", "new_CLP"),
        ],
        output_get_feature_specifications=[
            (FeatureType.DATA, "bands"),
            (FeatureType.DATA, "CLP"),
            (FeatureType.MASK, "bands"),
            (FeatureType.MASK, "CLP"),
        ],
    ),
]


@pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + TEST_CASES)
def test_FeatureParser(test_case: ParsingTestClass):
    parser = FeatureParser(test_case.input)
    assert parser.get_features() == test_case.output_get_features
    assert parser.get_renamed_features() == test_case.output_get_renamed_features
    assert parser.get_feature_specifications() == test_case.output_get_feature_specifications


@pytest.mark.parametrize("test_case", TEST_CASES_ELLIPSIS)
def test_FeatureParser_Error(test_case: ParsingTestClass):
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
    eopatch.mask["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["CLP"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    return eopatch


@pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + TEST_CASES + TEST_CASES_ELLIPSIS)
def test_FeatureParser_EOPatch(test_case: ParsingTestClass, eopatch: EOPatch):
    parser = FeatureParser(test_case.input)
    assert parser.get_features(eopatch) == test_case.output_get_features
    assert parser.get_renamed_features(eopatch) == test_case.output_get_renamed_features


@pytest.fixture(name="empty_eopatch")
def empty_eopatch_fixture():
    eopatch = EOPatch()
    eopatch.data["CLP"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    eopatch.mask["bands"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    return eopatch


@pytest.mark.parametrize("test_case", TEST_CASES + TEST_CASES_ELLIPSIS)
def test_FeatureParser_EOPatch_Error(test_case: ParsingTestClass, empty_eopatch: EOPatch):
    parser = FeatureParser(test_case.input)
    with pytest.raises(ValueError):
        parser.get_features(empty_eopatch)
    with pytest.raises(ValueError):
        parser.get_renamed_features(empty_eopatch)
