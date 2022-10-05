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
    description: str = ""


def get_test_case_description(test_case: TestCase) -> str:
    return test_case.description


BANDS_FEATURE = (FeatureType.DATA, "bands")
CLP_FEATURE = (FeatureType.DATA, "CLP")
CLM_FEATURE = (FeatureType.MASK, "CLM")
IS_VALID_FEATURE = (FeatureType.MASK, "IS_VALID")
LULC_FEATURE = (FeatureType.MASK_TIMELESS, "LULC")
GEOMS_FEATURE = (FeatureType.VECTOR_TIMELESS, "geoms")


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
        input=((FeatureType.TIMESTAMP, ...),),
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
            BANDS_FEATURE,
            (FeatureType.MASK, "bands"),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.MASK, "bands", "new_bands"),
        ],
        specifications=[
            BANDS_FEATURE,
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
            BANDS_FEATURE,
            CLP_FEATURE,
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
            BANDS_FEATURE,
            CLP_FEATURE,
            (FeatureType.MASK, "IS_VALID"),
            (FeatureType.MASK, "CLM"),
        ],
    ),
    TestCase(
        input=({FeatureType.DATA: [("bands", "new_bands"), ("CLP", "new_CLP")], FeatureType.BBOX: ...}),
        features=[
            BANDS_FEATURE,
            CLP_FEATURE,
            (FeatureType.BBOX, None),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.BBOX, None, None),
        ],
        specifications=[
            BANDS_FEATURE,
            CLP_FEATURE,
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
            BANDS_FEATURE,
            CLP_FEATURE,
            (FeatureType.BBOX, None),
        ],
        renaming=[
            (FeatureType.DATA, "bands", "new_bands"),
            (FeatureType.DATA, "CLP", "new_CLP"),
            (FeatureType.BBOX, None, None),
        ],
        specifications=[
            BANDS_FEATURE,
            CLP_FEATURE,
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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(input=[], features=[], renaming=[], specifications=[], description="Empty input"),
        TestCase(
            input=BANDS_FEATURE,
            features=[BANDS_FEATURE],
            renaming=[(FeatureType.DATA, "bands", "bands")],
            specifications=[BANDS_FEATURE],
            description="Singleton feature",
        ),
        TestCase(
            input=FeatureType.BBOX,
            features=[(FeatureType.BBOX, None)],
            renaming=[(FeatureType.BBOX, None, None)],
            specifications=[(FeatureType.BBOX, ...)],
            description="Parsing BBox feature",
        ),
        TestCase(
            input=(FeatureType.MASK, "CLM", "new_CLM"),
            features=[(FeatureType.MASK, "CLM")],
            renaming=[(FeatureType.MASK, "CLM", "new_CLM")],
            specifications=[(FeatureType.MASK, "CLM")],
            description="Parsing renamed feature",
        ),
        TestCase(
            input=[FeatureType.BBOX, BANDS_FEATURE, GEOMS_FEATURE],
            features=[(FeatureType.BBOX, None), BANDS_FEATURE, GEOMS_FEATURE],
            renaming=[
                (FeatureType.BBOX, None, None),
                (FeatureType.DATA, "bands", "bands"),
                (FeatureType.VECTOR_TIMELESS, "geoms", "geoms"),
            ],
            specifications=[(FeatureType.BBOX, ...), BANDS_FEATURE, GEOMS_FEATURE],
            description="Parsing a list of inputs",
        ),
        TestCase(
            input=((FeatureType.TIMESTAMP, ...), CLM_FEATURE, (FeatureType.SCALAR, "a", "b")),
            features=[(FeatureType.TIMESTAMP, None), CLM_FEATURE, (FeatureType.SCALAR, "a")],
            renaming=[
                (FeatureType.TIMESTAMP, None, None),
                (FeatureType.MASK, "CLM", "CLM"),
                (FeatureType.SCALAR, "a", "b"),
            ],
            specifications=[(FeatureType.TIMESTAMP, ...), CLM_FEATURE, (FeatureType.SCALAR, "a")],
            description="Parsing a tuple of inputs with rename",
        ),
        TestCase(
            input={
                FeatureType.DATA: ["bands_S2", ("bands_l8", "BANDS_L8")],
                FeatureType.MASK_TIMELESS: [],
                FeatureType.BBOX: ...,
                FeatureType.TIMESTAMP: None,
            },
            features=[
                (FeatureType.DATA, "bands_S2"),
                (FeatureType.DATA, "bands_l8"),
                (FeatureType.BBOX, None),
                (FeatureType.TIMESTAMP, None),
            ],
            renaming=[
                (FeatureType.DATA, "bands_S2", "bands_S2"),
                (FeatureType.DATA, "bands_l8", "BANDS_L8"),
                (FeatureType.BBOX, None, None),
                (FeatureType.TIMESTAMP, None, None),
            ],
            specifications=[
                (FeatureType.DATA, "bands_S2"),
                (FeatureType.DATA, "bands_l8"),
                (FeatureType.BBOX, ...),
                (FeatureType.TIMESTAMP, ...),
            ],
            description="Parsing dictionary",
        ),
    ],
    ids=get_test_case_description,
)
def test_feature_parser_no_eopatch(test_case: TestCase):
    """Test that input is parsed according to our expectations. No EOPatch provided."""
    parser = FeatureParser(test_case.input)
    assert parser.get_features() == test_case.features
    assert parser.get_renamed_features() == test_case.renaming
    assert parser.get_feature_specifications() == test_case.specifications


@pytest.mark.parametrize(
    "test_input, specifications",
    [
        [(FeatureType.DATA, ...), [(FeatureType.DATA, ...)]],
        [
            [FeatureType.BBOX, (FeatureType.MASK, "CLM"), FeatureType.DATA],
            [(FeatureType.BBOX, ...), (FeatureType.MASK, "CLM"), (FeatureType.DATA, ...)],
        ],
        [
            {FeatureType.BBOX: None, FeatureType.MASK: ["CLM"], FeatureType.DATA: ...},
            [(FeatureType.BBOX, ...), (FeatureType.MASK, "CLM"), (FeatureType.DATA, ...)],
        ],
    ],
)
def test_feature_parser_no_eopatch_failure(
    test_input: FeaturesSpecification, specifications: List[Tuple[FeatureType, Union[str, EllipsisType]]]
):
    """When a get-all request `...` without an eopatch the parser should fail unless parsing specifications."""
    parser = FeatureParser(test_input)
    with pytest.raises(ValueError):
        parser.get_features()
    with pytest.raises(ValueError):
        parser.get_renamed_features()
    assert parser.get_feature_specifications() == specifications


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


# @pytest.mark.parametrize("test_case", TEST_CASES_EMPTY + SPECIAL_CASES + TEST_CASES + TEST_CASES_ELLIPSIS)
# def test_FeatureParser_EOPatch(test_case: TestCase, eopatch: EOPatch):
#     """Test that input is parsed according to our expectations. EOPatch provided."""
#     parser = FeatureParser(test_case.input)
#     assert parser.get_features(eopatch) == test_case.features
#     assert parser.get_renamed_features(eopatch) == test_case.renaming


# @pytest.fixture(name="empty_intersection_eopatch")
# def empty_eopatch_fixture():
#     eopatch = EOPatch()
#     eopatch.data["CLP_S2C"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
#     eopatch.mask["CLM_S2C"] = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
#     return eopatch


# @pytest.mark.parametrize("test_case", TEST_CASES + TEST_CASES_ELLIPSIS)
# def test_FeatureParser_EOPatch_Error(test_case: TestCase, empty_intersection_eopatch: EOPatch):
#     """Test failing when the test_case.input is not subset of EOPatch attributes."""
#     parser = FeatureParser(test_case.input)
#     with pytest.raises(ValueError):
#         parser.get_features(empty_intersection_eopatch)
#     with pytest.raises(ValueError):
#         parser.get_renamed_features(empty_intersection_eopatch)


# def test_FeatureParser_allowed_Error():
#     """Test failing when some features are not allowed."""
#     with pytest.raises(ValueError):
#         FeatureParser(
#             features=(
#                 (FeatureType.DATA, "bands", "new_bands"),
#                 (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
#                 (FeatureType.MASK, "CLM", "new_CLM"),
#             ),
#             allowed_feature_types=(FeatureType.MASK,),
#         )
#     with pytest.raises(ValueError):
#         FeatureParser(
#             features=(
#                 (FeatureType.BBOX, ...),
#                 (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
#                 (FeatureType.MASK, "CLM", "new_CLM"),
#             ),
#             allowed_feature_types=(FeatureType.MASK,),
#         )


# def test_pars_dict_Error():
#     """Test failing when values of dictionary are not `...` or sequences with feature names."""
#     with pytest.raises(ValueError):
#         FeatureParser(
#             features={
#                 FeatureType.DATA: {"bands": "new_bands"},
#                 FeatureType.MASK: {"IS_VALID": "new_IS_VALID", "CLM": "new_CLM"},
#             }
#         )


# def test_get_renamed_features_eopatch_Error(empty_intersection_eopatch: EOPatch):
#     """Test failing when input of feature parser is not parameter of `eopatch`."""
#     parser = FeatureParser(
#         (FeatureType.DATA, "bands", "new_bands"),
#     )
#     with pytest.raises(ValueError):
#         parser.get_renamed_features(empty_intersection_eopatch)
