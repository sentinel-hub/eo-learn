import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pytest

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureParser, FeatureType
from eolearn.core.types import EllipsisType, FeatureRenameSpec, FeatureSpec, FeaturesSpecification


@dataclass
class ParserTestCase:
    input: FeaturesSpecification
    features: List[FeatureSpec]
    renaming: List[FeatureRenameSpec]
    specifications: Optional[List[Tuple[FeatureType, Union[str, EllipsisType]]]] = None
    description: str = ""


def get_test_case_description(test_case: ParserTestCase) -> str:
    return test_case.description


@pytest.mark.parametrize(
    "test_case",
    [
        ParserTestCase(input=[], features=[], renaming=[], specifications=[], description="Empty input"),
        ParserTestCase(
            input=(FeatureType.DATA, "bands"),
            features=[(FeatureType.DATA, "bands")],
            renaming=[(FeatureType.DATA, "bands", "bands")],
            specifications=[(FeatureType.DATA, "bands")],
            description="Singleton feature",
        ),
        ParserTestCase(
            input=FeatureType.BBOX,
            features=[(FeatureType.BBOX, None)],
            renaming=[(FeatureType.BBOX, None, None)],
            specifications=[(FeatureType.BBOX, ...)],
            description="BBox feature",
        ),
        ParserTestCase(
            input=(FeatureType.MASK, "CLM", "new_CLM"),
            features=[(FeatureType.MASK, "CLM")],
            renaming=[(FeatureType.MASK, "CLM", "new_CLM")],
            specifications=[(FeatureType.MASK, "CLM")],
            description="Renamed feature",
        ),
        ParserTestCase(
            input=[FeatureType.BBOX, (FeatureType.DATA, "bands"), (FeatureType.VECTOR_TIMELESS, "geoms")],
            features=[(FeatureType.BBOX, None), (FeatureType.DATA, "bands"), (FeatureType.VECTOR_TIMELESS, "geoms")],
            renaming=[
                (FeatureType.BBOX, None, None),
                (FeatureType.DATA, "bands", "bands"),
                (FeatureType.VECTOR_TIMELESS, "geoms", "geoms"),
            ],
            specifications=[
                (FeatureType.BBOX, ...),
                (FeatureType.DATA, "bands"),
                (FeatureType.VECTOR_TIMELESS, "geoms"),
            ],
            description="List of inputs",
        ),
        ParserTestCase(
            input=((FeatureType.TIMESTAMP, ...), (FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "a", "b")),
            features=[(FeatureType.TIMESTAMP, None), (FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "a")],
            renaming=[
                (FeatureType.TIMESTAMP, None, None),
                (FeatureType.MASK, "CLM", "CLM"),
                (FeatureType.SCALAR, "a", "b"),
            ],
            specifications=[(FeatureType.TIMESTAMP, ...), (FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "a")],
            description="Tuple of inputs with rename",
        ),
        ParserTestCase(
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
            description="Dictionary",
        ),
    ],
    ids=get_test_case_description,
)
def test_feature_parser_no_eopatch(test_case: ParserTestCase):
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


@pytest.mark.parametrize(
    "test_input, allowed_types",
    [
        [
            (
                (FeatureType.DATA, "bands", "new_bands"),
                (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
                (FeatureType.MASK, "CLM", "new_CLM"),
            ),
            (FeatureType.MASK,),
        ],
        [
            {
                FeatureType.MASK: ["CLM", "IS_VALID"],
                FeatureType.DATA: [("bands", "new_bands")],
                FeatureType.BBOX: None,
            },
            (
                FeatureType.MASK,
                FeatureType.DATA,
            ),
        ],
    ],
)
def test_allowed_feature_types(test_input: FeaturesSpecification, allowed_types: Iterable[FeatureType]):
    """Ensure that the parser raises an error if features don't comply with allowed feature types."""
    with pytest.raises(ValueError):
        FeatureParser(features=test_input, allowed_feature_types=allowed_types)


@pytest.fixture(name="eopatch", scope="module")
def eopatch_fixture():
    return EOPatch(
        data=dict(data=np.zeros((2, 2, 2, 2)), CLP=np.zeros((2, 2, 2, 2))),  # name duplication intentional
        bbox=BBox((1, 2, 3, 4), CRS.WGS84),
        timestamps=[dt.datetime(2020, 5, 1), dt.datetime(2020, 5, 25)],
        mask=dict(data=np.zeros((2, 2, 2, 2), dtype=int), IS_VALID=np.zeros((2, 2, 2, 2), dtype=int)),
        mask_timeless=dict(LULC=np.zeros((2, 2, 2), dtype=int)),
        meta_info={"something": "else"},
    )


@pytest.mark.parametrize(
    "test_case",
    [
        ParserTestCase(
            input=...,
            features=[
                (FeatureType.BBOX, None),
                (FeatureType.DATA, "data"),
                (FeatureType.DATA, "CLP"),
                (FeatureType.MASK, "data"),
                (FeatureType.MASK, "IS_VALID"),
                (FeatureType.MASK_TIMELESS, "LULC"),
                (FeatureType.META_INFO, "something"),
                (FeatureType.TIMESTAMP, None),
            ],
            renaming=[
                (FeatureType.BBOX, None, None),
                (FeatureType.DATA, "data", "data"),
                (FeatureType.DATA, "CLP", "CLP"),
                (FeatureType.MASK, "data", "data"),
                (FeatureType.MASK, "IS_VALID", "IS_VALID"),
                (FeatureType.MASK_TIMELESS, "LULC", "LULC"),
                (FeatureType.META_INFO, "something", "something"),
                (FeatureType.TIMESTAMP, None, None),
            ],
            description="Get-all",
        ),
        ParserTestCase(
            input=(FeatureType.DATA, ...),
            features=[(FeatureType.DATA, "data"), (FeatureType.DATA, "CLP")],
            renaming=[(FeatureType.DATA, "data", "data"), (FeatureType.DATA, "CLP", "CLP")],
            description="Get-all for a feature type",
        ),
        ParserTestCase(
            input=[
                FeatureType.BBOX,
                FeatureType.MASK,
                (FeatureType.META_INFO, ...),
                (FeatureType.MASK_TIMELESS, "LULC", "new_LULC"),
            ],
            features=[
                (FeatureType.BBOX, None),
                (FeatureType.MASK, "data"),
                (FeatureType.MASK, "IS_VALID"),
                (FeatureType.META_INFO, "something"),
                (FeatureType.MASK_TIMELESS, "LULC"),
            ],
            renaming=[
                (FeatureType.BBOX, None, None),
                (FeatureType.MASK, "data", "data"),
                (FeatureType.MASK, "IS_VALID", "IS_VALID"),
                (FeatureType.META_INFO, "something", "something"),
                (FeatureType.MASK_TIMELESS, "LULC", "new_LULC"),
            ],
            description="Sequence with ellipsis",
        ),
        ParserTestCase(
            input={
                FeatureType.DATA: ["data", ("CLP", "new_CLP")],
                FeatureType.MASK_TIMELESS: ...,
            },
            features=[(FeatureType.DATA, "data"), (FeatureType.DATA, "CLP"), (FeatureType.MASK_TIMELESS, "LULC")],
            renaming=[
                (FeatureType.DATA, "data", "data"),
                (FeatureType.DATA, "CLP", "new_CLP"),
                (FeatureType.MASK_TIMELESS, "LULC", "LULC"),
            ],
            description="Dictionary with ellipsis",
        ),
        ParserTestCase(
            input={FeatureType.VECTOR: ...}, features=[], renaming=[], description="Request all of an empty feature"
        ),
    ],
    ids=get_test_case_description,
)
def test_feature_parser_with_eopatch(test_case: ParserTestCase, eopatch: EOPatch):
    """Test that input is parsed according to our expectations. EOPatch provided."""
    parser = FeatureParser(test_case.input)
    assert parser.get_features(eopatch) == test_case.features, f"{parser.get_features(eopatch)}"
    assert parser.get_renamed_features(eopatch) == test_case.renaming


@pytest.mark.parametrize(
    "test_input",
    [
        (FeatureType.VECTOR, "geoms"),
        {FeatureType.DATA: ["data"], FeatureType.MASK: ["bands_l8"]},
        (FeatureType.MASK, (FeatureType.SCALAR, "something", "else")),
    ],
)
def test_feature_parser_with_eopatch_failure(test_input: FeaturesSpecification, eopatch: EOPatch):
    """These cases should fail because the request feature is not part of the EOPatch."""
    parser = FeatureParser(test_input)
    with pytest.raises(ValueError):
        parser.get_features(eopatch)
    with pytest.raises(ValueError):
        parser.get_renamed_features(eopatch)


def test_all_features_allowed_feature_types(eopatch: EOPatch):
    """Ensure that allowed_feature_types is respected when requesting all features."""
    parser = FeatureParser(..., allowed_feature_types=(FeatureType.DATA, FeatureType.BBOX))
    assert parser.get_feature_specifications() == [(FeatureType.BBOX, ...), (FeatureType.DATA, ...)]
    assert parser.get_features(eopatch) == [
        (FeatureType.BBOX, None),
        (FeatureType.DATA, "data"),
        (FeatureType.DATA, "CLP"),
    ]
    assert parser.get_renamed_features(eopatch) == [
        (FeatureType.BBOX, None, None),
        (FeatureType.DATA, "data", "data"),
        (FeatureType.DATA, "CLP", "CLP"),
    ]
