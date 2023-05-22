from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import pytest

from eolearn.core import EOPatch, FeatureParser, FeatureType
from eolearn.core.types import EllipsisType, FeatureRenameSpec, FeatureSpec, FeaturesSpecification
from eolearn.core.utils.testing import generate_eopatch


@dataclass
class ParserTestCase:
    parser_input: FeaturesSpecification
    features: List[FeatureSpec]
    renaming: List[FeatureRenameSpec]
    specifications: Optional[List[Tuple[FeatureType, Union[str, EllipsisType]]]] = None
    description: str = ""


def get_test_case_description(test_case: ParserTestCase) -> str:
    return test_case.description


@pytest.mark.parametrize(
    "test_case",
    [
        ParserTestCase(parser_input=[], features=[], renaming=[], specifications=[], description="Empty input"),
        ParserTestCase(
            parser_input=(FeatureType.DATA, "bands"),
            features=[(FeatureType.DATA, "bands")],
            renaming=[(FeatureType.DATA, "bands", "bands")],
            specifications=[(FeatureType.DATA, "bands")],
            description="Singleton feature",
        ),
        ParserTestCase(
            parser_input=FeatureType.BBOX,
            features=[(FeatureType.BBOX, None)],
            renaming=[(FeatureType.BBOX, None, None)],
            specifications=[(FeatureType.BBOX, ...)],
            description="BBox feature",
        ),
        ParserTestCase(
            parser_input=(FeatureType.MASK, "CLM", "new_CLM"),
            features=[(FeatureType.MASK, "CLM")],
            renaming=[(FeatureType.MASK, "CLM", "new_CLM")],
            specifications=[(FeatureType.MASK, "CLM")],
            description="Renamed feature",
        ),
        ParserTestCase(
            parser_input=[FeatureType.BBOX, (FeatureType.DATA, "bands"), (FeatureType.VECTOR_TIMELESS, "geoms")],
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
            parser_input=((FeatureType.TIMESTAMPS, ...), (FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "a", "b")),
            features=[(FeatureType.TIMESTAMPS, None), (FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "a")],
            renaming=[
                (FeatureType.TIMESTAMPS, None, None),
                (FeatureType.MASK, "CLM", "CLM"),
                (FeatureType.SCALAR, "a", "b"),
            ],
            specifications=[(FeatureType.TIMESTAMPS, ...), (FeatureType.MASK, "CLM"), (FeatureType.SCALAR, "a")],
            description="Tuple of inputs with rename",
        ),
        ParserTestCase(
            parser_input={
                FeatureType.DATA: ["bands_S2", ("bands_l8", "BANDS_L8")],
                FeatureType.MASK_TIMELESS: [],
                FeatureType.BBOX: ...,
                FeatureType.TIMESTAMPS: None,
            },
            features=[
                (FeatureType.DATA, "bands_S2"),
                (FeatureType.DATA, "bands_l8"),
                (FeatureType.BBOX, None),
                (FeatureType.TIMESTAMPS, None),
            ],
            renaming=[
                (FeatureType.DATA, "bands_S2", "bands_S2"),
                (FeatureType.DATA, "bands_l8", "BANDS_L8"),
                (FeatureType.BBOX, None, None),
                (FeatureType.TIMESTAMPS, None, None),
            ],
            specifications=[
                (FeatureType.DATA, "bands_S2"),
                (FeatureType.DATA, "bands_l8"),
                (FeatureType.BBOX, ...),
                (FeatureType.TIMESTAMPS, ...),
            ],
            description="Dictionary",
        ),
    ],
    ids=get_test_case_description,
)
def test_feature_parser_no_eopatch(test_case: ParserTestCase):
    """Test that input is parsed according to our expectations. No EOPatch provided."""
    parser = FeatureParser(test_case.parser_input)
    assert parser.get_features() == test_case.features
    assert parser.get_renamed_features() == test_case.renaming
    assert parser.get_feature_specifications() == test_case.specifications


@pytest.mark.parametrize(
    ("test_input", "specifications"),
    [
        ((FeatureType.DATA, ...), [(FeatureType.DATA, ...)]),
        (
            [FeatureType.BBOX, (FeatureType.MASK, "CLM"), FeatureType.DATA],
            [(FeatureType.BBOX, ...), (FeatureType.MASK, "CLM"), (FeatureType.DATA, ...)],
        ),
        (
            {FeatureType.BBOX: None, FeatureType.MASK: ["CLM"], FeatureType.DATA: ...},
            [(FeatureType.BBOX, ...), (FeatureType.MASK, "CLM"), (FeatureType.DATA, ...)],
        ),
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
    ("test_input", "allowed_types"),
    [
        (
            (
                (FeatureType.DATA, "bands", "new_bands"),
                (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
                (FeatureType.MASK, "CLM", "new_CLM"),
            ),
            (FeatureType.MASK,),
        ),
        (
            {
                FeatureType.MASK: ["CLM", "IS_VALID"],
                FeatureType.DATA: [("bands", "new_bands")],
                FeatureType.BBOX: None,
            },
            (
                FeatureType.MASK,
                FeatureType.DATA,
            ),
        ),
    ],
)
def test_allowed_feature_types_iterable(test_input: FeaturesSpecification, allowed_types: Iterable[FeatureType]):
    """Ensure that the parser raises an error if features don't comply with allowed feature types."""
    with pytest.raises(ValueError):
        FeatureParser(features=test_input, allowed_feature_types=allowed_types)


@pytest.fixture(name="eopatch", scope="module")
def eopatch_fixture():
    return generate_eopatch(
        {
            FeatureType.DATA: ["data", "CLP"],
            FeatureType.MASK: ["data", "IS_VALID"],
            FeatureType.MASK_TIMELESS: ["LULC"],
            FeatureType.META_INFO: ["something"],
        }
    )


@pytest.mark.parametrize(
    ("test_input", "allowed_types"),
    [
        (
            (
                (FeatureType.DATA, "bands", "new_bands"),
                (FeatureType.MASK, "IS_VALID", "new_IS_VALID"),
                (FeatureType.MASK, "CLM", "new_CLM"),
            ),
            lambda x: x == FeatureType.MASK,
        ),
        (
            {
                FeatureType.META_INFO: ["something"],
                FeatureType.DATA: [("bands", "new_bands")],
            },
            lambda ftype: not ftype.is_meta(),
        ),
    ],
)
def test_allowed_feature_types_callable(
    test_input: FeaturesSpecification, allowed_types: Callable[[FeatureType], bool]
):
    """Ensure that the parser raises an error if features don't comply with allowed feature types."""
    with pytest.raises(ValueError):
        FeatureParser(features=test_input, allowed_feature_types=allowed_types)


@pytest.mark.parametrize(
    "allowed_types",
    [
        (FeatureType.MASK_TIMELESS, FeatureType.DATA_TIMELESS),
        lambda ftype: ftype.is_timeless() and ftype.ndim() == 3,
    ],
)
def test_all_features_allowed_feature_types(
    eopatch: EOPatch, allowed_types: Union[Iterable[FeatureType], Callable[[FeatureType], bool]]
):
    """Ensure that allowed_feature_types is respected when requesting all features."""
    parser = FeatureParser(..., allowed_feature_types=allowed_types)
    assert parser.get_feature_specifications() == [(FeatureType.DATA_TIMELESS, ...), (FeatureType.MASK_TIMELESS, ...)]
    assert parser.get_features(eopatch) == [(FeatureType.MASK_TIMELESS, "LULC")]
    assert parser.get_renamed_features(eopatch) == [(FeatureType.MASK_TIMELESS, "LULC", "LULC")]


@pytest.mark.parametrize(
    "test_case",
    [
        ParserTestCase(
            parser_input=...,
            features=[
                (FeatureType.BBOX, None),
                (FeatureType.DATA, "data"),
                (FeatureType.DATA, "CLP"),
                (FeatureType.MASK, "data"),
                (FeatureType.MASK, "IS_VALID"),
                (FeatureType.MASK_TIMELESS, "LULC"),
                (FeatureType.META_INFO, "something"),
                (FeatureType.TIMESTAMPS, None),
            ],
            renaming=[
                (FeatureType.BBOX, None, None),
                (FeatureType.DATA, "data", "data"),
                (FeatureType.DATA, "CLP", "CLP"),
                (FeatureType.MASK, "data", "data"),
                (FeatureType.MASK, "IS_VALID", "IS_VALID"),
                (FeatureType.MASK_TIMELESS, "LULC", "LULC"),
                (FeatureType.META_INFO, "something", "something"),
                (FeatureType.TIMESTAMPS, None, None),
            ],
            description="Get-all",
        ),
        ParserTestCase(
            parser_input=(FeatureType.DATA, ...),
            features=[(FeatureType.DATA, "data"), (FeatureType.DATA, "CLP")],
            renaming=[(FeatureType.DATA, "data", "data"), (FeatureType.DATA, "CLP", "CLP")],
            description="Get-all for a feature type",
        ),
        ParserTestCase(
            parser_input=[
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
            parser_input={
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
            parser_input={FeatureType.VECTOR: ...},
            features=[],
            renaming=[],
            description="Request all of an empty feature",
        ),
    ],
    ids=get_test_case_description,
)
def test_feature_parser_with_eopatch(test_case: ParserTestCase, eopatch: EOPatch):
    """Test that input is parsed according to our expectations. EOPatch provided."""
    parser = FeatureParser(test_case.parser_input)
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
