"""
A module implementing the FeatureParser class that simplifies specifying features.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, Union, cast

from ..constants import FeatureType
from ..types import (
    DictFeatureSpec,
    EllipsisType,
    FeatureRenameSpec,
    FeatureSpec,
    FeaturesSpecification,
    SequenceFeatureSpec,
    SingleFeatureSpec,
)

if TYPE_CHECKING:
    from ..eodata import EOPatch

_ParserFeaturesSpec = Union[Tuple[FeatureType, None, None], Tuple[FeatureType, str, str]]


class FeatureParser:
    """Class for parsing a variety of feature specifications into a streamlined format.

    This class takes care of parsing multiple inputs that specify features and includes some additional options:

    - Fix allowed types, which raises an appropriate exception if a forbidden type is detected.
    - Parsing directly or parsing over an EOPatch. If an EOPatch object is provided the parser fails if a specified
      feature is missing from the EOPatch. Because EOPatch objects are usually provided to EOTasks at runtime, the
      parser preprocesses and validates the input feature specifications at initialization to ensure exceptions at an
      appropriate point.
    - The user can provide ellipsis `...` as a way to specify all features. When combined with a feature type it is
      understood as all features of a given type. When using `...` an EOPatch must be provided when parsing features,
      except when used only for BBox and timestamp features.
    - The parser can output pairs `(feature_type, feature_name)` or triples `(feature_type, old_name, new_name)`, which
      come in hand in many cases. If the user does not provide an explicit new name, the `old_name` and `new_name` are
      equal.

    The main input formats are as follows:

    1. Ellipsis `...` signify that all features of all types should be parsed.

    2. Input representing a single feature, either `FeatureType.BBOX`, `FeatureType.TIMESTAMP` or a tuple where the
       first element is a `FeatureType` element and the other (or two for renaming) is a string.

    3. Dictionary mapping `feature_type` keys to sequences of feature names. Feature names are either a sequence of
       strings, pairs of shape `(old_name, new_name)` or `...`. For feature types with no names (BBox and timestamps)
       one should use `None` in place of the sequence. Example:

        .. code-block:: python

            {
                FeatureType.DATA: [
                    ('S2-BANDS', 'INTERPOLATED_S2_BANDS'),
                    ('L8-BANDS', 'INTERPOLATED_L8_BANDS'),
                    'NDVI',
                    'NDWI',
                },
                FeatureType.MASK: ...
                FeatureType.BBOX: None
            }

    4. Sequences of elements, each describing a feature. When describing all features of a given feature type use
       `(feature_type, ...)`. For specific features one can use `(feature_type, feature_name)` or even
       `(feature_type, old_name, new_name)` for renaming.

        .. code-block:: python

            [
                (FeatureType.DATA, 'BANDS'),
                (FeatureType.MASK, 'CLOUD_MASK', 'NEW_CLOUD_MASK'),
                FeatureType.BBOX
            ]

    Outputs of the FeatureParser are:

    - For `get_features` a list of pairs `(feature_type, feature_name)`.
    - For `get_renamed_features` a list of triples `(feature_type, old_name, new_name)`.
    """

    def __init__(
        self,
        features: FeaturesSpecification,
        allowed_feature_types: Union[Iterable[FeatureType], EllipsisType] = ...,
    ):
        """
        :param features: A collection of features in one of the supported formats
        :param allowed_feature_types: Makes sure that only features of these feature types will be returned, otherwise
            an error is raised
        :raises: ValueError
        """
        self.allowed_feature_types = (
            set(allowed_feature_types) if isinstance(allowed_feature_types, Iterable) else set(FeatureType)
        )
        self._feature_specs = self._parse_features(features)

    def _parse_features(self, features: FeaturesSpecification) -> List[_ParserFeaturesSpec]:
        """This method parses and validates input, returning a list of `(ftype, old_name, new_name)` triples.

        Due to typing issues the all-features requests are transformed from `(ftype, ...)` to `(ftype, None, None)`.
        This is a correct schema for BBOX and TIMESTAMP while for other features this is corrected when outputting,
        either by processing the request or by substituting ellipses back (case of `get_feature_specifications`).
        """

        if isinstance(features, FeatureType):
            return [(features, None, None)]

        if isinstance(features, dict):
            return self._parse_dict(features)

        if isinstance(features, Sequence):
            return self._parse_sequence(features)

        if features is ...:
            # we sort allowed_feature_types to keep behaviour deterministic
            ftypes = sorted(self.allowed_feature_types, key=lambda ftype: ftype.value)
            return [(ftype, None, None) for ftype in ftypes]

        raise ValueError(
            f"Unable to parse features {features}. Please see specifications of FeatureParser on viable inputs."
        )

    def _parse_dict(
        self,
        features: DictFeatureSpec,
    ) -> List[_ParserFeaturesSpec]:
        """Implements parsing and validation in case the input is a dictionary."""

        feature_specs: List[_ParserFeaturesSpec] = []

        for feature_type, feature_names in features.items():
            feature_type = self._parse_feature_type(feature_type, message_about_position="keys of the dictionary")

            if feature_names in (..., None):
                feature_specs.append((feature_type, None, None))
                continue

            self._fail_for_noname_features(feature_type, feature_names)

            if not isinstance(feature_names, Sequence):
                raise ValueError("Values of dictionary must be `...` or sequences with feature names.")

            parsed_names = [(feature_type, *self._parse_feature_name(feature_type, name)) for name in feature_names]
            feature_specs.extend(parsed_names)

        return feature_specs

    def _parse_sequence(
        self,
        features: Union[SingleFeatureSpec, SequenceFeatureSpec],
    ) -> List[_ParserFeaturesSpec]:
        """Implements parsing and validation in case the input is a tuple describing a single feature or a sequence."""

        feature_specs: List[_ParserFeaturesSpec] = []

        # Check for possible singleton
        if 2 <= len(features) <= 3:
            with contextlib.suppress(ValueError):
                return [(self._parse_singleton(features))]

        for feature in features:
            if isinstance(feature, (tuple, list)) and 2 <= len(feature) <= 3:
                feature_specs.append(self._parse_singleton(feature))

            elif isinstance(feature, FeatureType):
                feature_type = self._parse_feature_type(feature, message_about_position="singleton elements")
                feature_specs.append((feature_type, None, None))

            else:
                raise ValueError(
                    f"Failed to parse {feature}, expected a tuple of form `(feature_type, feature_name)` or "
                    "`(feature_type, old_name, new_name)`."
                )

        return feature_specs

    def _parse_singleton(self, feature: Sequence) -> FeatureRenameSpec:
        """Parses a pair or triple specifying a single feature or a get-all request."""
        feature_type, *feature_name = feature
        feature_type = self._parse_feature_type(feature_type, message_about_position="first elements of tuples")

        if len(feature_name) == 1 and feature_name[0] in (..., None):
            return (feature_type, None, None)

        self._fail_for_noname_features(feature_type, feature_name)

        feature_name = feature_name[0] if len(feature_name) == 1 else feature_name
        parsed_name = self._parse_feature_name(feature_type, feature_name)
        return (feature_type, *parsed_name)

    def _parse_feature_type(self, feature_type: Union[str, FeatureType], *, message_about_position: str) -> FeatureType:
        """Tries to extract a feature type if possible, fails otherwise.

        The parameter `message_about_position` is used for more informative error messages.
        """
        try:
            feature_type = FeatureType(feature_type)
        except ValueError as exception:
            raise ValueError(
                f"Failed to parse {feature_type}, {message_about_position} must be {FeatureType.__name__}"
            ) from exception

        if feature_type not in self.allowed_feature_types:
            raise ValueError(
                f"Allowed feature types were set to be {self.allowed_feature_types} but found {feature_type}"
            )
        return feature_type

    @staticmethod
    def _parse_feature_name(feature_type: FeatureType, name: object) -> Tuple[str, str]:
        """Parses input in places where a feature name is expected, handling the cases of a name and renaming pair."""
        if isinstance(name, str):
            return name, name
        if isinstance(name, (tuple, list)):
            if len(name) != 2 or not all(isinstance(n, str) for n in name):
                raise ValueError(
                    "When specifying a re-name for a feature it must be a pair of strings `(old_name, new_name)`, "
                    f"got {name}."
                )
            return cast(Tuple[str, str], tuple(name))
        raise ValueError(
            f"For {feature_type} found invalid feature name {name}. The sequence of feature names can contain only"
            " strings or pairs of form `(old_name, new_name)`"
        )

    @staticmethod
    def _fail_for_noname_features(feature_type: FeatureType, specification: object) -> None:
        """Fails if the feature type does not support names.

        Should only be used after the viable names `...` and `None` have already been handled.
        """
        if not feature_type.has_dict():
            raise ValueError(
                f"For features of type {feature_type} the only acceptable specification is `...` or `None`, got"
                f" {specification} instead."
            )

    def get_feature_specifications(self) -> List[Tuple[FeatureType, Union[str, EllipsisType]]]:
        """Returns the feature specifications in a more streamlined fashion.

        Requests for all features, e.g. `(FeatureType.DATA, ...)`, are returned directly.
        """
        return [(ftype, ... if fname is None else fname) for ftype, fname, _ in self._feature_specs]

    def get_features(self, eopatch: Optional[EOPatch] = None) -> List[FeatureSpec]:
        """Returns a list of `(feature_type, feature_name)` pairs.

        For features that specify renaming, the new name of the feature is ignored.

        If `eopatch` is provided, the method checks that the EOPatch contains all the specified data and processes
        requests for all features, e.g. `(FeatureType.DATA, ...)`, by listing all available features of given type.

        If `eopatch` is not provided the method fails if an all-feature request is in the specification.
        """
        renamed_features = self.get_renamed_features(eopatch)
        return [feature[:2] for feature in renamed_features]  # pattern unpacking messes with typechecking

    def get_renamed_features(self, eopatch: Optional[EOPatch] = None) -> List[FeatureRenameSpec]:
        """Returns a list of `(feature_type, old_name, new_name)` triples.

        For features without a specified renaming the new name is equal to the old one.

        If `eopatch` is provided, the method checks that the EOPatch contains all the specified data and processes
        requests for all features, e.g. `(FeatureType.DATA, ...)`, by listing all available features of given type.
        In these cases the returned `old_name` and `new_name` are equal.

        If `eopatch` is not provided the method fails if an all-feature request is in the specification.
        """
        parsed_features: List[FeatureRenameSpec] = []
        for feature_spec in self._feature_specs:
            ftype, old_name, new_name = feature_spec

            if ftype is FeatureType.BBOX or ftype is FeatureType.TIMESTAMP:
                parsed_features.append((ftype, None, None))

            elif old_name is not None and new_name is not None:
                # checking both is redundant, but typechecker has difficulties otherwise
                if eopatch is not None and (ftype, old_name) not in eopatch:
                    raise ValueError(f"Requested feature {(ftype, old_name)} not part of eopatch.")
                parsed_features.append((ftype, old_name, new_name))

            elif eopatch is not None:
                parsed_features.extend((ftype, name, name) for name in eopatch[ftype])
            else:
                raise ValueError(
                    f"Input of feature parser specifies that for feature type {ftype} all existing features are parsed,"
                    " but the `eopatch` parameter was not provided."
                )
        return parsed_features


def parse_feature(
    feature: SingleFeatureSpec,
    eopatch: Optional[EOPatch] = None,
    allowed_feature_types: Union[Iterable[FeatureType], EllipsisType] = ...,
) -> Tuple[FeatureType, Optional[str]]:
    """Parses input describing a single feature into a `(feature_type, feature_name)` pair.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """

    features = FeatureParser([feature], allowed_feature_types=allowed_feature_types).get_features(eopatch)
    if len(features) != 1:
        raise ValueError(f"Specification {feature} resulted in {len(features)} features, expected 1.")
    return features[0]


def parse_renamed_feature(
    feature: SingleFeatureSpec,
    eopatch: Optional[EOPatch] = None,
    allowed_feature_types: Union[Iterable[FeatureType], EllipsisType] = ...,
) -> FeatureRenameSpec:
    """Parses input describing a single feature into a `(feature_type, old_name, new_name)` triple.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """

    features = FeatureParser([feature], allowed_feature_types=allowed_feature_types).get_renamed_features(eopatch)
    if len(features) != 1:
        raise ValueError(f"Specification {feature} resulted in {len(features)} features, expected 1.")
    return features[0]


def parse_features(
    features: FeaturesSpecification,
    eopatch: Optional[EOPatch] = None,
    allowed_feature_types: Union[Iterable[FeatureType], EllipsisType] = ...,
) -> List[FeatureSpec]:
    """Parses input describing features into a list of `(feature_type, feature_name)` pairs.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """
    return FeatureParser(features, allowed_feature_types=allowed_feature_types).get_features(eopatch)


def parse_renamed_features(
    features: FeaturesSpecification,
    eopatch: Optional[EOPatch] = None,
    allowed_feature_types: Union[Iterable[FeatureType], EllipsisType] = ...,
) -> List[FeatureRenameSpec]:
    """Parses input describing features into a list of `(feature_type, old_name, new_name)` triples.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """
    return FeatureParser(features, allowed_feature_types=allowed_feature_types).get_renamed_features(eopatch)
