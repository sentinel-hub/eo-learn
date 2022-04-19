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

from itertools import repeat
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, Union, cast

from ..constants import FeatureType

if TYPE_CHECKING:
    from ..eodata import EOPatch


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

    4. Sequences of elements, each describing a feature. For elements describing a feature type it is understood as
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

    def __init__(self, features: Union[dict, Sequence], allowed_feature_types: Optional[Iterable[FeatureType]] = None):
        """
        :param features: A collection of features in one of the supported formats
        :param allowed_feature_types: Makes sure that only features of these feature types will be returned, otherwise
            an error is raised
        :raises: ValueError
        """
        self.allowed_feature_types = set(FeatureType) if allowed_feature_types is None else set(allowed_feature_types)
        self._feature_specs = self._parse_features(features)

    def _parse_features(
        self, features: Union[dict, Sequence]
    ) -> List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]]:
        """This method parses and validates input, returning a list of `(ftype, old_name, new_name)` triples.

        Due to typing issues the all-features requests are transformed from `(ftype, ...)` to `(ftype, None, None)`.
        This is a correct schema for BBOX and TIMESTAMP while for other features this is corrected when outputting,
        either by processing the request or by substituting ellipses back (case of `get_feature_specifications`).
        """
        if isinstance(features, dict):
            return self._parse_dict(features)

        if isinstance(features, Sequence):
            return self._parse_sequence(features)

        if features is ...:
            return list(zip(self.allowed_feature_types, repeat(None), repeat(None)))

        if features is FeatureType.BBOX or features is FeatureType.TIMESTAMP:
            return [(features, None, None)]

        raise ValueError(
            f"Unable to parse features {features}. Please see specifications of FeatureParser on viable inputs."
        )

    def _parse_dict(self, features: dict) -> List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]]:
        """Implements parsing and validation in case the input is a dictionary."""

        feature_specs: List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]] = []

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
        self, features: Sequence
    ) -> List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]]:
        """Implements parsing and validation in case the input is a sequence."""

        feature_specs: List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]] = []

        # Check for possible singleton
        if 2 <= len(features) <= 3:
            try:
                return [(self._parse_singleton(features))]
            except ValueError:
                pass

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

    def _parse_singleton(
        self, feature: Sequence
    ) -> Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]:
        """Parses a pair or triple specifying a single feature or a get-all request."""
        feature_type, *feature_name = feature
        feature_type = self._parse_feature_type(feature_type, message_about_position="first elements of tuples")

        if len(feature_name) == 1 and feature_name[0] in (..., None):
            return (feature_type, None, None)

        self._fail_for_noname_features(feature_type, feature_name)

        feature_name = feature_name[0] if len(feature_name) == 1 else feature_name
        parsed_name = self._parse_feature_name(feature_type, feature_name)
        return (feature_type, *parsed_name)

    def _parse_feature_type(self, feature_type, *, message_about_position: str) -> FeatureType:
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
    def _fail_for_noname_features(feature_type: FeatureType, specification: object):
        """Fails if the feature type does not support names.

        Should only be used after the viable names `...` and `None` have already been handled.
        """
        if not feature_type.has_dict():
            raise ValueError(
                f"For features of type {feature_type} the only acceptable specification is `...` or `None`, got"
                f" {specification} instead."
            )

    @staticmethod
    def _validate_parsing_request(feature_type: FeatureType, name: Optional[str], eopatch: Optional[EOPatch]):
        """Checks if the parsing request is viable with current arguments.

        This means checking that `eopatch` is provided if the request is an all-features request and in the case
        where an EOPatch is provided, that the feature exists in the EOPatch.
        """
        if not feature_type.has_dict():
            return

        if name is None and eopatch is None:
            raise ValueError(
                f"Input specifies that for feature type {feature_type} all existing features are parsed, but the "
                "`eopatch` parameter was not provided."
            )

        if eopatch is not None and name is not None and (feature_type, name) not in eopatch:
            raise ValueError(f"Requested feature {(feature_type, name)} not part of eopatch.")

    def get_feature_specifications(self) -> List[Tuple[FeatureType, object]]:
        """Returns the feature specifications in a more streamlined fashion.

        Requests for all features, e.g. `(FeatureType.DATA, ...)`, are returned directly.
        """
        return [(ftype, ... if fname is None else fname) for ftype, fname, _ in self._feature_specs]

    def get_features(self, eopatch: Optional[EOPatch] = None) -> List[Tuple[FeatureType, Optional[str]]]:
        """Returns a list of `(feature_type, feature_name)` pairs.

        For features that specify renaming, the new name of the feature is ignored.

        If `eopatch` is provided, the method checks that the EOPatch contains all the specified data and processes
        requests for all features, e.g. `(FeatureType.DATA, ...)`, by listing all available features of given type.

        If `eopatch` is not provided the method fails if an all-feature request is in the specification.
        """
        feature_names = []
        for feature_type, name, _ in self._feature_specs:
            self._validate_parsing_request(feature_type, name, eopatch)
            if name is None and feature_type.has_dict():
                feature_names.extend(list(zip(repeat(feature_type), eopatch[feature_type])))  # type: ignore
            else:
                feature_names.append((feature_type, name))
        return feature_names

    def get_renamed_features(
        self,
        eopatch: Optional[EOPatch] = None,
    ) -> List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]]:
        """Returns a list of `(feature_type, old_name, new_name)` triples.

        For features without a specified renaming the new name is equal to the old one.

        If `eopatch` is provided, the method checks that the EOPatch contains all the specified data and processes
        requests for all features, e.g. `(FeatureType.DATA, ...)`, by listing all available features of given type.
        In these cases the returned `old_name` and `new_name` are equal.

        If `eopatch` is not provided the method fails if an all-feature request is in the specification.
        """
        feature_names = []
        for feature_type, old_name, new_name in self._feature_specs:
            self._validate_parsing_request(feature_type, old_name, eopatch)
            if old_name is None and feature_type.has_dict():
                feature_names.extend(
                    list(zip(repeat(feature_type), eopatch[feature_type], eopatch[feature_type]))  # type: ignore
                )
            else:
                feature_names.append((feature_type, old_name, new_name))
        return feature_names


def parse_feature(
    feature, eopatch: Optional[EOPatch] = None, allowed_feature_types: Optional[Iterable[FeatureType]] = None
) -> Tuple[FeatureType, Optional[str]]:
    """Parses input describing a single feature into a `(feature_type, feature_name)` pair.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """

    features = FeatureParser([feature], allowed_feature_types=allowed_feature_types).get_features(eopatch)
    if len(features) != 1:
        raise ValueError(f"Specification {feature} resulted in {len(features)} features, expected 1.")
    return features[0]


def parse_renamed_feature(
    feature, eopatch: Optional[EOPatch] = None, allowed_feature_types: Optional[Iterable[FeatureType]] = None
) -> Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]:
    """Parses input describing a single feature into a `(feature_type, old_name, new_name)` triple.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """

    features = FeatureParser([feature], allowed_feature_types=allowed_feature_types).get_renamed_features(eopatch)
    if len(features) != 1:
        raise ValueError(f"Specification {feature} resulted in {len(features)} features, expected 1.")
    return features[0]


def parse_features(
    features, eopatch: Optional[EOPatch] = None, allowed_feature_types: Optional[Iterable[FeatureType]] = None
) -> List[Tuple[FeatureType, Optional[str]]]:
    """Parses input describing features into a list of `(feature_type, feature_name)` pairs.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """
    return FeatureParser(features, allowed_feature_types=allowed_feature_types).get_features(eopatch)


def parse_renamed_features(
    features, eopatch: Optional[EOPatch] = None, allowed_feature_types: Optional[Iterable[FeatureType]] = None
) -> List[Union[Tuple[FeatureType, str, str], Tuple[FeatureType, None, None]]]:
    """Parses input describing features into a list of `(feature_type, old_name, new_name)` triples.

    See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>` for viable inputs.
    """
    return FeatureParser(features, allowed_feature_types=allowed_feature_types).get_renamed_features(eopatch)
