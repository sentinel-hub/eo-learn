"""
Types and type aliases used throughout the code.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import sys

# pylint: disable=unused-import
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

from .constants import FeatureType

if sys.version_info >= (3, 10):
    from types import EllipsisType  # pylint: disable=ungrouped-imports
    from typing import TypeAlias
else:
    import builtins  # noqa: F401, RUF100

    from typing_extensions import TypeAlias

    EllipsisType: TypeAlias = "builtins.ellipsis"


# DEVELOPER NOTE: the #: comments are applied as docstrings

#: Specification describing a single feature
FeatureSpec: TypeAlias = Union[Tuple[Literal[FeatureType.BBOX, FeatureType.TIMESTAMPS], None], Tuple[FeatureType, str]]
#: Specification describing a feature with its current and desired new name
FeatureRenameSpec: TypeAlias = Union[
    Tuple[Literal[FeatureType.BBOX, FeatureType.TIMESTAMPS], None, None], Tuple[FeatureType, str, str]
]
SingleFeatureSpec: TypeAlias = Union[FeatureSpec, FeatureRenameSpec]

SequenceFeatureSpec: TypeAlias = Sequence[
    Union[SingleFeatureSpec, FeatureType, Tuple[FeatureType, Optional[EllipsisType]]]
]
DictFeatureSpec: TypeAlias = Dict[FeatureType, Union[None, EllipsisType, Iterable[Union[str, Tuple[str, str]]]]]
MultiFeatureSpec: TypeAlias = Union[
    EllipsisType, FeatureType, Tuple[FeatureType, EllipsisType], SequenceFeatureSpec, DictFeatureSpec
]

#: Specification of a single or multiple features. See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`.
FeaturesSpecification: TypeAlias = Union[SingleFeatureSpec, MultiFeatureSpec]
