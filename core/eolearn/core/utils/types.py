"""Deprecated module for types, moved to `eolearn.core.types`."""
from warnings import warn

from ..exceptions import EODeprecationWarning
from ..types import *  # noqa # pylint: disable=wildcard-import,unused-wildcard-import

warn(
    "The module `eolearn.core.utils.types` is deprecated, use `eolearn.core.types` instead.",
    category=EODeprecationWarning,
    stacklevel=2,
)
