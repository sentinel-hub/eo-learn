"""Deprecated module for types, moved to `eolearn.core.types`."""

from __future__ import annotations

from warnings import warn

from ..exceptions import EODeprecationWarning

# pylint: disable-next=wildcard-import,unused-wildcard-import
from ..types import *  # noqa: F403

warn(
    "The module `eolearn.core.utils.types` is deprecated, use `eolearn.core.types` instead.",
    category=EODeprecationWarning,
    stacklevel=2,
)
