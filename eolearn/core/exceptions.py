"""
Implementation of custom eo-learn exceptions and warnings

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import warnings


class EODeprecationWarning(DeprecationWarning):
    """A custom deprecation warning for eo-learn package."""


class EOUserWarning(UserWarning):
    """A custom user warning for eo-learn package."""


class EORuntimeWarning(RuntimeWarning):
    """A custom runtime warning for eo-learn package."""


class TemporalDimensionWarning(RuntimeWarning):
    """A custom runtime warning for cases where EOPatches are temporally ill defined."""


warnings.simplefilter("default", EODeprecationWarning)
warnings.simplefilter("default", EOUserWarning)
warnings.simplefilter("always", EORuntimeWarning)
