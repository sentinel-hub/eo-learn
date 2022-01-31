"""
Implementation of custom eo-learn exceptions and warnings

Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import warnings


class EODeprecationWarning(DeprecationWarning):
    """A custom deprecation warning for eo-learn package."""


class EOUserWarning(UserWarning):
    """A custom user warning for eo-learn package."""


class EORuntimeWarning(RuntimeWarning):
    """A custom runtime warning for eo-learn package."""


warnings.simplefilter("default", EODeprecationWarning)
warnings.simplefilter("default", EOUserWarning)
warnings.simplefilter("always", EORuntimeWarning)
