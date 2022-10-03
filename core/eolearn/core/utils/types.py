"""
Types and type aliases used throughout the code.

Credits:
Copyright (c) 2022 Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
# pylint: disable=unused-import
import sys

if sys.version_info >= (3, 10):
    from types import EllipsisType  # pylint: disable=ungrouped-imports
else:
    import builtins  # noqa: F401

    from typing_extensions import TypeAlias

    EllipsisType: TypeAlias = "builtins.ellipsis"

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal  # pylint: disable=ungrouped-imports # noqa: F401
