"""
Credits:
Copyright (c) 2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest

from eolearn.core.exceptions import renamed_and_deprecated


def test_renamed_and_deprecated():
    """Ensures that the decorator works as intended, i.e. warns on every initialization without changing it."""

    class TestClass:
        def __init__(self, arg1, kwarg1=10):
            self.arg = arg1
            self.kwarg = kwarg1
            self.secret = "exists"

    @renamed_and_deprecated
    class DeprecatedClass(TestClass):
        pass

    _ = TestClass(1)
    _ = TestClass(2, kwarg1=20)
    with pytest.warns():
        case1 = DeprecatedClass(1)
    with pytest.warns():
        case2 = DeprecatedClass(2, kwarg1=20)
    with pytest.warns():
        case3 = DeprecatedClass(3, 30)

    assert case1.arg == 1 and case1.kwarg == 10 and case1.secret == "exists"
    assert case2.arg == 2 and case2.kwarg == 20 and case2.secret == "exists"
    assert case3.arg == 3 and case3.kwarg == 30 and case3.secret == "exists"
