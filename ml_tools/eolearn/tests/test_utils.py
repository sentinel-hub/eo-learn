from typing import Any, Dict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.ml_tools.utils import rolling_window


@pytest.mark.parametrize(
    "input_array, args, expected_array",
    [
        (
            np.arange(9).reshape(3, 3),
            {"window": (2, 2)},
            np.array([[[[0, 1], [3, 4]], [[1, 2], [4, 5]]], [[[3, 4], [6, 7]], [[4, 5], [7, 8]]]]),
        ),
        (np.arange(9).reshape(3, 3), {"window": (2, 0), "asteps": (2, 1)}, np.array([[[0, 3], [1, 4], [2, 5]]])),
        (
            np.arange(10),
            {"window": 3, "wsteps": 2},
            np.array([[0, 2, 4], [1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9]]),
        ),
    ],
)
def test_rolling_window(input_array: np.ndarray, args: Dict[str, Any], expected_array: np.ndarray) -> None:
    assert_array_equal(rolling_window(input_array, **args), expected_array)
