import numpy as np

from eolearn.mask.utils import map_over_axis


def test_map_over_axis():
    data = np.ones((5, 10, 10))
    result = map_over_axis(data, lambda _: np.zeros((7, 20)), axis=0)
    assert result.shape == (5, 7, 20)
    result = map_over_axis(data, lambda _: np.zeros((7, 20)), axis=1)
    assert result.shape == (7, 10, 20)
    result = map_over_axis(data, lambda _: np.zeros((5, 10)), axis=1)
    assert result.shape == (5, 10, 10)
