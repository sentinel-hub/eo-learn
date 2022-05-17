"""
Credits:
Copyright (c) 2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from eolearn.core.utils.parallelize import join_futures, join_futures_iter


def plus_one(value):
    return value + 1


@pytest.mark.parametrize("executor_class", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_join_futures(executor_class):
    with executor_class() as executor:
        futures = [executor.submit(plus_one, value) for value in range(5)]
        results = join_futures(futures)

    assert results == list(range(1, 6))
    # assert futures == []


@pytest.mark.parametrize("executor_class", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_join_futures_iter(executor_class):
    with executor_class() as executor:
        futures = [executor.submit(plus_one, value) for value in range(5)]
        results = []
        for value in join_futures_iter(futures):
            # assert futures == []
            results.append(value)

    assert sorted(results) == [(num, num + 1) for num in range(5)]
