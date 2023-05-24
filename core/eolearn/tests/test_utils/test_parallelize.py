"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import itertools as it
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from eolearn.core.utils.parallelize import (
    _decide_processing_type,
    _ProcessingType,
    execute_with_mp_lock,
    join_futures,
    join_futures_iter,
    parallelize,
    submit_and_monitor_execution,
)


@pytest.mark.parametrize(
    ("workers", "multiprocess", "expected_type"),
    [
        (1, False, _ProcessingType.SINGLE_PROCESS),
        (1, True, _ProcessingType.SINGLE_PROCESS),
        (3, False, _ProcessingType.MULTITHREADING),
        (2, True, _ProcessingType.MULTIPROCESSING),
    ],
)
def test_decide_processing_type(workers, multiprocess, expected_type):
    processing_type = _decide_processing_type(workers=workers, multiprocess=multiprocess)
    assert processing_type is expected_type


def test_execute_with_mp_lock():
    """For now just a basic dummy test."""
    result = execute_with_mp_lock(sorted, range(10), key=lambda value: -value)
    assert result == list(range(9, -1, -1))


@pytest.mark.parametrize(
    ("workers", "multiprocess"),
    [
        (1, True),
        (3, False),
        (2, True),
    ],
)
def test_parallelize(workers, multiprocess):
    results = parallelize(max, range(10), it.repeat(5), workers=workers, multiprocess=multiprocess, desc="Test")
    assert results == [5] * 5 + list(range(5, 10))


@pytest.mark.parametrize("executor_class", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_submit_and_monitor_execution(executor_class):
    with executor_class(max_workers=2) as executor:
        results = submit_and_monitor_execution(executor, max, range(10), it.repeat(5), disable=True)

    assert results == [5] * 5 + list(range(5, 10))


def plus_one(value):
    return value + 1


@pytest.mark.parametrize("executor_class", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_join_futures(executor_class):
    with executor_class() as executor:
        futures = [executor.submit(plus_one, value) for value in range(5)]
        results = join_futures(futures)

    assert results == list(range(1, 6))
    assert futures == []


@pytest.mark.parametrize("executor_class", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_join_futures_iter(executor_class):
    with executor_class() as executor:
        futures = [executor.submit(plus_one, value) for value in range(5)]
        results = []
        for value in join_futures_iter(futures):
            assert futures == []
            results.append(value)

    assert sorted(results) == [(num, num + 1) for num in range(5)]
