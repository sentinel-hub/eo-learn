"""
Utilities for a basic Python parallelization that build on top of `concurrent.futures` module from Standard Python
library.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Collection, Generator, Iterable, List, TypeVar, cast

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from threading import Lock

    MULTIPROCESSING_LOCK: Lock | None = None
else:
    MULTIPROCESSING_LOCK = None

# pylint: disable=invalid-name
T = TypeVar("T")
FutureType = TypeVar("FutureType")
OutputType = TypeVar("OutputType")


class _ProcessingType(Enum):
    """Type of EOExecutor processing"""

    SINGLE_PROCESS = "single process"
    MULTIPROCESSING = "multiprocessing"
    MULTITHREADING = "multithreading"
    RAY = "ray"


def _decide_processing_type(workers: int | None, multiprocess: bool) -> _ProcessingType:
    """Decides processing type according to given parameters.

    :param workers: A number of workers to be used (either threads or processes). If a single worker is given it will
        always use the current thread and process.
    :param multiprocess: A flag to decide between multiple processes and multiple threads in a single process.
    :return: An enum defining a type of processing.
    """
    if workers == 1:
        return _ProcessingType.SINGLE_PROCESS
    return _ProcessingType.MULTIPROCESSING if multiprocess else _ProcessingType.MULTITHREADING


def parallelize(
    function: Callable[..., OutputType],
    *params: Iterable[Any],
    workers: int | None,
    multiprocess: bool = True,
    **tqdm_kwargs: Any,
) -> list[OutputType]:
    """Parallelizes the function on given parameters using the specified number of workers.

    :param function: A function to be parallelized.
    :param params: Sequences of parameters to be given to the function. It uses the same logic as Python `map` function.
    :param workers: Maximum number of time the function will be executed in parallel.
    :param multiprocess: If `True` it will use `concurrent.futures.ProcessPoolExecutor` which will distribute
        workflow executions among multiple processors. If `False` it will use
        `concurrent.futures.ThreadPoolExecutor` which will distribute workflow among multiple threads. In case of
        `workers=1` this parameter is ignored and workflows will be executed consecutively.
    :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
    :return: A list of function results.
    """
    if not params:
        return []
    processing_type = _decide_processing_type(workers=workers, multiprocess=multiprocess)

    if processing_type is _ProcessingType.SINGLE_PROCESS:
        size = len(params[0] if isinstance(params[0], Collection) else list(params[0]))
        return list(tqdm(map(function, *params), total=size, **tqdm_kwargs))

    if processing_type is _ProcessingType.MULTITHREADING:
        with ThreadPoolExecutor(max_workers=workers) as thread_executor:
            return submit_and_monitor_execution(thread_executor, function, *params, **tqdm_kwargs)

    # pylint: disable=global-statement
    global MULTIPROCESSING_LOCK
    try:
        MULTIPROCESSING_LOCK = multiprocessing.Manager().Lock()
        with ProcessPoolExecutor(max_workers=workers) as process_executor:
            return submit_and_monitor_execution(process_executor, function, *params, **tqdm_kwargs)
    finally:
        MULTIPROCESSING_LOCK = None


def execute_with_mp_lock(function: Callable[..., OutputType], *args: Any, **kwargs: Any) -> OutputType:
    """A helper utility function that executes a given function with multiprocessing lock if the process is being
    executed in a multiprocessing mode.

    :param function: A function
    :param args: Function's positional arguments
    :param kwargs: Function's keyword arguments
    """
    if multiprocessing.current_process().name == "MainProcess" or MULTIPROCESSING_LOCK is None:
        return function(*args, **kwargs)

    # pylint: disable=not-context-manager
    with MULTIPROCESSING_LOCK:
        return function(*args, **kwargs)


def submit_and_monitor_execution(
    executor: Executor,
    function: Callable[..., OutputType],
    *params: Iterable[Any],
    **tqdm_kwargs: Any,
) -> list[OutputType]:
    """Performs the execution parallelization and monitors the process using a progress bar.

    :param executor: An object that performs parallelization.
    :param function: A function to be parallelized.
    :param params: Each element in a sequence are parameters for a single call of `function`.
    :return: A list of results in the same order as input parameters given by `executor_params`.
    """
    futures = [executor.submit(function, *function_params) for function_params in zip(*params)]
    return join_futures(futures, **tqdm_kwargs)


def join_futures(futures: list[Future], **tqdm_kwargs: Any) -> list[Any]:
    """Resolves futures, monitors progress, and returns a list of results.

    :param futures: A list of futures to be joined. Note that this list will be reduced into an empty list as a side
        effect of this function. This way future objects will get cleared from memory already during the execution
        which will free some extra memory. But this can be achieved only if future objects aren't kept in memory
        outside `futures` list.
    :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
    :return: A list of results in the order that corresponds with the order of the given input `futures`.
    """
    results: list[Any | None] = [None] * len(futures)
    for position, result in join_futures_iter(futures, **tqdm_kwargs):
        results[position] = result

    return cast(List[Any], results)


def join_futures_iter(
    futures: list[Future], update_interval: float = 0.5, **tqdm_kwargs: Any
) -> Generator[tuple[int, Any], None, None]:
    """Resolves futures, monitors progress, and serves as an iterator over results.

    :param futures: A list of futures to be joined. Note that this list will be reduced into an empty list as a side
        effect of this function. This way future objects will get cleared from memory already during the execution
        which will free some extra memory. But this can be achieved only if future objects aren't kept in memory
        outside `futures` list.
    :param update_interval: A number of seconds to wait between consecutive updates of a progress bar.
    :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
    :return: A generator that will be returning pairs `(index, result)` where `index` will define the position of future
        in the original list to which `result` belongs to.
    """

    def _wait_function(remaining_futures: Collection[Future]) -> tuple[Collection[Future], Collection[Future]]:
        return concurrent.futures.wait(remaining_futures, timeout=float(update_interval), return_when=FIRST_COMPLETED)

    def _get_result(future: Future) -> Any:
        return future.result()

    return _base_join_futures_iter(_wait_function, _get_result, futures, **tqdm_kwargs)


def _base_join_futures_iter(
    wait_function: Callable[[Collection[FutureType]], tuple[Collection[FutureType], Collection[FutureType]]],
    get_result_function: Callable[[FutureType], OutputType],
    futures: list[FutureType],
    **tqdm_kwargs: Any,
) -> Generator[tuple[int, OutputType], None, None]:
    """A generalized utility function that resolves futures, monitors progress, and serves as an iterator over
    results."""
    remaining_futures: Collection[FutureType] = _make_copy_and_empty_given(futures)

    id_to_position_map = {id(future): index for index, future in enumerate(remaining_futures)}

    with tqdm(total=len(remaining_futures), **tqdm_kwargs) as pbar:
        while remaining_futures:
            done, remaining_futures = wait_function(remaining_futures)
            for future in done:
                result = get_result_function(future)
                pbar.update(1)
                yield id_to_position_map[id(future)], result


def _make_copy_and_empty_given(items: list[T]) -> list[T]:
    """Removes items from the given list and returns its copy. The side effect of removing items is intentional."""
    items_copy = items[:]
    while items:
        items.pop()
    return items_copy
