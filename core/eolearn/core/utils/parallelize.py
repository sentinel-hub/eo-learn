"""
Utilities for a basic Python parallelization that build on top of `concurrent.futures` module from Standard Python
library.

Credits:
Copyright (c) 2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import concurrent.futures
import multiprocessing
from concurrent.futures import Future
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from threading import Lock

    MULTIPROCESSING_LOCK: Optional[Lock] = None
else:
    MULTIPROCESSING_LOCK = None

# pylint: disable=invalid-name
_InputType = TypeVar("_InputType")
_OutputType = TypeVar("_OutputType")


class _ProcessingType(Enum):
    """Type of EOExecutor processing"""

    SINGLE_PROCESS = "single process"
    MULTIPROCESSING = "multiprocessing"
    MULTITHREADING = "multithreading"
    RAY = "ray"


def decide_processing_type(workers: int, multiprocess: bool) -> _ProcessingType:  # TODO: private or public??
    """Decides processing type according to given parameters.

    :param workers: A number of workers to be used (either threads or processes). If a single worker is given it will
        always use the current thread and process.
    :param multiprocess: A flag to decide between multiple processes and multiple threads in a single process.
    :return: An enum defining a type of processing.
    """
    if workers == 1:
        return _ProcessingType.SINGLE_PROCESS
    if multiprocess:
        return _ProcessingType.MULTIPROCESSING
    return _ProcessingType.MULTITHREADING


def parallelize(function: Callable, *params: Any, workers: int, multiprocess: bool = False, **tqdm_kwargs: Any) -> list:
    """TODO"""
    if not params:
        raise ValueError(
            "At least 1 list of parameters should be given. Otherwise it is not clear how many times the"
            "function has to be executed."
        )
    processing_type = decide_processing_type(workers=workers, multiprocess=multiprocess)

    if processing_type is _ProcessingType.SINGLE_PROCESS:
        return list(tqdm(map(function, *params), total=len(params[0])))

    if processing_type is _ProcessingType.MULTITHREADING:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as thread_executor:
            return submit_and_monitor_execution(thread_executor, function, *params, **tqdm_kwargs)

    # pylint: disable=global-statement
    global MULTIPROCESSING_LOCK
    try:
        MULTIPROCESSING_LOCK = multiprocessing.Manager().Lock()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as process_executor:
            return submit_and_monitor_execution(process_executor, function, *params, **tqdm_kwargs)
    finally:
        MULTIPROCESSING_LOCK = None


def submit_and_monitor_execution(
    executor: concurrent.futures.Executor,
    function: Callable[[_InputType], _OutputType],
    *params: Iterable[_InputType],  # TODO
    **tqdm_kwargs: Any,
) -> List[_OutputType]:
    """Performs the execution parallelization and monitors the process using a progress bar.

    :param executor: An object that performs parallelization.
    :param function: A function to be parallelized.
    :param params: Each element in a sequence are parameters for a single call of `function`.
    :return: A list of results in the same order as input parameters given by `executor_params`.
    """
    futures = [executor.submit(function, *function_params) for function_params in zip(*params)]
    return join_futures(futures, **tqdm_kwargs)


def join_futures(futures: List[Future], **tqdm_kwargs: Any) -> List[Any]:
    results: List[Optional[Any]] = [None] * len(futures)
    for position, result in join_futures_iter(futures, **tqdm_kwargs):
        results[position] = result

    return cast(List[Any], results)


def join_futures_iter(futures: List[Future], **tqdm_kwargs: Any) -> Generator[Tuple[int, Any], None, None]:
    id_to_position_map = {id(future): index for index, future in enumerate(futures)}

    with tqdm(total=len(futures), **tqdm_kwargs) as pbar:
        for future in concurrent.futures.as_completed(futures):  # TODO
            result = future.result()
            result_position = id_to_position_map[id(future)]
            pbar.update(1)
            yield result_position, result


def execute_with_mp_lock(execution_function: Callable, *args, **kwargs) -> object:
    """A helper utility function that executes a given function with multiprocessing lock if the process is being
    executed in a multiprocessing mode.

    :param execution_function: A function
    :param args: Function's positional arguments
    :param kwargs: Function's keyword arguments
    :return: Function's results
    """
    if multiprocessing.current_process().name == "MainProcess" or MULTIPROCESSING_LOCK is None:
        return execution_function(*args, **kwargs)

    # pylint: disable=not-context-manager
    with MULTIPROCESSING_LOCK:
        return execution_function(*args, **kwargs)
