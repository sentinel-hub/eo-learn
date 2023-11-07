"""
Module containing integrations with Ray framework

In order to use this module you have to install `ray` Python package.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from logging import FileHandler, Filter
from typing import Any, Callable, Collection, Generator, Iterable, List, Sequence, TypeVar, cast

from fs.base import FS

from eolearn.core.eonode import EONode

try:
    import ray
except ImportError as exception:
    raise ImportError("This module requires an installation of Ray Python package") from exception

from ..eoexecution import EOExecutor, _ExecutionRunParams, _HandlerFactoryType, _ProcessingData
from ..eoworkflow import EOWorkflow, WorkflowResults
from ..utils.parallelize import _base_join_futures_iter

# pylint: disable=invalid-name
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class RayExecutor(EOExecutor):
    """A special type of `EOExecutor` that works with Ray framework"""

    def __init__(
        self,
        workflow: EOWorkflow,
        execution_kwargs: Sequence[dict[EONode, dict[str, object]]],
        *,
        execution_names: list[str] | None = None,
        save_logs: bool = False,
        logs_folder: str = ".",
        filesystem: FS | None = None,
        logs_filter: Filter | None = None,
        logs_handler_factory: _HandlerFactoryType = FileHandler,
        raise_on_temporal_mismatch: bool = False,
        ray_remote_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            workflow,
            execution_kwargs,
            execution_names=execution_names,
            save_logs=save_logs,
            logs_folder=logs_folder,
            filesystem=filesystem,
            logs_filter=logs_filter,
            logs_handler_factory=logs_handler_factory,
            raise_on_temporal_mismatch=raise_on_temporal_mismatch,
        )
        self.ray_remote_kwargs = ray_remote_kwargs

    def run(self, **tqdm_kwargs: Any) -> list[WorkflowResults]:  # type: ignore[override]
        """Runs the executor using a Ray cluster

        Before calling this method make sure to initialize a Ray cluster using `ray.init`.

        :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
        :return: A list of EOWorkflow results
        """
        if not ray.is_initialized():
            raise RuntimeError("Please initialize a Ray cluster before calling this method")

        workers = ray.available_resources().get("CPU")
        return super().run(workers=workers, multiprocess=True, **tqdm_kwargs)

    def _run_execution(
        self, processing_args: list[_ProcessingData], run_params: _ExecutionRunParams
    ) -> list[WorkflowResults]:
        """Runs ray execution"""
        remote_kwargs = self.ray_remote_kwargs or {}
        exec_func = _ray_workflow_executor.options(**remote_kwargs)  # type: ignore[attr-defined]
        futures = [exec_func.remote(workflow_args) for workflow_args in processing_args]
        return join_ray_futures(futures, **run_params.tqdm_kwargs)


@ray.remote
def _ray_workflow_executor(workflow_args: _ProcessingData) -> WorkflowResults:
    """Called to execute a workflow on a ray worker"""
    # pylint: disable=protected-access
    return RayExecutor._execute_workflow(workflow_args)  # noqa: SLF001


def parallelize_with_ray(
    function: Callable[[InputType], OutputType],
    *params: Iterable[InputType],
    ray_remote_kwargs: dict[str, Any] | None = None,
    **tqdm_kwargs: Any,
) -> list[OutputType]:
    """Parallelizes function execution with Ray.

    Note that this function will automatically connect to a Ray cluster, if a connection wouldn't exist yet. But it
    won't automatically shut down the connection.

    :param function: A normal function that is not yet decorated by `ray.remote`.
    :param params: Iterables of parameters that will be used with given function.
    :param ray_remote_kwargs: Keyword arguments passed to `ray.remote`.
    :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
    :return: A list of results in the order that corresponds with the order of the given input `params`.
    """
    ray_remote_kwargs = ray_remote_kwargs or {}
    if not ray.is_initialized():
        raise RuntimeError("Please initialize a Ray cluster before calling this method")

    ray_function = ray.remote(function, **ray_remote_kwargs)
    futures = [ray_function.remote(*function_params) for function_params in zip(*params)]
    return join_ray_futures(futures, **tqdm_kwargs)


def join_ray_futures(futures: list[ray.ObjectRef], **tqdm_kwargs: Any) -> list[Any]:
    """Resolves futures, monitors progress, and returns a list of results.

    :param futures: A list of futures to be joined. Note that this list will be reduced into an empty list as a side
        effect of this function. This way Ray future objects will get cleared from memory already during the execution
        and this will free memory from Ray Plasma store. But this can be achieved only if future objects aren't kept in
        memory outside `futures` list.
    :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
    :return: A list of results in the order that corresponds with the order of the given input `futures`.
    """
    results: list[Any | None] = [None] * len(futures)
    for position, result in join_ray_futures_iter(futures, **tqdm_kwargs):
        results[position] = result

    return cast(List[Any], results)


def join_ray_futures_iter(
    futures: list[ray.ObjectRef], update_interval: float = 0.5, **tqdm_kwargs: Any
) -> Generator[tuple[int, Any], None, None]:
    """Resolves futures, monitors progress, and serves as an iterator over results.

    :param futures: A list of futures to be joined. Note that this list will be reduced into an empty list as a side
        effect of this function. This way Ray future objects will get cleared from memory already during the execution
        and this will free memory from Ray Plasma store. But this can be achieved only if future objects aren't kept in
        memory outside `futures` list.
    :param update_interval: A number of seconds to wait between consecutive updates of a progress bar.
    :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
    :return: A generator that will be returning pairs `(index, result)` where `index` will define the position of future
        in the original list to which `result` belongs to.
    """

    def _ray_wait_function(
        remaining_futures: Collection[ray.ObjectRef],
    ) -> tuple[Collection[ray.ObjectRef], Collection[ray.ObjectRef]]:
        return ray.wait(remaining_futures, num_returns=len(remaining_futures), timeout=float(update_interval))

    return _base_join_futures_iter(_ray_wait_function, ray.get, futures, **tqdm_kwargs)
