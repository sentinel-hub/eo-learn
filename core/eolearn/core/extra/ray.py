"""
Module containing integrations with Ray framework

In order to use this module you have to install `ray` Python package.

Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from typing import Generator, List

try:
    import ray
except ImportError as exception:
    raise ImportError("This module requires an installation of Ray Python package") from exception
from tqdm.auto import tqdm

from ..eoexecution import EOExecutor, _ProcessingData, _ProcessingType
from ..eoworkflow import WorkflowResults


class RayExecutor(EOExecutor):
    """A special type of `EOExecutor` that works with Ray framework"""

    def run(self) -> List[WorkflowResults]:
        """Runs the executor using a Ray cluster

        Before calling this method make sure to initialize a Ray cluster using `ray.init`.

        :return: A list of EOWorkflow results
        """
        if not ray.is_initialized():
            raise RuntimeError("Please initialize a Ray cluster before calling this method")

        workers = ray.available_resources().get("CPU")
        return super().run(workers=workers, multiprocess=True)

    @staticmethod
    def _get_processing_type(*_, **__) -> _ProcessingType:
        """Provides a type of processing for later references"""
        return _ProcessingType.RAY

    @classmethod
    def _run_execution(cls, processing_args: List[_ProcessingData], *_, **__) -> List[WorkflowResults]:
        """Runs ray execution"""
        futures = [_ray_workflow_executor.remote(workflow_args) for workflow_args in processing_args]

        for _ in tqdm(_progress_bar_iterator(futures), total=len(futures)):
            pass

        return ray.get(futures)


@ray.remote
def _ray_workflow_executor(workflow_args: _ProcessingData) -> WorkflowResults:
    """Called to execute a workflow on a ray worker"""
    # pylint: disable=protected-access
    return RayExecutor._execute_workflow(workflow_args)


def _progress_bar_iterator(futures: list, update_interval: float = 0.5) -> Generator:
    """A utility to help track finished ray processes

    Note that using tqdm(futures) directly would cause memory problems and is not accurate

    :param futures: List of `ray` futures.
    :param update_interval: How many seconds to wait before updating progress bar.
    :return: A `None` value generator that accurately shows progress of futures.
    """
    while futures:
        done, futures = ray.wait(futures, num_returns=len(futures), timeout=float(update_interval))
        yield from (None for _ in done)
