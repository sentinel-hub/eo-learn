"""
Module containing integrations with Ray framework

In order to use this module you have to install `ray` Python package.

Credits:
Copyright (c) 2021-2021 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
try:
    import ray
except ImportError as exception:
    raise ImportError('This module requires an installation of ray Python package') from exception
from tqdm.auto import tqdm

from ..eoexecution import EOExecutor, _ProcessingType


class RayExecutor(EOExecutor):
    """ A special type of `EOExecutor` that works with Ray framework
    """
    def run(self, return_results=False):
        """ Runs the executor using a Ray cluster

        Before calling this method make sure to initialize a Ray cluster using `ray.init`.

        :param return_results: If `True` this method will return a list of all results of the execution. Note that
            this might exceed the available memory. By default this parameter is set to `False`.
        :type: bool
        :return: If `return_results` is set to `True` it will return a list of results, otherwise it will return `None`
        :rtype: None or list(eolearn.core.WorkflowResults)
        """
        if not ray.is_initialized():
            raise RuntimeError('Please initialize a Ray cluster before calling this method')

        workers = ray.available_resources().get('CPU')
        return super().run(workers=workers, multiprocess=True, return_results=return_results)

    @staticmethod
    def _get_processing_type(*_, **__):
        """ Provides a type of processing for later references
        """
        return _ProcessingType.RAY

    @classmethod
    def _run_execution(cls, processing_args, *_, **__):
        """ Runs ray execution
        """
        futures = [_ray_workflow_executor.remote(workflow_args) for workflow_args in processing_args]

        for _ in tqdm(_progress_bar_iterator(futures), total=len(futures)):
            pass

        return ray.get(futures)


@ray.remote
def _ray_workflow_executor(workflow_args):
    """ Called to execute a workflow on a ray worker
    """
    # pylint: disable=protected-access
    return RayExecutor._execute_workflow(workflow_args)


def _progress_bar_iterator(futures):
    """ A utility to help tracking finished ray processes

    Note that using tqdm(futures) directly would cause memory problems and is not accurate
    """
    while futures:
        _, futures = ray.wait(futures, num_returns=1)
        yield
