"""
Credits:
Copyright (c) 2021-2021 Žiga Lukšič, Matej Aleksandrov (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import logging
import tempfile
import datetime

import pytest
import ray

from eolearn.core import (
    EOTask, EOWorkflow, EOExecutor, Dependency,  WorkflowResults, LinearWorkflow
)
from eolearn.core.eoworkflow_tasks import OutputTask
from eolearn.core.extra.ray import RayExecutor


class ExampleTask(EOTask):

    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.debug('Debug statement of Example task with kwargs: %s', kwargs)
        my_logger.info('Info statement of Example task with kwargs: %s', kwargs)
        my_logger.warning('Warning statement of Example task with kwargs: %s', kwargs)
        my_logger.critical('Super important log')

        if 'arg1' in kwargs and kwargs['arg1'] is None:
            raise Exception


class FooTask(EOTask):

    @staticmethod
    def execute(*_, **__):
        return 42


class KeyboardExceptionTask(EOTask):

    @staticmethod
    def execute(*_, **__):
        raise KeyboardInterrupt


class CustomLogFilter(logging.Filter):
    """ A custom filter that keeps only logs with level warning or critical
    """
    def filter(self, record):
        return record.levelno >= logging.WARNING


@pytest.fixture(name='simple_cluster', scope='module')
def simple_cluster_fixture():
    ray.init(log_to_driver=False)
    yield
    ray.shutdown()


@pytest.fixture(scope='session', name='test_tasks')
def test_tasks_fixture():
    tasks = {
        'example': ExampleTask(),
        'foo': FooTask(),
        'output': OutputTask('output')
    }
    return tasks


@pytest.fixture(name='workflow')
def workflow_fixture(test_tasks):
    example_task = test_tasks['example']
    foo_task = test_tasks['foo']
    output_task = test_tasks['output']

    workflow = EOWorkflow([
        (example_task, []),
        Dependency(task=foo_task, inputs=[example_task, example_task]),
        (output_task, [foo_task])
    ])
    return workflow


@pytest.fixture(name='execution_args')
def execution_args_fixture(test_tasks):
    example_task = test_tasks['example']

    execution_args = [
        {example_task: {'arg1': 1}},
        {},
        {example_task: {'arg1': 3, 'arg3': 10}},
        {example_task: {'arg1': None}}
    ]
    return execution_args


def test_fail_without_ray(workflow, execution_args):
    executor = RayExecutor(workflow, execution_args)
    with pytest.raises(RuntimeError):
        executor.run()


@pytest.mark.parametrize('filter_logs', [True, False])
@pytest.mark.parametrize('execution_names', [None, [4, 'x', 'y', 'z']])
def test_execution_logs(filter_logs, execution_names, workflow, execution_args, simple_cluster):

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = RayExecutor(
            workflow, execution_args, save_logs=True,
            logs_folder=tmp_dir_name,
            logs_filter=CustomLogFilter() if filter_logs else None,
            execution_names=execution_names
        )
        executor.run()

        assert len(executor.execution_logs) == 4
        for log in executor.execution_logs:
            assert len(log.split()) >= 3

        log_filenames = sorted(os.listdir(executor.report_folder))
        assert len(log_filenames) == 4

        if execution_names:
            for name, log_filename in zip(execution_names, log_filenames):
                assert log_filename == f'eoexecution-{name}.log'

        log_path = os.path.join(executor.report_folder, log_filenames[0])
        with open(log_path, 'r') as fp:
            line_count = len(fp.readlines())
            expected_line_count = 2 if filter_logs else 12
            assert line_count == expected_line_count


def test_execution_stats(workflow, execution_args, simple_cluster):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = RayExecutor(workflow, execution_args, logs_folder=tmp_dir_name)
        executor.run()

        assert len(executor.execution_stats) == 4
        for stats in executor.execution_stats:
            for time_stat in ['start_time', 'end_time']:
                assert time_stat in stats and isinstance(stats[time_stat], datetime.datetime)


def test_execution_errors(workflow, execution_args, simple_cluster):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = RayExecutor(workflow, execution_args, logs_folder=tmp_dir_name)
        executor.run()

        for idx, stats in enumerate(executor.execution_stats):
            if idx != 3:
                assert 'error' not in stats, f'Workflow {idx} should be executed without errors'
            else:
                assert 'error' in stats and stats['error'], 'This workflow should be executed with an error'

        assert executor.get_successful_executions() == [0, 1, 2]
        assert executor.get_failed_executions() == [3]


@pytest.mark.parametrize('return_results', [True, False])
def test_execution_results(return_results, workflow, execution_args, simple_cluster):
    executor = RayExecutor(workflow, execution_args)
    results = executor.run(return_results=return_results)

    if return_results:
        assert isinstance(results, list)

        for idx, workflow_results in enumerate(results):
            if idx == 3:
                assert workflow_results is None
            else:
                assert isinstance(workflow_results, WorkflowResults)
                assert workflow_results.outputs['output'] == 42
    else:
        assert results is None


def test_keyboard_interrupt(simple_cluster):
    exception_task = KeyboardExceptionTask()
    workflow = LinearWorkflow(exception_task)
    execution_args = []
    for _ in range(10):
        execution_args.append({exception_task: {'arg1': 1}})

    with pytest.raises((ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError)):
        RayExecutor(workflow, execution_args).run()


def test_reruns(workflow, execution_args, simple_cluster):
    executor = RayExecutor(workflow, execution_args)
    for _ in range(100):
        executor.run()

    for _ in range(10):
        RayExecutor(workflow, execution_args).run()

    executors = [RayExecutor(workflow, execution_args) for _ in range(10)]
    for executor in executors:
        executor.run()


def test_run_after_interrupt(workflow, execution_args, simple_cluster):
    foo_task = FooTask()
    exception_task = KeyboardExceptionTask()
    exception_workflow = LinearWorkflow(foo_task, exception_task)
    exception_executor = RayExecutor(exception_workflow, [{}])
    executor = RayExecutor(workflow, execution_args[:-1])  # removes args for exception

    result_preexception = executor.run(return_results=True)
    with pytest.raises((ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError)):
        exception_executor.run()
    result_postexception = executor.run(return_results=True)

    assert [res.outputs for res in result_preexception] == [res.outputs for res in result_postexception]


def test_mix_with_eoexecutor(workflow, execution_args, simple_cluster):
    rayexecutor = RayExecutor(workflow, execution_args)
    eoexecutor = EOExecutor(workflow, execution_args)
    for _ in range(10):
        ray_results = rayexecutor.run()
        eo_results = eoexecutor.run()
        assert ray_results == eo_results
