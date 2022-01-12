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
    EOTask, EOWorkflow, EOExecutor, EONode,  WorkflowResults
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


@pytest.fixture(scope='session', name='test_nodes')
def test_nodes_fixture():
    example = EONode(ExampleTask())
    foo = EONode(FooTask(), inputs=[example, example])
    output = EONode(OutputTask('output'), inputs=[foo])
    nodes = {'example': example, 'foo': foo, 'output': output}
    return nodes


@pytest.fixture(name='workflow')
def workflow_fixture(test_nodes):
    workflow = EOWorkflow(list(test_nodes.values()))
    return workflow


@pytest.fixture(name='execution_args')
def execution_args_fixture(test_nodes):
    example_node = test_nodes['example']

    execution_args = [
        {example_node: {'arg1': 1}},
        {},
        {example_node: {'arg1': 3, 'arg3': 10}},
        {example_node: {'arg1': None}}
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


def test_execution_results(workflow, execution_args, simple_cluster):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = RayExecutor(workflow, execution_args, logs_folder=tmp_dir_name)
        executor.run()

        assert len(executor.execution_results) == 4
        for results in executor.execution_results:
            for time_stat in [results.start_time, results.end_time]:
                assert isinstance(time_stat, datetime.datetime)


def test_execution_errors(workflow, execution_args, simple_cluster):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = RayExecutor(workflow, execution_args, logs_folder=tmp_dir_name)
        executor.run()

        for idx, results in enumerate(executor.execution_results):
            if idx == 3:
                assert results.workflow_failed()
            else:
                assert not results.workflow_failed()

        assert executor.get_successful_executions() == [0, 1, 2]
        assert executor.get_failed_executions() == [3]


def test_execution_results(workflow, execution_args, simple_cluster):
    executor = RayExecutor(workflow, execution_args)
    results = executor.run()

    assert isinstance(results, list)
    for idx, workflow_results in enumerate(results):
        assert isinstance(workflow_results, WorkflowResults)
        if idx != 3:
            assert workflow_results.outputs['output'] == 42


def test_keyboard_interrupt(simple_cluster):
    exception_node = EONode(KeyboardExceptionTask())
    workflow = EOWorkflow([exception_node])
    execution_args = []
    for _ in range(10):
        execution_args.append({exception_node: {'arg1': 1}})

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
    foo_node = EONode(FooTask())
    exception_node = EONode(KeyboardExceptionTask(), inputs=[foo_node])
    exception_workflow = EOWorkflow([foo_node, exception_node])
    exception_executor = RayExecutor(exception_workflow, [{}])
    executor = RayExecutor(workflow, execution_args[:-1])  # removes args for exception

    result_preexception = executor.run()
    with pytest.raises((ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError)):
        exception_executor.run()
    result_postexception = executor.run()

    assert [res.outputs for res in result_preexception] == [res.outputs for res in result_postexception]


def test_mix_with_eoexecutor(workflow, execution_args, simple_cluster):
    rayexecutor = RayExecutor(workflow, execution_args)
    eoexecutor = EOExecutor(workflow, execution_args)
    for _ in range(10):
        ray_results = rayexecutor.run()
        eo_results = eoexecutor.run()

        ray_outputs = [results.outputs for results in ray_results]
        eo_outputs = [results.outputs for results in eo_results]

        assert ray_outputs == eo_outputs
