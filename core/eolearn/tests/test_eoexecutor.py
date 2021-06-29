"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import logging
import tempfile
import datetime
import concurrent.futures
import multiprocessing
import time

import pytest

from eolearn.core import (
    EOTask, EOWorkflow, Dependency, EOExecutor, WorkflowResults, execute_with_mp_lock, LinearWorkflow
)
from eolearn.core.eoworkflow_tasks import OutputTask


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


def logging_function(_=None):
    """ Logs start, sleeps for 0.5s, logs end
    """
    logging.info(multiprocessing.current_process().name)
    time.sleep(0.5)
    logging.info(multiprocessing.current_process().name)


@pytest.fixture(scope='session', name='num_workers')
def num_workers_fixture():
    return 5


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


@pytest.mark.parametrize(
    'test_args',
    [
        (1, True, False), (1, False, True),   # singleprocess
        (5, True, False), (3, True, True),    # multiprocess
        (3, False, False), (2, False, True),  # multithread
    ]
)
def test_execution_logs(test_args, workflow, execution_args):
    workers, multiprocess, filter_logs = test_args
    for execution_names in [None, [4, 'x', 'y', 'z']]:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executor = EOExecutor(
                workflow, execution_args, save_logs=True,
                logs_folder=tmp_dir_name,
                logs_filter=CustomLogFilter() if filter_logs else None,
                execution_names=execution_names
            )
            executor.run(workers=workers, multiprocess=multiprocess)

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


def test_execution_stats(workflow, execution_args):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = EOExecutor(workflow, execution_args, logs_folder=tmp_dir_name)
        executor.run(workers=2)

        assert len(executor.execution_stats) == 4
        for stats in executor.execution_stats:
            for time_stat in ['start_time', 'end_time']:
                assert time_stat in stats and isinstance(stats[time_stat], datetime.datetime)


def test_execution_errors(workflow, execution_args):
    for multiprocess in [True, False]:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executor = EOExecutor(workflow, execution_args, logs_folder=tmp_dir_name)
            executor.run(workers=5, multiprocess=multiprocess)

            for idx, stats in enumerate(executor.execution_stats):
                if idx != 3:
                    assert 'error' not in stats, f'Workflow {idx} should be executed without errors'
                else:
                    assert 'error' in stats and stats['error'], 'This workflow should be executed with an error'

            assert executor.get_successful_executions() == [0, 1, 2]
            assert executor.get_failed_executions() == [3]


def test_execution_results(workflow, execution_args):
    for return_results in [True, False]:

        executor = EOExecutor(workflow, execution_args)
        results = executor.run(workers=2, multiprocess=True, return_results=return_results)

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


def test_exceptions(workflow, execution_args):

    with pytest.raises(ValueError):
        EOExecutor(workflow, {})

    with pytest.raises(ValueError):
        EOExecutor(workflow, execution_args, execution_names={1, 2, 3, 4})
    with pytest.raises(ValueError):
        EOExecutor(workflow, execution_args, execution_names=['a', 'b'])


def test_keyboard_interrupt():
    exception_task = KeyboardExceptionTask()
    workflow = LinearWorkflow(exception_task)
    execution_args = []
    for _ in range(10):
        execution_args.append({exception_task: {'arg1': 1}})

    run_args = [{'workers': 1},
                {'workers': 3, 'multiprocess': True},
                {'workers': 3, 'multiprocess': False}]
    for arg in run_args:
        with pytest.raises(KeyboardInterrupt):
            EOExecutor(workflow, execution_args).run(**arg)


def test_with_lock(num_workers):
    with tempfile.NamedTemporaryFile() as fp:
        logger = logging.getLogger()
        handler = logging.FileHandler(fp.name)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            pool.map(execute_with_mp_lock, [logging_function] * num_workers)

        handler.close()
        logger.removeHandler(handler)

        with open(fp.name, 'r') as log_file:
            lines = log_file.read().strip('\n ').split('\n')

        assert (len(lines) == 2 * num_workers)
        for idx in range(num_workers):
            assert lines[2 * idx], lines[2 * idx + 1]
        for idx in range(1, num_workers):
            assert lines[2 * idx - 1] != lines[2 * idx]


def test_without_lock(num_workers):
    with tempfile.NamedTemporaryFile() as fp:
        logger = logging.getLogger()
        handler = logging.FileHandler(fp.name)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            pool.map(logging_function, [None] * num_workers)

        handler.close()
        logger.removeHandler(handler)

        with open(fp.name, 'r') as log_file:
            lines = log_file.read().strip('\n ').split('\n')

        assert len(lines) == 2 * num_workers
        assert len(set(lines[: num_workers])) == num_workers, 'All processes should start'
        assert len(set(lines[num_workers:])) == num_workers, 'All processes should finish'
