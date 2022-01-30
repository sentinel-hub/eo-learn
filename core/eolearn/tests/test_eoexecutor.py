"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

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

from eolearn.core import EOTask, EOWorkflow, EONode, EOExecutor, WorkflowResults, execute_with_mp_lock, OutputTask


class ExampleTask(EOTask):
    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.debug("Debug statement of Example task with kwargs: %s", kwargs)
        my_logger.info("Info statement of Example task with kwargs: %s", kwargs)
        my_logger.warning("Warning statement of Example task with kwargs: %s", kwargs)
        my_logger.critical("Super important log")

        if "arg1" in kwargs and kwargs["arg1"] is None:
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
    """A custom filter that keeps only logs with level warning or critical"""

    def filter(self, record):
        return record.levelno >= logging.WARNING


def logging_function(_=None):
    """Logs start, sleeps for 0.5s, logs end"""
    logging.info(multiprocessing.current_process().name)
    time.sleep(0.5)
    logging.info(multiprocessing.current_process().name)


@pytest.fixture(scope="session", name="num_workers")
def num_workers_fixture():
    return 5


@pytest.fixture(scope="session", name="test_nodes")
def test_nodes_fixture():
    example = EONode(ExampleTask())
    foo = EONode(FooTask(), inputs=[example, example])
    output = EONode(OutputTask("output"), inputs=[foo])
    nodes = {"example": example, "foo": foo, "output": output}
    return nodes


@pytest.fixture(name="workflow")
def workflow_fixture(test_nodes):
    workflow = EOWorkflow(list(test_nodes.values()))
    return workflow


@pytest.fixture(name="execution_kwargs")
def execution_kwargs_fixture(test_nodes):
    example_node = test_nodes["example"]

    execution_kwargs = [
        {example_node: {"arg1": 1}},
        {},
        {example_node: {"arg1": 3, "arg3": 10}},
        {example_node: {"arg1": None}},
    ]
    return execution_kwargs


@pytest.mark.parametrize(
    "test_args",
    [
        (1, True, False),
        (1, False, True),  # singleprocess
        (5, True, False),
        (3, True, True),  # multiprocess
        (3, False, False),
        (2, False, True),  # multithread
    ],
)
@pytest.mark.parametrize("execution_names", [None, [4, "x", "y", "z"]])
def test_read_logs(test_args, execution_names, workflow, execution_kwargs):
    workers, multiprocess, filter_logs = test_args
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = EOExecutor(
            workflow,
            execution_kwargs,
            save_logs=True,
            logs_folder=tmp_dir_name,
            logs_filter=CustomLogFilter() if filter_logs else None,
            execution_names=execution_names,
        )
        executor.run(workers=workers, multiprocess=multiprocess)

        execution_logs = executor.read_logs()
        assert len(execution_logs) == 4
        for log in execution_logs:
            assert len(log.split()) >= 3

        log_filenames = sorted(executor.filesystem.listdir(executor.report_folder))
        assert len(log_filenames) == 4

        if execution_names:
            for name, log_filename in zip(execution_names, log_filenames):
                assert log_filename == f"eoexecution-{name}.log"

        log_path = os.path.join(executor.report_folder, log_filenames[0])
        with executor.filesystem.open(log_path, "r") as fp:
            line_count = len(fp.readlines())
            expected_line_count = 2 if filter_logs else 12
            assert line_count == expected_line_count


def test_execution_results(workflow, execution_kwargs):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = EOExecutor(workflow, execution_kwargs, logs_folder=tmp_dir_name)
        executor.run(workers=2)

        assert len(executor.execution_results) == 4
        for results in executor.execution_results:
            for time_stat in [results.start_time, results.end_time]:
                assert isinstance(time_stat, datetime.datetime)


@pytest.mark.parametrize("multiprocess", [True, False])
def test_execution_errors(multiprocess, workflow, execution_kwargs):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = EOExecutor(workflow, execution_kwargs, logs_folder=tmp_dir_name)
        executor.run(workers=5, multiprocess=multiprocess)

        for idx, results in enumerate(executor.execution_results):
            if idx == 3:
                assert results.workflow_failed()
            else:
                assert not results.workflow_failed()

        assert executor.get_successful_executions() == [0, 1, 2]
        assert executor.get_failed_executions() == [3]


def test_execution_results(workflow, execution_kwargs):
    executor = EOExecutor(workflow, execution_kwargs)
    results = executor.run(workers=2, multiprocess=True)

    assert isinstance(results, list)

    for idx, workflow_results in enumerate(results):
        assert isinstance(workflow_results, WorkflowResults)
        if idx != 3:
            assert workflow_results.outputs["output"] == 42


def test_exceptions(workflow, execution_kwargs):
    with pytest.raises(ValueError):
        EOExecutor(workflow, {})
    with pytest.raises(ValueError):
        EOExecutor(workflow, execution_kwargs, execution_names={1, 2, 3, 4})
    with pytest.raises(ValueError):
        EOExecutor(workflow, execution_kwargs, execution_names=["a", "b"])


def test_keyboard_interrupt():
    exception_node = EONode(KeyboardExceptionTask())
    workflow = EOWorkflow([exception_node])
    execution_kwargs = []
    for _ in range(10):
        execution_kwargs.append({exception_node: {"arg1": 1}})

    run_kwargs = [{"workers": 1}, {"workers": 3, "multiprocess": True}, {"workers": 3, "multiprocess": False}]
    for kwarg in run_kwargs:
        with pytest.raises(KeyboardInterrupt):
            EOExecutor(workflow, execution_kwargs).run(**kwarg)


def test_with_lock(num_workers):
    with tempfile.NamedTemporaryFile() as fp:
        logger = logging.getLogger()
        handler = logging.FileHandler(fp.name)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            pool.map(execute_with_mp_lock, [logging_function] * num_workers)

        handler.close()
        logger.removeHandler(handler)

        with open(fp.name, "r") as log_file:
            lines = log_file.read().strip("\n ").split("\n")

        assert len(lines) == 2 * num_workers
        for idx in range(num_workers):
            assert lines[2 * idx], lines[2 * idx + 1]
        for idx in range(1, num_workers):
            assert lines[2 * idx - 1] != lines[2 * idx]


def test_without_lock(num_workers):
    with tempfile.NamedTemporaryFile() as fp:
        logger = logging.getLogger()
        handler = logging.FileHandler(fp.name)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            pool.map(logging_function, [None] * num_workers)

        handler.close()
        logger.removeHandler(handler)

        with open(fp.name, "r") as log_file:
            lines = log_file.read().strip("\n ").split("\n")

        assert len(lines) == 2 * num_workers
        assert len(set(lines[:num_workers])) == num_workers, "All processes should start"
        assert len(set(lines[num_workers:])) == num_workers, "All processes should finish"
