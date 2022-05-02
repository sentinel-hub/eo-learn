"""
The module handles execution and monitoring of workflows. It enables executing a workflow multiple times and in
parallel. It monitors execution times and handles any error that might occur in the process. At the end it generates a
report which contains summary of the workflow and process of execution.

All this is implemented in EOExecutor class.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import concurrent.futures
import datetime as dt
import logging
import multiprocessing
import threading
import warnings
from dataclasses import dataclass
from enum import Enum
from logging import Filter, Handler, Logger
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, cast

import fs
from fs.base import FS
from tqdm.auto import tqdm

from .eonode import EONode
from .eoworkflow import EOWorkflow, WorkflowResults
from .exceptions import EORuntimeWarning
from .utils.fs import get_base_filesystem_and_path, get_full_path
from .utils.logging import LogFileFilter

LOGGER = logging.getLogger(__name__)
MULTIPROCESSING_LOCK = None

# pylint: disable=invalid-name
_InputType = TypeVar("_InputType")
_OutputType = TypeVar("_OutputType")
_HandlerFactoryType = Callable[[str], Handler]


class _ProcessingType(Enum):
    """Type of EOExecutor processing"""

    SINGLE_PROCESS = "single process"
    MULTIPROCESSING = "multiprocessing"
    MULTITHREADING = "multithreading"
    RAY = "ray"


@dataclass(frozen=True)
class _ProcessingData:
    """Data to be used in EOExecutor processing. This will be passed to a process pool, so everything has to be
    serializable with pickle."""

    workflow: EOWorkflow
    workflow_kwargs: Dict[EONode, Dict[str, object]]
    log_path: Optional[str]
    filter_logs_by_thread: bool
    logs_filter: Optional[Filter]
    logs_handler_factory: _HandlerFactoryType


class EOExecutor:
    """Simultaneously executes a workflow with different input arguments. In the process it monitors execution and
    handles errors. It can also save logs and create a html report about each execution.
    """

    REPORT_FILENAME = "report.html"
    STATS_START_TIME = "start_time"
    STATS_END_TIME = "end_time"

    def __init__(
        self,
        workflow: EOWorkflow,
        execution_kwargs: Sequence[Dict[EONode, Dict[str, object]]],
        *,
        execution_names: Optional[List[str]] = None,
        save_logs: bool = False,
        logs_folder: str = ".",
        filesystem: Optional[FS] = None,
        logs_filter: Optional[Filter] = None,
        logs_handler_factory: _HandlerFactoryType = logging.FileHandler,
    ):
        """
        :param workflow: A prepared instance of EOWorkflow class
        :param execution_kwargs: A list of dictionaries where each dictionary represents execution inputs for the
            workflow. `EOExecutor` will execute the workflow for each of the given dictionaries in the list. The
            content of such dictionary will be used as `input_kwargs` parameter in `EOWorkflow.execution` method.
            Check `EOWorkflow.execution` for definition of a dictionary structure.
        :param execution_names: A list of execution names, which will be shown in execution report
        :param save_logs: Flag used to specify if execution log files should be saved locally on disk
        :param logs_folder: A folder where logs and execution report should be saved. If `filesystem` parameter is
            defined the folder path should be relative to the filesystem.
        :param filesystem: A filesystem object for saving logs and a report.
        :param logs_filter: An instance of a custom filter object that will filter certain logs from being written into
            logs. It works only if save_logs parameter is set to True.
        :param logs_handler_factory: A callable class or function that takes logging path as its only input parameter
            and creates an instance of logging handler object
        """
        self.workflow = workflow
        self.execution_kwargs = self._parse_and_validate_execution_kwargs(execution_kwargs)
        self.execution_names = self._parse_execution_names(execution_names, self.execution_kwargs)
        self.save_logs = save_logs
        self.filesystem, self.logs_folder = self._parse_logs_filesystem(filesystem, logs_folder)
        self.logs_filter = logs_filter
        self.logs_handler_factory = logs_handler_factory

        self.start_time: Optional[dt.datetime] = None
        self.report_folder: Optional[str] = None
        self.general_stats: Dict[str, object] = {}
        self.execution_results: List[WorkflowResults] = []

    @staticmethod
    def _parse_and_validate_execution_kwargs(
        execution_kwargs: Sequence[Dict[EONode, Dict[str, object]]]
    ) -> List[Dict[EONode, Dict[str, object]]]:
        """Parses and validates execution arguments provided by user and raises an error if something is wrong."""
        if not isinstance(execution_kwargs, (list, tuple)):
            raise ValueError("Parameter 'execution_kwargs' should be a list.")

        for input_kwargs in execution_kwargs:
            EOWorkflow.validate_input_kwargs(input_kwargs)

        return [input_kwargs or {} for input_kwargs in execution_kwargs]

    @staticmethod
    def _parse_execution_names(execution_names: Optional[List[str]], execution_kwargs: Sequence) -> List[str]:
        """Parses a list of execution names."""
        if execution_names is None:
            return [str(num) for num in range(1, len(execution_kwargs) + 1)]

        if not isinstance(execution_names, (list, tuple)) or len(execution_names) != len(execution_kwargs):
            raise ValueError(
                "Parameter 'execution_names' has to be a list of the same size as the list of execution arguments."
            )
        return execution_names

    @staticmethod
    def _parse_logs_filesystem(filesystem: Optional[FS], logs_folder: str) -> Tuple[FS, str]:
        """Ensures a filesystem and a file path relative to it."""
        if filesystem is None:
            return get_base_filesystem_and_path(logs_folder)
        return filesystem, logs_folder

    def run(self, workers: int = 1, multiprocess: bool = True) -> List[WorkflowResults]:
        """Runs the executor with n workers.

        :param workers: Maximum number of workflows which will be executed in parallel. Default value is `1` which will
            execute workflows consecutively. If set to `None` the number of workers will be the number of processors
            of the system.
        :param multiprocess: If `True` it will use `concurrent.futures.ProcessPoolExecutor` which will distribute
            workflow executions among multiple processors. If `False` it will use
            `concurrent.futures.ThreadPoolExecutor` which will distribute workflow among multiple threads.
            However, even when `multiprocess=False`, tasks from workflow could still be using multiple processors.
            This parameter is used especially because certain task cannot run with
            `concurrent.futures.ProcessPoolExecutor`.
            In case of `workers=1` this parameter is ignored and workflows will be executed consecutively.
        :return: A list of EOWorkflow results
        """
        self.start_time = dt.datetime.now()
        self.report_folder = fs.path.combine(
            self.logs_folder, f'eoexecution-report-{self.start_time.strftime("%Y_%m_%d-%H_%M_%S")}'
        )
        if self.save_logs:
            self.filesystem.makedirs(self.report_folder, recreate=True)

        log_paths = self.get_log_paths(full_path=True) if self.save_logs else [None] * len(self.execution_kwargs)

        filter_logs_by_thread = not multiprocess and workers > 1
        processing_type = self._get_processing_type(workers, multiprocess)
        processing_args = [
            _ProcessingData(
                workflow=self.workflow,
                workflow_kwargs=workflow_kwargs,
                log_path=log_path,
                filter_logs_by_thread=filter_logs_by_thread,
                logs_filter=self.logs_filter,
                logs_handler_factory=self.logs_handler_factory,
            )
            for workflow_kwargs, log_path in zip(self.execution_kwargs, log_paths)
        ]

        full_execution_results = self._run_execution(processing_args, workers, processing_type)

        self.execution_results = [results.drop_outputs() for results in full_execution_results]
        self.general_stats = self._prepare_general_stats(workers, processing_type)

        return full_execution_results

    @staticmethod
    def _get_processing_type(workers: int, multiprocess: bool) -> _ProcessingType:
        """Decides processing type according to parameters."""
        if workers == 1:
            return _ProcessingType.SINGLE_PROCESS
        if multiprocess:
            return _ProcessingType.MULTIPROCESSING
        return _ProcessingType.MULTITHREADING

    def _run_execution(
        self, processing_args: List[_ProcessingData], workers: int, processing_type: _ProcessingType
    ) -> List[WorkflowResults]:
        """Runs the execution for each item of processing_args list."""
        if processing_type is _ProcessingType.SINGLE_PROCESS:
            return list(tqdm(map(self._execute_workflow, processing_args), total=len(processing_args)))

        if processing_type is _ProcessingType.MULTITHREADING:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as thread_executor:
                return submit_and_monitor_execution(thread_executor, self._execute_workflow, processing_args)

        # pylint: disable=global-statement
        global MULTIPROCESSING_LOCK
        try:
            MULTIPROCESSING_LOCK = multiprocessing.Manager().Lock()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as process_executor:
                return submit_and_monitor_execution(process_executor, self._execute_workflow, processing_args)
        finally:
            MULTIPROCESSING_LOCK = None

    @classmethod
    def _try_add_logging(
        cls,
        log_path: Optional[str],
        filter_logs_by_thread: bool,
        logs_filter: Optional[Filter],
        logs_handler_factory: _HandlerFactoryType,
    ) -> Tuple[Optional[Logger], Optional[Handler]]:
        """Adds a handler to a logger and returns them both. In case this fails it shows a warning."""
        if log_path:
            try:
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
                handler = cls._build_log_handler(log_path, filter_logs_by_thread, logs_filter, logs_handler_factory)
                logger.addHandler(handler)
                return logger, handler
            except BaseException as exception:
                warnings.warn(f"Failed to start logging with exception: {repr(exception)}", category=EORuntimeWarning)

        return None, None

    @classmethod
    def _try_remove_logging(cls, log_path: Optional[str], logger: Optional[Logger], handler: Optional[Handler]):
        """Removes a handler from a logger in case that handler exists."""
        if log_path and logger and handler:
            try:
                handler.close()
                logger.removeHandler(handler)
            except BaseException as exception:
                warnings.warn(f"Failed to end logging with exception: {repr(exception)}", category=EORuntimeWarning)

    @classmethod
    def _execute_workflow(cls, data: _ProcessingData) -> WorkflowResults:
        """Handles a single execution of a workflow."""
        logger, handler = cls._try_add_logging(
            data.log_path, data.filter_logs_by_thread, data.logs_filter, data.logs_handler_factory
        )

        results = data.workflow.execute(data.workflow_kwargs, raise_errors=False)

        cls._try_remove_logging(data.log_path, logger, handler)
        return results

    @staticmethod
    def _build_log_handler(
        log_path: str,
        filter_logs_by_thread: bool,
        logs_filter: Optional[Filter],
        logs_handler_factory: _HandlerFactoryType,
    ) -> Handler:
        """Provides object which handles logs."""
        handler = logs_handler_factory(log_path)

        if not handler.formatter:
            formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
            handler.setFormatter(formatter)

        if filter_logs_by_thread:
            handler.addFilter(LogFileFilter(threading.currentThread().getName()))

        if logs_filter:
            handler.addFilter(logs_filter)

        return handler

    def _prepare_general_stats(self, workers: int, processing_type: _ProcessingType) -> Dict[str, object]:
        """Prepares a dictionary with a general statistics about executions."""
        failed_count = sum(results.workflow_failed() for results in self.execution_results)
        return {
            self.STATS_START_TIME: self.start_time,
            self.STATS_END_TIME: dt.datetime.now(),
            "finished": len(self.execution_results) - failed_count,
            "failed": failed_count,
            "processing_type": processing_type.value,
            "workers": workers,
        }

    def get_successful_executions(self) -> List[int]:
        """Returns a list of IDs of successful executions. The IDs are integers from interval
        `[0, len(execution_kwargs) - 1]`, sorted in increasing order.

        :return: List of successful execution IDs
        """
        return [idx for idx, results in enumerate(self.execution_results) if not results.workflow_failed()]

    def get_failed_executions(self) -> List[int]:
        """Returns a list of IDs of failed executions. The IDs are integers from interval
        `[0, len(execution_kwargs) - 1]`, sorted in increasing order.

        :return: List of failed execution IDs
        """
        return [idx for idx, results in enumerate(self.execution_results) if results.workflow_failed()]

    def get_report_path(self, full_path: bool = True) -> str:
        """Returns the filename and file path of the report.

        :param full_path: A flag to specify if it should return full absolute paths or paths relative to the
            filesystem object.
        :return: Report filename
        """
        if self.report_folder is None:
            raise RuntimeError("Executor has to be run before the report path is created.")
        report_path = fs.path.combine(self.report_folder, self.REPORT_FILENAME)
        if full_path:
            return get_full_path(self.filesystem, report_path)
        return report_path

    def make_report(self, include_logs: bool = True):
        """Makes a html report and saves it into the same folder where logs are stored.

        :param include_logs: If `True` log files will be loaded into the report file. If `False` they will be just
            referenced with a link to a log file. In case of a very large number of executions it is recommended that
            this parameter is set to `False` to avoid compiling a too large report file.
        """
        # pylint: disable=import-outside-toplevel,raise-missing-from
        try:
            from eolearn.visualization.eoexecutor import EOExecutorVisualization
        except ImportError:
            raise RuntimeError(
                "Subpackage eo-learn-visualization has to be installed in order to create EOExecutor reports."
            )

        return EOExecutorVisualization(self).make_report(include_logs=include_logs)

    def get_log_paths(self, full_path: bool = True) -> List[str]:
        """Returns a list of file paths containing logs.

        :param full_path: A flag to specify if it should return full absolute paths or paths relative to the
            filesystem object.
        :return: A list of paths to log files.
        """
        if self.report_folder is None:
            raise RuntimeError("Executor has to be run before log paths are created.")
        log_paths = [fs.path.combine(self.report_folder, f"eoexecution-{name}.log") for name in self.execution_names]
        if full_path:
            return [get_full_path(self.filesystem, path) for path in log_paths]
        return log_paths

    def read_logs(self) -> List[Optional[str]]:
        """Loads the content of log files if logs have been saved."""
        if not self.save_logs:
            return [None] * len(self.execution_kwargs)

        log_paths = self.get_log_paths(full_path=False)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self._read_log_file, log_paths))

    def _read_log_file(self, log_path: str) -> str:
        """Read a content of a log file."""
        try:
            with self.filesystem.open(log_path, "r") as file_handle:
                return file_handle.read()
        except BaseException as exception:
            warnings.warn(f"Failed to load logs with exception: {repr(exception)}", category=EORuntimeWarning)
            return "Failed to load logs"


def submit_and_monitor_execution(
    executor: concurrent.futures.Executor,
    function: Callable[[_InputType], _OutputType],
    execution_params: Iterable[_InputType],
) -> List[_OutputType]:
    """Performs the execution parallelization and monitors the process using a progress bar.

    :param executor: An object that performs parallelization.
    :param function: A function to be parallelized.
    :param execution_params: Each element in a sequence are parameters for a single call of `function`.
    :return: A list of results in the same order as input parameters given by `executor_params`.
    """
    futures = [executor.submit(function, params) for params in execution_params]
    future_order = {future: i for i, future in enumerate(futures)}

    results: List[Optional[_OutputType]] = [None] * len(futures)
    with tqdm(total=len(futures)) as pbar:
        for future in concurrent.futures.as_completed(futures):
            results[future_order[future]] = future.result()
            pbar.update(1)

    return cast(List[_OutputType], results)


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
