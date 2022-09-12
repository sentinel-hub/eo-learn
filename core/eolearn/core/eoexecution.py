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
import inspect
import logging
import sys
import threading
import warnings
from dataclasses import dataclass
from logging import FileHandler, Filter, Handler, Logger
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import fs
from fs.base import FS

from .eonode import EONode
from .eoworkflow import EOWorkflow, WorkflowResults
from .exceptions import EORuntimeWarning
from .utils.fs import get_base_filesystem_and_path, get_full_path, pickle_fs, unpickle_fs
from .utils.logging import LogFileFilter
from .utils.parallelize import _decide_processing_type, _ProcessingType, parallelize

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol  # pylint: disable=ungrouped-imports

if TYPE_CHECKING:
    from eolearn.visualization.eoexecutor import EOExecutorVisualization


class _HandlerWithFsFactoryType(Protocol):
    """Type definition for a callable that accepts a path and a filesystem object"""

    def __call__(self, path: str, filesystem: FS, **kwargs: Any) -> Handler:
        ...


# pylint: disable=invalid-name
_HandlerFactoryType = Union[Callable[[str], Handler], _HandlerWithFsFactoryType]


@dataclass(frozen=True)
class _ProcessingData:
    """Data to be used in EOExecutor processing. This will be passed to a process pool, so everything has to be
    serializable with pickle."""

    workflow: EOWorkflow
    workflow_kwargs: Dict[EONode, Dict[str, object]]
    pickled_filesystem: bytes
    log_path: Optional[str]
    filter_logs_by_thread: bool
    logs_filter: Optional[Filter]
    logs_handler_factory: _HandlerFactoryType


@dataclass(frozen=True)
class _ExecutionRunParams:
    """Parameters that are used during execution run."""

    workers: Optional[int]
    multiprocess: bool
    tqdm_kwargs: Dict[str, Any]


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
        logs_handler_factory: _HandlerFactoryType = FileHandler,
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
        :param logs_handler_factory: A callable class or function that initializes an instance of a logging `Handler`
            object. Its signature should support one of the following options:

            - A single parameter describing a full path to the log file.
            - Parameters `path` and `filesystem` where path to the log file is relative to the given `filesystem`
              object.

            The 2nd option is chosen only if `filesystem` parameter exists in the signature.
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

    def run(self, workers: Optional[int] = 1, multiprocess: bool = True, **tqdm_kwargs: Any) -> List[WorkflowResults]:
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
        :param tqdm_kwargs: Keyword arguments that will be propagated to `tqdm` progress bar.
        :return: A list of EOWorkflow results
        """
        self.start_time = dt.datetime.now()
        self.report_folder = fs.path.combine(
            self.logs_folder, f'eoexecution-report-{self.start_time.strftime("%Y_%m_%d-%H_%M_%S")}'
        )
        if self.save_logs:
            self.filesystem.makedirs(self.report_folder, recreate=True)

        log_paths: Sequence[Optional[str]]
        if self.save_logs:
            log_paths = self.get_log_paths(full_path=False)
        else:
            log_paths = [None] * len(self.execution_kwargs)

        filter_logs_by_thread = not multiprocess and workers is not None and workers > 1
        processing_args = [
            _ProcessingData(
                workflow=self.workflow,
                workflow_kwargs=workflow_kwargs,
                pickled_filesystem=pickle_fs(self.filesystem),
                log_path=log_path,
                filter_logs_by_thread=filter_logs_by_thread,
                logs_filter=self.logs_filter,
                logs_handler_factory=self.logs_handler_factory,
            )
            for workflow_kwargs, log_path in zip(self.execution_kwargs, log_paths)
        ]
        run_params = _ExecutionRunParams(workers=workers, multiprocess=multiprocess, tqdm_kwargs=tqdm_kwargs)
        full_execution_results = self._run_execution(processing_args, run_params)

        self.execution_results = [results.drop_outputs() for results in full_execution_results]
        processing_type = self._get_processing_type(workers=workers, multiprocess=multiprocess)
        self.general_stats = self._prepare_general_stats(workers, processing_type)

        return full_execution_results

    @classmethod
    def _run_execution(
        cls, processing_args: List[_ProcessingData], run_params: _ExecutionRunParams
    ) -> List[WorkflowResults]:
        """Parallelizes the execution for each item of processing_args list."""
        return parallelize(
            cls._execute_workflow,
            processing_args,
            workers=run_params.workers,
            multiprocess=run_params.multiprocess,
            **run_params.tqdm_kwargs,
        )

    @classmethod
    def _try_add_logging(
        cls,
        log_path: Optional[str],
        pickled_filesystem: bytes,
        filter_logs_by_thread: bool,
        logs_filter: Optional[Filter],
        logs_handler_factory: _HandlerFactoryType,
    ) -> Tuple[Optional[Logger], Optional[Handler]]:
        """Adds a handler to a logger and returns them both. In case this fails it shows a warning."""
        if log_path:
            try:
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
                handler = cls._build_log_handler(
                    log_path, pickled_filesystem, filter_logs_by_thread, logs_filter, logs_handler_factory
                )
                logger.addHandler(handler)
                return logger, handler
            except BaseException as exception:
                warnings.warn(f"Failed to start logging with exception: {repr(exception)}", category=EORuntimeWarning)

        return None, None

    @classmethod
    def _try_remove_logging(cls, log_path: Optional[str], logger: Optional[Logger], handler: Optional[Handler]) -> None:
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
            data.log_path,
            data.pickled_filesystem,
            data.filter_logs_by_thread,
            data.logs_filter,
            data.logs_handler_factory,
        )

        results = data.workflow.execute(data.workflow_kwargs, raise_errors=False)

        cls._try_remove_logging(data.log_path, logger, handler)
        return results

    @staticmethod
    def _build_log_handler(
        log_path: str,
        pickled_filesystem: bytes,
        filter_logs_by_thread: bool,
        logs_filter: Optional[Filter],
        logs_handler_factory: _HandlerFactoryType,
    ) -> Handler:
        """Provides object which handles logs."""
        filesystem = unpickle_fs(pickled_filesystem)

        factory_signature = inspect.signature(logs_handler_factory)
        if "filesystem" in factory_signature.parameters:
            handler = logs_handler_factory(log_path, filesystem=filesystem)  # type: ignore[call-arg]
        else:
            full_path = get_full_path(filesystem, log_path)
            handler = logs_handler_factory(full_path)  # type: ignore[call-arg]

        if not handler.formatter:
            formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
            handler.setFormatter(formatter)

        if filter_logs_by_thread:
            handler.addFilter(LogFileFilter(threading.current_thread().name))

        if logs_filter:
            handler.addFilter(logs_filter)

        return handler

    @staticmethod
    def _get_processing_type(workers: Optional[int], multiprocess: bool) -> _ProcessingType:
        """Provides a type of processing according to given parameters."""
        return _decide_processing_type(workers=workers, multiprocess=multiprocess)

    def _prepare_general_stats(self, workers: Optional[int], processing_type: _ProcessingType) -> Dict[str, object]:
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

    def make_report(self, include_logs: bool = True) -> "EOExecutorVisualization":
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
