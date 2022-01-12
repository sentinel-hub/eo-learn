"""
The module handles execution and monitoring of workflows. It enables executing a workflow multiple times and in
parallel. It monitors execution times and handles any error that might occur in the process. At the end it generates a
report which contains summary of the workflow and process of execution.

All this is implemented in EOExecutor class.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os
import logging
import threading
import concurrent.futures
import datetime as dt
import multiprocessing
import warnings
from enum import Enum
from logging import Filter
from typing import List, Dict, Optional

from tqdm.auto import tqdm

from .eonode import EONode
from .eoworkflow import EOWorkflow, WorkflowResults
from .exceptions import EORuntimeWarning
from .utilities import LogFileFilter

LOGGER = logging.getLogger(__name__)

MULTIPROCESSING_LOCK = None


class _ProcessingType(Enum):
    """ Type of EOExecutor processing
    """
    SINGLE_PROCESS = 'single process'
    MULTIPROCESSING = 'multiprocessing'
    MULTITHREADING = 'multithreading'
    RAY = 'ray'


class EOExecutor:
    """ Simultaneously executes a workflow with different input arguments. In the process it monitors execution and
    handles errors. It can also save logs and create a html report about each execution.
    """

    REPORT_FILENAME = 'report.html'
    STATS_START_TIME = 'start_time'
    STATS_END_TIME = 'end_time'

    def __init__(self, workflow: EOWorkflow, execution_args: List[Dict[EONode, Dict[str, object]]], *,
                 save_logs: bool = False, logs_folder: str = '.', logs_filter: Optional[Filter] = None,
                 execution_names: Optional[List[str]] = None):
        """
        :param workflow: A prepared instance of EOWorkflow class
        :param execution_args: A list of dictionaries where each dictionary represents execution inputs for the
            workflow. `EOExecutor` will execute the workflow for each of the given dictionaries in the list. The
            content of such dictionary will be used as `input_args` parameter in `EOWorkflow.execution` method.
            Check `EOWorkflow.execution` for definition of a dictionary structure.
        :param save_logs: Flag used to specify if execution log files should be saved locally on disk
        :param logs_folder: A folder where logs and execution report should be saved
        :param logs_filter: An instance of a custom filter object that will filter certain logs from being written into
            logs. It works only if save_logs parameter is set to True.
        :param execution_names: A list of execution names, which will be shown in execution report
        """
        self.workflow = workflow
        self.execution_args = self._parse_and_validate_execution_args(execution_args)
        self.save_logs = save_logs
        self.logs_folder = os.path.abspath(logs_folder)
        self.logs_filter = logs_filter
        self.execution_names = self._parse_execution_names(execution_names, self.execution_args)

        self.start_time = None
        self.report_folder = None
        self.general_stats = {}
        self.execution_logs = None
        self.execution_results = None

    @staticmethod
    def _parse_and_validate_execution_args(execution_args: object) -> List[Dict[EONode, Dict[str, object]]]:
        """ Parses and validates execution arguments provided by user and raises an error if something is wrong
        """
        if not isinstance(execution_args, (list, tuple)):
            raise ValueError("Parameter 'execution_args' should be a list")

        for input_kwargs in execution_args:
            EOWorkflow.validate_input_kwargs(input_kwargs)

        return [input_kwargs or {} for input_kwargs in execution_args]

    @staticmethod
    def _parse_execution_names(execution_names, execution_args: Optional[List[str]]) -> List[str]:
        """ Parses a list of execution names
        """
        if execution_names is None:
            return [str(num) for num in range(1, len(execution_args) + 1)]

        if not isinstance(execution_names, (list, tuple)) or len(execution_names) != len(execution_args):
            raise ValueError("Parameter 'execution_names' has to be a list of the same size as the list of "
                             "execution arguments")
        return execution_names

    def run(self, workers: int = 1, multiprocess: bool = True) -> List[WorkflowResults]:
        """ Runs the executor with n workers.

        :param workers: Maximum number of workflows which will be executed in parallel. Default value is `1` which will
            execute workflows consecutively. If set to `None` the number of workers will be the number of processors
            of the system.
        :param multiprocess: If `True` it will use `concurrent.futures.ProcessPoolExecutor` which will distribute
            workflow executions among multiple processors. If `False` it will use
            `concurrent.futures.ThreadPoolExecutor` which will distribute workflow among multiple threads.
            However even when `multiprocess=False`, tasks from workflow could still be using multiple processors.
            This parameter is used especially because certain task cannot run with
            `concurrent.futures.ProcessPoolExecutor`.
            In case of `workers=1` this parameter is ignored and workflows will be executed consecutively.
        :return: A list of EOWorkflow results
        """
        self.start_time = dt.datetime.now()
        self.report_folder = self._get_report_folder()
        if self.save_logs and not os.path.isdir(self.report_folder):
            os.makedirs(self.report_folder)

        log_paths = self._get_log_paths()

        filter_logs_by_thread = not multiprocess and workers > 1
        processing_args = [(self.workflow, init_args, log_path, filter_logs_by_thread, self.logs_filter)
                           for init_args, log_path in zip(self.execution_args, log_paths)]
        processing_type = self._get_processing_type(workers, multiprocess)

        full_execution_results = self._run_execution(processing_args, workers, processing_type)

        self.execution_results = [results.drop_outputs() for results in full_execution_results]
        self.general_stats = self._prepare_general_stats(workers, processing_type)

        self.execution_logs = [None] * len(self.execution_args)
        if self.save_logs:
            for idx, log_path in enumerate(log_paths):
                with open(log_path) as fin:
                    self.execution_logs[idx] = fin.read()

        return full_execution_results

    @staticmethod
    def _get_processing_type(workers: int, multiprocess: bool) -> _ProcessingType:
        """ Decides processing type according to parameters
        """
        if workers == 1:
            return _ProcessingType.SINGLE_PROCESS
        if multiprocess:
            return _ProcessingType.MULTIPROCESSING
        return _ProcessingType.MULTITHREADING

    def _run_execution(self, processing_args, workers: int, processing_type: _ProcessingType) -> List[WorkflowResults]:
        """ Runs the execution an each item of processing_args list
        """
        if processing_type is _ProcessingType.SINGLE_PROCESS:
            return list(tqdm(map(self._execute_workflow, processing_args), total=len(processing_args)))

        if processing_type is _ProcessingType.MULTITHREADING:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                return list(tqdm(executor.map(self._execute_workflow, processing_args), total=len(processing_args)))

        # pylint: disable=global-statement
        global MULTIPROCESSING_LOCK
        try:
            MULTIPROCESSING_LOCK = multiprocessing.Manager().Lock()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                result = list(tqdm(executor.map(self._execute_workflow, processing_args), total=len(processing_args)))
        finally:
            MULTIPROCESSING_LOCK = None

        return result

    @classmethod
    def _try_add_logging(cls, log_path, filter_logs_by_thread, logs_filter):
        """ Adds a handler to a logger and returns them both. In case this fails it shows a warning.
        """
        if log_path:
            try:
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
                handler = cls._get_log_handler(log_path, filter_logs_by_thread, logs_filter)
                logger.addHandler(handler)
                return logger, handler
            except BaseException as exception:
                warnings.warn(f'Failed to start logging with exception: {repr(exception)}', category=EORuntimeWarning)

        return None, None

    @classmethod
    def _try_remove_logging(cls, log_path, logger, handler):
        """ Removes a handler from a logger in case that handler exists.
        """
        if log_path and logger:
            try:
                handler.close()
                logger.removeHandler(handler)
            except BaseException as exception:
                warnings.warn(f'Failed to end logging with exception: {repr(exception)}', category=EORuntimeWarning)

    @classmethod
    def _execute_workflow(cls, process_args):
        """ Handles a single execution of a workflow
        """
        workflow, input_args, log_path, filter_logs_by_thread, logs_filter = process_args
        logger, handler = cls._try_add_logging(log_path, filter_logs_by_thread, logs_filter)

        results = workflow.execute(input_args, raise_errors=False)

        if logger:
            status = 'failed' if results.workflow_failed() else 'finished'
            message = f'EOWorkflow execution {status}!'
            logger.debug(message)

        cls._try_remove_logging(log_path, logger, handler)

        return results

    @staticmethod
    def _get_log_handler(log_path, filter_logs_by_thread, logs_filter):
        """ Provides object which handles logs
        """
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        if filter_logs_by_thread:
            handler.addFilter(LogFileFilter(threading.currentThread().getName()))

        if logs_filter:
            handler.addFilter(logs_filter)

        return handler

    def _prepare_general_stats(self, workers, processing_type):
        """ Prepares a dictionary with a general statistics about executions
        """
        failed_count = sum(results.workflow_failed() for results in self.execution_results)
        return {
            self.STATS_START_TIME: self.start_time,
            self.STATS_END_TIME: dt.datetime.now(),
            'finished': len(self.execution_results) - failed_count,
            'failed': failed_count,
            'processing_type': processing_type.value,
            'workers': workers
        }

    def _get_report_folder(self):
        """ Returns file path of folder where report will be saved
        """
        return os.path.join(self.logs_folder,
                            f'eoexecution-report-{self.start_time.strftime("%Y_%m_%d-%H_%M_%S")}')

    def _get_log_paths(self):
        """ Returns a list of file paths containing logs
        """
        if self.save_logs:
            return [os.path.join(self.report_folder, f'eoexecution-{name}.log')
                    for name in self.execution_names]

        return [None] * len(self.execution_names)

    def get_successful_executions(self):
        """ Returns a list of IDs of successful executions. The IDs are integers from interval
        `[0, len(execution_args) - 1]`, sorted in increasing order.

        :return: List of successful execution IDs
        :rtype: list(int)
        """
        return [idx for idx, results in enumerate(self.execution_results) if not results.workflow_failed()]

    def get_failed_executions(self):
        """ Returns a list of IDs of failed executions. The IDs are integers from interval
        `[0, len(execution_args) - 1]`, sorted in increasing order.

        :return: List of failed execution IDs
        :rtype: list(int)
        """
        return [idx for idx, results in enumerate(self.execution_results) if results.workflow_failed()]

    def get_report_filename(self):
        """ Returns the filename and file path of the report

        :return: Report filename
        :rtype: str
        """
        return os.path.join(self.report_folder, self.REPORT_FILENAME)

    def make_report(self):
        """ Makes a html report and saves it into the same folder where logs are stored.
        """
        # pylint: disable=import-outside-toplevel,raise-missing-from
        try:
            from eolearn.visualization import EOExecutorVisualization
        except ImportError:
            raise RuntimeError('Subpackage eo-learn-visualization has to be installed in order to create EOExecutor '
                               'reports')

        return EOExecutorVisualization(self).make_report()


def execute_with_mp_lock(execution_function, *args, **kwargs):
    """ A helper utility function that executes a given function with multiprocessing lock if the process is being
    executed in a multi-processing mode

    :param execution_function: A function
    :param args: Function's positional arguments
    :param kwargs: Function's keyword arguments
    :return: Function's results
    """
    if multiprocessing.current_process().name == 'MainProcess' or MULTIPROCESSING_LOCK is None:
        return execution_function(*args, **kwargs)

    # pylint: disable=not-context-manager
    with MULTIPROCESSING_LOCK:
        return execution_function(*args, **kwargs)
