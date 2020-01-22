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
import traceback
import concurrent.futures
import datetime as dt
import multiprocessing

from tqdm.auto import tqdm

from .eoworkflow import EOWorkflow

from .utilities import LogFileFilter

LOGGER = logging.getLogger(__name__)

try:
    MULTIPROCESSING_LOCK = multiprocessing.Manager().Lock()
except BaseException:
    MULTIPROCESSING_LOCK = None


class EOExecutor:
    """ Simultaneously executes a workflow with different input arguments. In the process it monitors execution and
    handles errors. It can also save logs and create a html report about each execution.
    """
    REPORT_FILENAME = 'report.html'

    STATS_START_TIME = 'start_time'
    STATS_END_TIME = 'end_time'
    STATS_ERROR = 'error'
    RESULTS = 'results'

    def __init__(self, workflow, execution_args, *, save_logs=False, logs_folder='.', execution_names=None):
        """
        :param workflow: A prepared instance of EOWorkflow class
        :type workflow: EOWorkflow
        :param execution_args: A list of dictionaries where each dictionary represents execution inputs for the
            workflow. `EOExecutor` will execute the workflow for each of the given dictionaries in the list. The
            content of such dictionary will be used as `input_args` parameter in `EOWorkflow.execution` method.
            Check `EOWorkflow.execution` for definition of a dictionary structure.
        :type execution_args: list(dict(EOTask: dict(str: object) or tuple(object)))
        :param save_logs: Flag used to specify if execution log files should be saved locally on disk
        :type save_logs: bool
        :param logs_folder: A folder where logs and execution report should be saved
        :type logs_folder: str
        :param execution_names: A list of execution names, which will be shown in execution report
        :type execution_names: list(str) or None
        """
        self.workflow = workflow
        self.execution_args = self._parse_execution_args(execution_args)
        self.save_logs = save_logs
        self.logs_folder = logs_folder
        self.execution_names = self._parse_execution_names(execution_names, self.execution_args)

        self.start_time = None
        self.report_folder = None
        self.general_stats = {}
        self.execution_logs = None
        self.execution_stats = None

    @staticmethod
    def _parse_execution_args(execution_args):
        """ Parses execution arguments provided by user and raises an error if something is wrong
        """
        if not isinstance(execution_args, (list, tuple)):
            raise ValueError("Parameter 'execution_args' should be a list")

        return [EOWorkflow.parse_input_args(input_args) for input_args in execution_args]

    @staticmethod
    def _parse_execution_names(execution_names, execution_args):
        """ Parses a list of execution names
        """
        if execution_names is None:
            return [str(num) for num in range(1, len(execution_args) + 1)]

        if not isinstance(execution_names, (list, tuple)) or len(execution_names) != len(execution_args):
            raise ValueError("Parameter 'execution_names' has to be a list of the same size as the list of "
                             "execution arguments")
        return execution_names

    def run(self, workers=1, multiprocess=True, return_results=False):
        """ Runs the executor with n workers.

        :param workers: Maximum number of workflows which will be executed in parallel. Default value is `1` which will
            execute workflows consecutively. If set to `None` the number of workers will be the number of processors
            of the system.
        :type workers: int or None
        :param multiprocess: If `True` it will use `concurrent.futures.ProcessPoolExecutor` which will distribute
            workflow executions among multiple processors. If `False` it will use
            `concurrent.futures.ThreadPoolExecutor` which will distribute workflow among multiple threads.
            However even when `multiprocess=False`, tasks from workflow could still be using multiple processors.
            This parameter is used especially because certain task cannot run with
            `concurrent.futures.ProcessPoolExecutor`.
            In case of `workers=1` this parameter is ignored and workflows will be executed consecutively.
        :type multiprocess: bool
        :param return_results: If `True` this method will return a list of all results of the execution. Note that
            this might exceed the available memory. By default this parameter is set to `False`.
        :type: bool
        :return: If `return_results` is set to `True` it will return a list of results, otherwise it will return `None`
        :rtype: None or list(eolearn.core.WorkflowResults)
        """
        self.start_time = dt.datetime.now()
        self.report_folder = self._get_report_folder()
        if self.save_logs and not os.path.isdir(self.report_folder):
            os.mkdir(self.report_folder)

        execution_num = len(self.execution_args)
        log_paths = self._get_log_paths()

        processing_args = [(self.workflow, init_args, log_path, return_results)
                           for init_args, log_path in zip(self.execution_args, log_paths)]

        if workers == 1:
            processing_type = 'single process'
            self.execution_stats = list(tqdm(map(self._execute_workflow, processing_args), total=len(processing_args)))
        else:
            if multiprocess:
                pool_executor_class = concurrent.futures.ProcessPoolExecutor
                processing_type = 'multiprocessing'
            else:
                pool_executor_class = concurrent.futures.ThreadPoolExecutor
                processing_type = 'multithreading'

            with pool_executor_class(max_workers=workers) as executor:
                self.execution_stats = list(tqdm(executor.map(self._execute_workflow, processing_args),
                                                 total=len(processing_args)))

        self.general_stats = self._prepare_general_stats(workers, processing_type)

        self.execution_logs = [None] * execution_num
        if self.save_logs:
            for idx, log_path in enumerate(log_paths):
                with open(log_path) as fin:
                    self.execution_logs[idx] = fin.read()

        return [stats.get(self.RESULTS) for stats in self.execution_stats] if return_results else None

    @classmethod
    def _try_add_logging(cls, log_path):
        if log_path:
            try:
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
                handler = cls._get_log_handler(log_path=log_path)
                logger.addHandler(handler)
                return logger, handler
            except BaseException:
                pass

        return None, None

    @classmethod
    def _try_remove_logging(cls, log_path, logger, handler, stats):
        if log_path:
            try:
                logger.info(msg='Pipeline failed.' if cls.STATS_ERROR in stats else 'Pipeline finished.')
                handler.close()
                logger.removeHandler(handler)
            except BaseException:
                pass


    @classmethod
    def _execute_workflow(cls, process_args):
        """ Handles a single execution of a workflow
        """
        workflow, input_args, log_path, return_results = process_args
        logger, handler = cls._try_add_logging(log_path)
        stats = {cls.STATS_START_TIME: dt.datetime.now()}
        try:
            results = workflow.execute(input_args, monitor=True)

            if return_results:
                stats[cls.RESULTS] = results

        except BaseException:
            stats[cls.STATS_ERROR] = traceback.format_exc()
        stats[cls.STATS_END_TIME] = dt.datetime.now()

        cls._try_remove_logging(log_path, logger, handler, stats)

        return stats

    @staticmethod
    def _get_log_handler(log_path):
        """ Provides object which handles logs
        """
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        handler.addFilter(LogFileFilter(thread_name=threading.currentThread().getName()))

        return handler

    def _prepare_general_stats(self, workers, processing_type):
        """ Prepares a dictionary with a general statistics about executions
        """
        failed_count = sum(self.STATS_ERROR in stats for stats in self.execution_stats)
        return {
            self.STATS_START_TIME: self.start_time,
            self.STATS_END_TIME: dt.datetime.now(),
            'finished': len(self.execution_stats) - failed_count,
            'failed': failed_count,
            'processing_type': processing_type,
            'workers': workers
        }

    def _get_report_folder(self):
        """ Returns file path of folder where report will be saved
        """
        return os.path.join(self.logs_folder,
                            'eoexecution-report-{}'.format(self.start_time.strftime("%Y_%m_%d-%H_%M_%S")))

    def _get_log_paths(self):
        """ Returns a list of file paths containing logs
        """
        if self.save_logs:
            return [os.path.join(self.report_folder, 'eoexecution-{}.log'.format(name))
                    for name in self.execution_names]

        return [None] * len(self.execution_names)

    def get_successful_executions(self):
        """ Returns a list of IDs of successful executions. The IDs are integers from interval
        `[0, len(execution_args) - 1]`, sorted in increasing order.

        :return: List of successful execution IDs
        :rtype: list(int)
        """
        return [idx for idx, stats in enumerate(self.execution_stats) if self.STATS_ERROR not in stats]

    def get_failed_executions(self):
        """ Returns a list of IDs of failed executions. The IDs are integers from interval
        `[0, len(execution_args) - 1]`, sorted in increasing order.

        :return: List of failed execution IDs
        :rtype: list(int)
        """
        return [idx for idx, stats in enumerate(self.execution_stats) if self.STATS_ERROR in stats]

    def get_report_filename(self):
        """ Returns the filename and file path of the report

        :return: Report filename
        :rtype: str
        """
        return os.path.join(self.report_folder, self.REPORT_FILENAME)

    def make_report(self):
        """ Makes a html report and saves it into the same folder where logs are stored.
        """
        try:
            # pylint: disable=import-outside-toplevel
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

    with MULTIPROCESSING_LOCK:
        return execution_function(*args, **kwargs)
