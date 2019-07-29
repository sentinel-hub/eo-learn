"""
The module handles execution and monitoring of workflows. It enables executing a workflow multiple times and in
parallel. It monitors execution times and handles any error that might occur in the process. At the end it generates a
report which contains summary of the workflow and process of execution.

All this is implemented in EOExecutor class.
"""

import os
import logging
import traceback
import concurrent.futures
import datetime as dt

from tqdm.auto import tqdm

from .eoworkflow import EOWorkflow

LOGGER = logging.getLogger(__name__)


class EOExecutor:
    """ Simultaneously executes a workflow with different input arguments. In the process it monitors execution and
    handles errors. It can also save logs and create a html report about each execution.

    :param workflow: A prepared instance of EOWorkflow class
    :type workflow: EOWorkflow
    :param execution_args: A list of dictionaries where each dictionary represents execution inputs for the workflow.
        `EOExecutor` will execute the workflow for each of the given dictionaries in the list. The content of such
        dictionary will be used as `input_args` parameter in `EOWorkflow.execution` method. Check `EOWorkflow.execution`
        for definition of a dictionary structure.
    :type execution_args: list(dict(EOTask: dict(str: object) or tuple(object)))
    :param save_logs: Flag used to specify if execution log files should be saved locally on disk
    :type save_logs: bool
    :param logs_folder: A folder where logs and execution report should be saved
    :type logs_folder: str
    """
    REPORT_FILENAME = 'report.html'

    STATS_START_TIME = 'start_time'
    STATS_END_TIME = 'end_time'
    STATS_ERROR = 'error'

    def __init__(self, workflow, execution_args, *, save_logs=False, logs_folder='.'):
        self.workflow = workflow
        self.execution_args = self._parse_execution_args(execution_args)
        self.save_logs = save_logs
        self.logs_folder = logs_folder

        self.report_folder = None
        self.execution_logs = None
        self.execution_stats = None

    @staticmethod
    def _parse_execution_args(execution_args):
        """ Parses execution arguments provided by user and raises an error if something is wrong
        """
        if not isinstance(execution_args, (list, tuple)):
            raise ValueError("Parameter 'execution_args' should be a list")

        return [EOWorkflow.parse_input_args(input_args) for input_args in execution_args]

    def run(self, workers=1, multiprocess=True):
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
        """
        self.report_folder = self._get_report_folder()
        if self.save_logs and not os.path.isdir(self.report_folder):
            os.mkdir(self.report_folder)

        execution_num = len(self.execution_args)
        log_paths = [self._get_log_filename(idx) if self.save_logs else None
                     for idx in range(execution_num)]

        processing_args = [(self.workflow, init_args, log_path) for init_args, log_path in zip(self.execution_args,
                                                                                               log_paths)]

        if workers == 1:
            self.execution_stats = list(tqdm(map(self._execute_workflow, processing_args), total=len(processing_args)))
        else:
            pool_executor_class = concurrent.futures.ProcessPoolExecutor if multiprocess else \
                concurrent.futures.ThreadPoolExecutor
            with pool_executor_class(max_workers=workers) as executor:
                self.execution_stats = list(tqdm(executor.map(self._execute_workflow, processing_args),
                                                 total=len(processing_args)))

        self.execution_logs = [None] * execution_num
        if self.save_logs:
            for idx, log_path in enumerate(log_paths):
                with open(log_path) as fin:
                    self.execution_logs[idx] = fin.read()

    @classmethod
    def _execute_workflow(cls, process_args):
        """ Handles a single execution of a workflow
        """
        workflow, input_args, log_path = process_args

        if log_path:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            handler = cls._get_log_handler(log_path)
            logger.addHandler(handler)

        stats = {cls.STATS_START_TIME: dt.datetime.now()}
        try:
            _ = workflow.execute(input_args, monitor=True)
        except BaseException:
            stats[cls.STATS_ERROR] = traceback.format_exc()
        stats[cls.STATS_END_TIME] = dt.datetime.now()

        if log_path:
            handler.close()
            logger.removeHandler(handler)

        return stats

    @staticmethod
    def _get_log_handler(log_path):
        """ Provides object which handles logs
        """
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        return handler

    def _get_report_folder(self):
        """ Returns file path of folder where report will be saved
        """
        return os.path.join(self.logs_folder,
                            'eoexecution-report-{}'.format(dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))

    def _get_log_filename(self, execution_nb):
        """ Returns file path of a log file
        """
        return os.path.join(self.report_folder, 'eoexecution-{}.log'.format(execution_nb))

    def get_successful_executions(self):
        """ Returns a list of IDs of successful executions. The IDs are integers from interval
        `[0, len(execution_args) - 1]`, sorted in increasing order.

        :return: List of succesful execution IDs
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
            from eolearn.visualization import EOExecutorVisualization
        except ImportError:
            raise RuntimeError('Subpackage eo-learn-visualization has to be installed in order to create EOExecutor '
                               'reports')

        return EOExecutorVisualization(self).make_report()
