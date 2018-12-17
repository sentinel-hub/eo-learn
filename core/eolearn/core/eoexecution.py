"""
The module handles execution and monitoring of workflows. It enables executing a workflow multiple times and in
parallel. It monitors execution times and handles any error that might occur in the process. At the end it generates a
report which contains summary of the workflow and process of execution.

All this is implemented in EOExecutor class.
"""

import os
import logging
import traceback
import inspect
import warnings
import multiprocessing

import matplotlib.pyplot as plt
import networkx as nx

from base64 import b64encode
from copy import deepcopy
from datetime import datetime
from io import StringIO, BytesIO
from jinja2 import Environment, FileSystemLoader
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters.html import HtmlFormatter

from .eoworkflow import EOWorkflow

LOGGER = logging.getLogger(__file__)


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

    def __init__(self, workflow, execution_args, *, save_logs=False, logs_folder='.', file_path=None):
        self.workflow = workflow
        self.execution_args = self._parse_execution_args(execution_args)
        self.save_logs = save_logs
        self.logs_folder = logs_folder
        if file_path is not None:
            warnings.warn("Parameter 'file_path' has been renamed to 'logs_folder' and will soon be removed. Please "
                          "use parameter 'logs_folder' instead.", DeprecationWarning, stacklevel=2)
            self.logs_folder = file_path

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

    def run(self, workers=1):
        """ Runs the executor with n workers.

        :param workers: Number of parallel processes used in the execution. Default is a single process. If set to
            `None` the number of workers will be the number of processors of the system.
        :type workers: int or None
        """
        self.report_folder = self._get_report_folder()
        if self.save_logs and not os.path.isdir(self.report_folder):
            os.mkdir(self.report_folder)

        execution_num = len(self.execution_args)
        log_paths = [self._get_log_filename(idx) if self.save_logs else None
                     for idx in range(execution_num)]

        processing_args = [(self.workflow, init_args, log_path) for init_args, log_path in zip(self.execution_args,
                                                                                               log_paths)]

        if workers is None or workers > 1:
            with multiprocessing.Pool(processes=workers) as pool:
                self.execution_stats = pool.map(self._execute_workflow, processing_args)
        else:
            self.execution_stats = [self._execute_workflow(process_args) for process_args in processing_args]

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

        stats = {'start_time': datetime.now()}
        try:
            _ = workflow.execute(input_args, monitor=True)
        except BaseException:
            stats['error'] = traceback.format_exc()
        stats['end_time'] = datetime.now()

        if log_path:
            handler.close()
            logger.removeHandler(handler)

        return stats

    @staticmethod
    def _get_log_handler(log_path):
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        return handler

    def _get_report_folder(self):
        return os.path.join(self.logs_folder,
                            'eoexecution-report-{}'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))

    def _get_log_filename(self, execution_nb):
        return os.path.join(self.report_folder, 'eoexecution-{}.log'.format(execution_nb))

    def get_report_filename(self):
        """ Returns the filename and file path of the report

        :return: Report filename
        :rtype: str
        """
        return os.path.join(self.report_folder, self.REPORT_FILENAME)

    def make_report(self):
        """ Makes a html report and saves it into the same folder where logs are stored.
        """
        if self.execution_stats is None:
            raise RuntimeError('Cannot produce a report without running the executor first, check EOExecutor.run '
                               'method')

        if os.environ.get('DISPLAY', '') == '':
            LOGGER.info('No display found, using non-interactive Agg backend')
            plt.switch_backend('Agg')

        dependency_graph = self._create_dependency_graph()
        task_descriptions = self._get_task_descriptions()

        formatter = HtmlFormatter(linenos=True)
        task_source = self._render_task_source(formatter)
        execution_stats = self._render_execution_errors(formatter)

        template = self._get_template()

        html = template.render(dependency_graph=dependency_graph,
                               task_descriptions=task_descriptions,
                               task_source=task_source,
                               execution_stats=execution_stats,
                               execution_logs=self.execution_logs,
                               code_css=formatter.get_style_defs())

        if not os.path.isdir(self.report_folder):
            os.mkdir(self.report_folder)

        with open(self.get_report_filename(), 'w') as fout:
            fout.write(html)

    def _create_dependency_graph(self):
        dot = self.workflow.get_dot()
        dot_file = StringIO()
        dot_file.write(dot.source)
        dot_file.seek(0)

        graph = nx.drawing.nx_pydot.read_dot(dot_file)
        image = BytesIO()
        nx.draw_spectral(graph, with_labels=True)
        plt.savefig(image, format='png')

        return b64encode(image.getvalue()).decode()

    def _get_task_descriptions(self):
        descriptions = []

        for task_id, dependency in self.workflow.uuid_dict.items():
            task = dependency.task
            desc = {
                'title': "{}_{} ({})".format(task.__class__.__name__, task_id[:6], task.__module__),
                'args': task.private_task_config.init_args
            }

            descriptions.append(desc)

        return descriptions

    def _render_task_source(self, formatter):
        lexer = get_lexer_by_name("python", stripall=True)
        sources = {}

        for dep in self.workflow.dependencies:
            task = dep.task
            if task.__module__.startswith("eolearn"):
                continue

            key = "{} ({})".format(task.__class__.__name__, task.__module__)
            if key in sources:
                continue

            try:
                source = inspect.getsource(task.__class__)
                source = highlight(source, lexer, formatter)
            except TypeError:
                # Jupyter notebook does not have __file__ method to collect source code
                # StackOverflow provides no solutions
                # Could be investigated further by looking into Jupyter Notebook source code
                source = 'Cannot collect source code of a task which is not defined in a .py file'

            sources[key] = source

        return sources

    def _render_execution_errors(self, formatter):
        tb_lexer = get_lexer_by_name("py3tb", stripall=True)

        executions = []

        for orig_execution in self.execution_stats:
            execution = deepcopy(orig_execution)

            if 'error' in execution:
                execution['error'] = highlight(execution['error'], tb_lexer, formatter)

            executions.append(execution)

        return executions

    @classmethod
    def _get_template(cls):
        templates_dir = os.path.join(os.path.dirname(__file__), 'report_templates')
        env = Environment(loader=FileSystemLoader(templates_dir))
        env.filters['datetime'] = cls._format_datetime
        env.globals.update(timedelta=cls._format_timedelta)
        template = env.get_template(cls.REPORT_FILENAME)

        return template

    @staticmethod
    def _format_datetime(value):
        return value.strftime('%X %x %Z')

    @staticmethod
    def _format_timedelta(value1, value2):
        return str(value2 - value1)
