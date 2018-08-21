"""
The module handles execution and monitoring of workflows. It enables executing a workflow multiple times and in
parallel. It monitors execution times and handles any error that might occur in the process. At the end it generates a
report which contains summary of the workflow and process of execution.

All this is implemented in EOExecutor class.
"""

import os
import logging
import concurrent.futures
import traceback
import inspect

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



LOGGER = logging.getLogger(__file__)


class EOExecutor:
    """Simultaneously executes a workflow with different input arguments.

    Can also create a html report on executions.

    :type workflow: EOWorkflow
    :type execution_args: list(dict)
    """
    REPORT_FILENAME = 'report.html'

    def __init__(self, workflow, execution_args, save_logs=True, file_path='.'):
        self.workflow = workflow
        self.execution_args = execution_args
        self.save_logs = save_logs
        self.report_folder = self._get_report_folder(file_path)

        self.execution_logs = None
        self.execution_stats = None

    def run(self, workers=1):
        """Runs the executor with n workers.

        :type workers: int
        """
        if self.save_logs and not os.path.isdir(self.report_folder):
            os.mkdir(self.report_folder)

        log_paths = [self._get_log_filename(idx) if self.save_logs else None
                     for idx in range(len(self.execution_args))]
        self.execution_logs = [None] * len(self.execution_args)
        self.execution_stats = [None] * len(self.execution_args)

        if workers > 1:
            future2idx = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for idx, fn_args in enumerate(zip(self.execution_args, log_paths)):
                    future = executor.submit(self._execute_workflow, self.workflow, *fn_args)
                    future2idx[future] = idx

            for future in concurrent.futures.as_completed(future2idx):
                idx = future2idx[future]
                self.execution_stats[idx] = future.result()
        else:
            for idx, fn_args in enumerate(zip(self.execution_args, log_paths)):
                self.execution_stats[idx] = self._execute_workflow(self.workflow, *fn_args)

        if self.save_logs:
            for idx, log_path in enumerate(log_paths):
                with open(log_path) as fin:
                    self.execution_logs[idx] = fin.read()

    @classmethod
    def _execute_workflow(cls, workflow, input_args, log_path):
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
            logger.removeHandler(handler)

        return stats

    @staticmethod
    def _get_log_handler(log_path):
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        return handler

    @staticmethod
    def _get_report_folder(file_path):
        return os.path.join(file_path, 'eoexecution-report-{}'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))

    def _get_log_filename(self, execution_nb):
        return os.path.join(self.report_folder, 'eoexecution-{}.log'.format(execution_nb))

    def _get_report_filename(self):
        return os.path.join(self.report_folder, self.REPORT_FILENAME)

    def make_report(self):
        """Makes a html report in the dir where logs are stored."""
        if self.execution_stats is None:
            raise Exception('First run the executor')

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

        with open(self._get_report_filename(), 'w') as fout:
            fout.write(html)

        return html

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
                'title': "{}_{} ({})".format(task.__class__.__name__, task_id[:6], task.__module__)
            }

            if hasattr(task, 'init_args'):
                desc['args'] = task.init_args

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
            except TypeError:
                source = traceback.format_exc()

            sources[key] = highlight(source, lexer, formatter)

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
