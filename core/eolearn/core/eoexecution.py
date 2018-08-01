import os
import logging
import concurrent.futures
from datetime import datetime
import traceback
import inspect
from io import StringIO, BytesIO
import base64
import copy

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import networkx as nx
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter


if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    plt.switch_backend('Agg')


class EOExecutor:
    """
    Simultaneously executes a workflow with different input arguments.

    Can also create a html report.

    :type workflow: Workflow
    :type executions_args: List[Dict]
    """
    def __init__(self, workflow, executions_args, out_dir=None):
        self.workflow = workflow
        self.executions_args = executions_args

        if out_dir is None:
            out_dir = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        self.out_dir = out_dir
        self.executions_logs = None
        self.executions_info = None

    def run(self, workers=1):
        """
        Run the executor with n workers.

        In a Jupyter Notebook on Windows it raises the following error:
            BrokenProcessPool: A process in the process pool was terminated
            abruptly while the future was running or pending.

        :type workers: int
        """
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        log_paths = []

        for idx in range(len(self.executions_args)):
            log_paths.append(os.path.join(self.out_dir,
                                          "execution-{}.log".format(idx)))

        future2idx = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, (exec_args, log_path) in enumerate(zip(self.executions_args,
                                                            log_paths)):
                future = executor.submit(self._execute_workflow,
                                         self.workflow,
                                         exec_args,
                                         log_path)

                future2idx[future] = idx

            self.executions_logs = [None for _ in range(len(self.executions_args))]
            self.executions_info = [None for _ in range(len(self.executions_args))]

        for future in concurrent.futures.as_completed(future2idx):
            idx = future2idx[future]

            self.executions_info[idx] = future.result()

            with open(log_paths[idx]) as fin:
                self.executions_logs[idx] = fin.read()

    @classmethod
    def _execute_workflow(cls, workflow, input_args, log_path):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = cls._get_log_handler(log_path)
        logger.addHandler(handler)

        info = {'start_time': datetime.now()}

        try:
            _ = workflow.execute(input_args)
        except:
            info['error'] = traceback.format_exc()

        info['end_time'] = datetime.now()

        return info

    @staticmethod
    def _get_log_handler(log_path):
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        return handler

    def create_html_report(self):
        """
        Make a html report in the dir where logs are stored.
        """
        if self.executions_info is None:
            raise Exception('First run the executor')

        dependency_graph = self._create_dependency_graph()
        tasks_info = self._get_tasks_info()

        formatter = HtmlFormatter(linenos=True)
        tasks_source = self._render_tasks_source(formatter)
        executions_info = self._render_executions_error(formatter)

        template = self._get_template()
        html_fpath = os.path.join(self.out_dir, 'report.html')
        html = template.render(dependency_graph=dependency_graph,
                               tasks_info=tasks_info,
                               tasks_source=tasks_source,
                               executions_info=executions_info,
                               executions_logs=self.executions_logs,
                               code_css=formatter.get_style_defs())

        with open(html_fpath, 'w') as fout:
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

        return base64.b64encode(image.getvalue()).decode()

    def _get_tasks_info(self):
        infos = []

        for task_id, task in self.workflow.id2task.items():
            info = {}
            info['title'] = "{}_{} ({})".format(task.__class__.__name__,
                                                task_id[:6],
                                                task.__module__)

            if hasattr(task, 'init_args'):
                info['args'] = task.init_args

            infos.append(info)

        return infos

    def _render_tasks_source(self, formatter):
        lexer = get_lexer_by_name("python", stripall=True)

        sources = {}

        for task in self.workflow.id2task.values():
            if task.__module__.startswith("eolearn"):
                continue

            key = "{} ({})".format(task.__class__.__name__,
                                   task.__module__)

            if key in sources:
                continue

            try:
                source = inspect.getsource(task.__class__)
            except TypeError:
                source = traceback.format_exc()

            sources[key] = highlight(source, lexer, formatter)

        return sources

    def _render_executions_error(self, formatter):
        tb_lexer = get_lexer_by_name("py3tb", stripall=True)

        executions_info = []

        for info_orig in self.executions_info:
            info = copy.deepcopy(info_orig)

            if 'error' in info:
                info['error'] = highlight(info['error'], tb_lexer, formatter)

            executions_info.append(info)

        return executions_info

    @classmethod
    def _get_template(cls):
        templates_dir = os.path.join(os.path.dirname(__file__),
                                     'report_templates')
        env = Environment(loader=FileSystemLoader(templates_dir))

        env.filters['datetime'] = cls._datetimeformat
        env.globals.update(timedelta=cls._timedeltaformat)

        template = env.get_template("report.html")

        return template

    @staticmethod
    def _datetimeformat(value):
        return value.strftime('%X %x %Z')

    @staticmethod
    def _timedeltaformat(value1, value2):
        return str(value2 - value1)
