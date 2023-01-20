"""
Module with utilities for visualizing EOExecutor

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import base64
import datetime as dt
import importlib
import inspect
import os
import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple, cast

import fs
import graphviz
import matplotlib.pyplot as plt
import pygments
import pygments.formatter
import pygments.lexers
from jinja2 import Environment, FileSystemLoader, Template
from pygments.formatters.html import HtmlFormatter

from eolearn.core import EOExecutor
from eolearn.core.exceptions import EOUserWarning


class EOExecutorVisualization:
    """Class handling EOExecutor visualizations, particularly creating reports"""

    def __init__(self, eoexecutor: EOExecutor):
        """
        :param eoexecutor: An instance of EOExecutor
        """
        self.eoexecutor = eoexecutor

    def make_report(self, include_logs: bool = True) -> None:
        """Makes a html report and saves it into the same folder where logs are stored."""
        if self.eoexecutor.execution_results is None:
            raise RuntimeError(
                "Cannot produce a report without running the executor first, check EOExecutor.run method"
            )

        # These should be set automatically after a run
        start_time = cast(dt.datetime, self.eoexecutor.start_time)
        report_folder = cast(str, self.eoexecutor.report_folder)

        if os.environ.get("DISPLAY", "") == "":
            plt.switch_backend("Agg")

        try:
            dependency_graph = self._create_dependency_graph()
        except graphviz.backend.ExecutableNotFound as ex:
            dependency_graph = None
            warnings.warn(
                (
                    f"{ex}.\nPlease install the system package 'graphviz' (in addition to the python package) to have "
                    "the dependency graph in the final report!"
                ),
                EOUserWarning,
            )

        formatter = HtmlFormatter(linenos=True)

        template = self._get_template()

        execution_log_filenames = [fs.path.basename(log_path) for log_path in self.eoexecutor.get_log_paths()]
        if self.eoexecutor.save_logs:
            execution_logs = self.eoexecutor.read_logs() if include_logs else None
        else:
            execution_logs = ["No logs saved"] * len(self.eoexecutor.execution_kwargs)

        html = template.render(
            title=f"Report {self._format_datetime(start_time)}",
            dependency_graph=dependency_graph,
            general_stats=self.eoexecutor.general_stats,
            exception_stats=self._get_exception_stats(),
            task_descriptions=self._get_node_descriptions(),
            task_sources=self._render_task_sources(formatter),
            execution_results=self.eoexecutor.execution_results,
            execution_tracebacks=self._render_execution_tracebacks(formatter),
            execution_logs=execution_logs,
            execution_log_filenames=execution_log_filenames,
            execution_names=self.eoexecutor.execution_names,
            code_css=formatter.get_style_defs(),
        )
        self.eoexecutor.filesystem.makedirs(report_folder, recreate=True)

        with self.eoexecutor.filesystem.open(self.eoexecutor.get_report_path(full_path=False), "w") as file_handle:
            file_handle.write(html)

    def _create_dependency_graph(self) -> str:
        """Provides an image of dependency graph"""
        dot = self.eoexecutor.workflow.dependency_graph()
        return base64.b64encode(dot.pipe()).decode()

    def _get_exception_stats(self) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
        """Creates aggregated stats about exceptions"""
        formatter = HtmlFormatter()
        lexer = pygments.lexers.get_lexer_by_name("python", stripall=True)

        exception_stats: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))

        for workflow_results in self.eoexecutor.execution_results:
            if not workflow_results.error_node_uid:
                continue

            error_node = workflow_results.stats[workflow_results.error_node_uid]
            exception_str = pygments.highlight(
                f"{error_node.exception.__class__.__name__}: {error_node.exception}", lexer, formatter
            )
            exception_stats[error_node.node_uid][exception_str] += 1

        return self._to_ordered_stats(exception_stats)

    def _to_ordered_stats(
        self, exception_stats: DefaultDict[str, DefaultDict[str, int]]
    ) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
        """Exception stats get ordered by nodes in their execution order in workflows. Exception stats that happen
        for the same node get ordered by number of occurrences in a decreasing order.
        """
        ordered_exception_stats = []
        for node in self.eoexecutor.workflow.get_nodes():
            if node.uid not in exception_stats:
                continue

            node_stats = exception_stats[node.uid]
            ordered_exception_stats.append(
                (node.get_name(), node.uid, sorted(node_stats.items(), key=lambda item: -item[1]))
            )

        return ordered_exception_stats

    def _get_node_descriptions(self) -> List[Dict[str, Any]]:
        """Prepares a list of node names and initialization parameters of their tasks"""
        descriptions = []
        name_counts: Dict[str, int] = defaultdict(lambda: 0)

        for node in self.eoexecutor.workflow.get_nodes():
            node_name = node.get_name(name_counts[node.get_name()])
            name_counts[node.get_name()] += 1

            descriptions.append(
                {
                    "name": f"{node_name} ({node.uid})",
                    "uid": node.uid,
                    "args": {
                        key: value.replace("<", "&lt;").replace(">", "&gt;")  # type: ignore
                        for key, value in node.task.private_task_config.init_args.items()
                    },
                }
            )

        return descriptions

    def _render_task_sources(self, formatter: pygments.formatter.Formatter) -> Dict[str, Any]:
        """Renders source code of EOTasks"""
        lexer = pygments.lexers.get_lexer_by_name("python", stripall=True)
        sources = {}

        for node in self.eoexecutor.workflow.get_nodes():
            task = node.task

            key = f"{task.__class__.__name__} ({task.__module__})"
            if key in sources:
                continue

            source: Any
            if task.__module__.startswith("eolearn"):
                subpackage_name = ".".join(task.__module__.split(".")[:2])
                subpackage = importlib.import_module(subpackage_name)
                subpackage_version = subpackage.__version__ if hasattr(subpackage, "__version__") else "unknown"
                source = subpackage_name, subpackage_version
            else:
                try:
                    source = inspect.getsource(task.__class__)
                    source = pygments.highlight(source, lexer, formatter)
                except TypeError:
                    # Jupyter notebook does not have __file__ method to collect source code
                    # StackOverflow provides no solutions
                    # Could be investigated further by looking into Jupyter Notebook source code
                    source = None

            sources[key] = source

        return sources

    def _render_execution_tracebacks(self, formatter: pygments.formatter.Formatter) -> list:
        """Renders stack traces of those executions which failed"""
        tb_lexer = pygments.lexers.get_lexer_by_name("py3tb", stripall=True)

        tracebacks = []
        for results in self.eoexecutor.execution_results:
            if results.workflow_failed() and results.error_node_uid is not None:
                # second part of above check needed only for typechecking purposes
                failed_node_stats = results.stats[results.error_node_uid]
                traceback = pygments.highlight(failed_node_stats.exception_traceback, tb_lexer, formatter)
            else:
                traceback = None

            tracebacks.append(traceback)

        return tracebacks

    def _get_template(self) -> Template:
        """Loads and sets up a template for report"""
        templates_dir = os.path.join(os.path.dirname(__file__), "report_templates")
        env = Environment(loader=FileSystemLoader(templates_dir))
        env.filters["datetime"] = self._format_datetime
        env.globals.update(timedelta=self._format_timedelta)
        template = env.get_template(self.eoexecutor.REPORT_FILENAME)

        return template

    @staticmethod
    def _format_datetime(value: dt.datetime) -> str:
        """Method for formatting datetime objects into report"""
        return value.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_timedelta(value1: dt.datetime, value2: dt.datetime) -> str:
        """Method for formatting time delta into report"""
        return str(value2 - value1)
