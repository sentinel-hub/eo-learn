"""
Module with utilities for visualizing EOExecutor

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import fs
import graphviz
import matplotlib as mpl
import numpy as np
import pygments
import pygments.formatter
import pygments.lexers
from jinja2 import Environment, FileSystemLoader, Template
from pygments.formatters.html import HtmlFormatter

from eolearn.core import EOExecutor
from eolearn.core.eonode import ExceptionInfo
from eolearn.core.exceptions import EORuntimeWarning, EOUserWarning


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

        backend_ctx = mpl.rc_context({"backend": "Agg"}) if os.environ.get("DISPLAY", "") == "" else nullcontext()
        with backend_ctx:
            try:
                dependency_graph = self._create_dependency_graph()
            except graphviz.backend.ExecutableNotFound as ex:
                dependency_graph = None
                warnings.warn(
                    f"{ex}.\nPlease install the system package 'graphviz' (in addition to the python package) to"
                    " have the dependency graph in the final report!",
                    EOUserWarning,
                )

            formatter = HtmlFormatter(linenos=True)

            template = self._get_template()

            log_paths = self.eoexecutor.get_log_paths(full_path=False)
            if not include_logs:
                execution_logs = None
            elif self.eoexecutor.save_logs:
                with ThreadPoolExecutor() as executor:
                    execution_logs = list(executor.map(self._read_log_file, log_paths))
            else:
                execution_logs = ["No logs saved"] * len(self.eoexecutor.execution_kwargs)
            html = template.render(
                title=f"Report {self._format_datetime(start_time)}",
                dependency_graph=dependency_graph,
                general_stats=self.eoexecutor.general_stats,
                exception_stats=self._get_exception_stats(),
                task_descriptions=self._get_node_descriptions(),
                execution_results=self.eoexecutor.execution_results,
                execution_tracebacks=self._render_execution_tracebacks(formatter),
                execution_logs=execution_logs,
                execution_log_filenames=[fs.path.basename(log_path) for log_path in log_paths],
                execution_names=self.eoexecutor.execution_names,
                code_css=formatter.get_style_defs(),
            )
            self.eoexecutor.filesystem.makedirs(report_folder, recreate=True)

            with self.eoexecutor.filesystem.open(self.eoexecutor.get_report_path(full_path=False), "w") as file_handle:
                file_handle.write(html)

    def _read_log_file(self, log_path: str) -> str:
        try:
            with self.eoexecutor.filesystem.open(log_path, "r") as file_handle:
                return file_handle.read()
        except BaseException as exception:
            warnings.warn(f"Failed to load logs with exception: {exception!r}", category=EORuntimeWarning)
            return "Failed to load logs"

    def _create_dependency_graph(self) -> str:
        """Provides an image of dependency graph"""
        dot = self.eoexecutor.workflow.dependency_graph()
        return base64.b64encode(dot.pipe()).decode()

    def _get_exception_stats(self) -> list[tuple[str, str, list[_ErrorSummary]]]:
        """Creates aggregated stats about exceptions

        Returns tuples of form (name, uid, [error_summary])
        """

        exception_stats: defaultdict[str, dict[str, _ErrorSummary]] = defaultdict(dict)

        for execution_idx, (execution, results) in enumerate(
            zip(self.eoexecutor.execution_names, self.eoexecutor.execution_results)
        ):
            if not results.error_node_uid:
                continue

            error_node = results.stats[results.error_node_uid]
            exception_info: ExceptionInfo = error_node.exception_info  # type: ignore[assignment]
            origin_str = f"<b>{exception_info.exception.__class__.__name__}</b> raised from {exception_info.origin}"

            if origin_str not in exception_stats[error_node.node_uid]:
                exception_stats[error_node.node_uid][origin_str] = _ErrorSummary(
                    origin_str, str(exception_info.exception), []
                )

            exception_stats[error_node.node_uid][origin_str].add_execution(execution_idx, execution)

        ordered_exception_stats = []
        for node in self.eoexecutor.workflow.get_nodes():
            if node.uid not in exception_stats:
                continue

            node_stats = exception_stats[node.uid]
            error_summaries = sorted(node_stats.values(), key=lambda summary: -len(summary.failed_indexed_executions))
            ordered_exception_stats.append((node.get_name(), node.uid, error_summaries))

        return ordered_exception_stats

    def _get_node_descriptions(self) -> list[dict[str, Any]]:
        """Prepares a list of node names and initialization parameters of their tasks"""
        descriptions = []
        name_counts: dict[str, int] = defaultdict(lambda: 0)

        for node in self.eoexecutor.workflow.get_nodes():
            node_name = node.get_name(name_counts[node.get_name()])
            name_counts[node.get_name()] += 1

            node_stats = filter(None, (results.stats.get(node.uid) for results in self.eoexecutor.execution_results))
            durations = np.array([(stats.end_time - stats.start_time).total_seconds() for stats in node_stats])
            if len(durations) == 0:
                duration_report = "unknown"
            else:
                duration_report = (
                    f"Between {np.min(durations):.4g} and {np.max(durations):.4g} seconds,"
                    f" usually {np.mean(durations):.4g} ± {np.std(durations):.4g} seconds"
                )

            descriptions.append({
                "name": f"{node_name} ({node.uid})",
                "uid": node.uid,
                "args": {
                    key: value.replace("<", "&lt;").replace(">", "&gt;")  # type: ignore[attr-defined]
                    for key, value in node.task.private_task_config.init_args.items()
                },
                "duration_report": duration_report,
            })
        return descriptions

    def _render_execution_tracebacks(self, formatter: pygments.formatter.Formatter) -> list:
        """Renders stack traces of those executions which failed"""
        tb_lexer = pygments.lexers.get_lexer_by_name("py3tb", stripall=True)

        tracebacks = []
        for results in self.eoexecutor.execution_results:
            if results.workflow_failed() and results.error_node_uid is not None:
                # second part of above check needed only for typechecking purposes
                failed_node_stats = results.stats[results.error_node_uid]
                traceback_str = failed_node_stats.exception_info.traceback  # type: ignore[union-attr]
                traceback = pygments.highlight(traceback_str, tb_lexer, formatter)
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

        return env.get_template(self.eoexecutor.REPORT_FILENAME)

    @staticmethod
    def _format_datetime(value: dt.datetime) -> str:
        """Method for formatting datetime objects into report"""
        return value.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_timedelta(value1: dt.datetime, value2: dt.datetime) -> str:
        """Method for formatting time delta into report"""
        return str(value2 - value1)


@dataclass()
class _ErrorSummary:
    """Contains data for errors of a node."""

    origin: str
    example_message: str
    failed_indexed_executions: list[tuple[int, str]]

    def add_execution(self, index: int, name: str) -> None:
        """Adds an execution to the summary."""
        self.failed_indexed_executions.append((index, name))

    @property
    def num_failed(self) -> int:
        """Helps with jinja"""
        return len(self.failed_indexed_executions)
