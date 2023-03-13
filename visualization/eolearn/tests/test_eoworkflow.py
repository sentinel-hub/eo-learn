"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import pytest
from graphviz import Digraph

from eolearn.core import EOTask, EOWorkflow, linearly_connect_tasks


class FooTask(EOTask):
    def execute(self, *eopatch):
        return eopatch


class BarTask(EOTask):
    def execute(self, eopatch):
        return eopatch


@pytest.fixture(name="workflow")
def linear_workflow_fixture():
    nodes = linearly_connect_tasks(FooTask(), FooTask(), BarTask())
    return EOWorkflow(nodes)


def test_graph_nodes_and_edges(workflow):
    dot = workflow.get_dot()
    assert isinstance(dot, Digraph)
    dot_repr = str(dot).strip("\n")
    assert dot_repr == "digraph {\n\tFooTask_1 -> FooTask_2\n\tFooTask_2 -> BarTask\n}"

    digraph = workflow.dependency_graph()
    assert isinstance(digraph, Digraph)
    digraph_repr = str(digraph).strip("\n")
    assert digraph_repr == "digraph {\n\tFooTask_1 -> FooTask_2\n\tFooTask_2 -> BarTask\n\trankdir=LR\n}"
