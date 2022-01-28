"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest
from graphviz import Digraph

from eolearn.core import EOTask, EOWorkflow, EONode


class FooTask(EOTask):
    def execute(self, *eopatch):
        return eopatch


@pytest.fixture(name="workflow")
def workflow_fixture():
    node1, node2 = EONode(FooTask()), EONode(FooTask())
    node3 = EONode(FooTask(), [node1, node2])

    workflow = EOWorkflow([node1, node2, node3])
    return workflow


def test_graph_nodes_and_edges(workflow):
    dot = workflow.get_dot()
    assert isinstance(dot, Digraph)

    digraph = workflow.dependency_graph()
    assert isinstance(digraph, Digraph)
