"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest
from graphviz import Digraph

from eolearn.core import EONode, EOTask, EOWorkflow


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
