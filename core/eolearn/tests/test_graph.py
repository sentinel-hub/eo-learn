"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pytest

from eolearn.core.graph import DirectedGraph


@pytest.fixture(name='test_graph')
def test_graph_fixture():
    return DirectedGraph({1: [2, 3], 2: [4], 3: [4]})


def test_is_edge():
    d = DirectedGraph({1: [2]})
    assert d.is_edge(1, 2)
    assert not d.is_edge(2, 1)


def test_add_edge():
    d = DirectedGraph()
    assert not d.is_edge(1, 2)
    assert d.get_outdegree(1) == 0

    d.add_edge(1, 2)

    assert d.get_outdegree(1) == 1
    assert d.get_indegree(2) == 1
    assert d.is_edge(1, 2)

    d.add_edge(1, 2)

    assert d.get_indegree(1) == 0
    assert d.get_indegree(2) == 1


def test_del_edge():
    d = DirectedGraph({1: [2]})
    assert d.is_edge(1, 2)
    d.del_edge(1, 2)
    assert not d.is_edge(1, 2)


def test_neigbors(test_graph):
    assert test_graph.neighbors(1) == [2, 3]
    assert test_graph.neighbors(2) == [4]
    assert test_graph.neighbors(3) == [4]
    assert test_graph.neighbors(4) == []


def test_get_indegree(test_graph):
    assert test_graph.get_indegree(1) == 0
    assert test_graph.get_indegree(2) == 1
    assert test_graph.get_indegree(3) == 1
    assert test_graph.get_indegree(4) == 2


def test_get_outdegree(test_graph):
    assert test_graph.get_outdegree(1) == 2
    assert test_graph.get_outdegree(2) == 1
    assert test_graph.get_outdegree(3) == 1
    assert test_graph.get_outdegree(4) == 0


def test_vertices(test_graph):
    assert set(test_graph.vertices()) == set([1, 2, 3, 4])

    d2 = DirectedGraph()
    d2.add_edge(1, 2)
    d2.add_edge(2, 3)
    d2.add_edge(3, 4)
    assert set(d2.vertices()) == set([1, 2, 3, 4])


def test_add_vertex():
    d = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})
    d.add_vertex(5)
    assert 5 in d
    assert d.get_indegree(5) == 0
    assert d.get_outdegree(5) == 0
    assert d[5] == []


def test_del_vertex():
    d = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})

    assert d.get_outdegree(1) == 2
    assert d.get_indegree(4) == 2
    assert len(d) == 4
    assert 2 in d

    d.del_vertex(2)

    assert d.get_outdegree(1) == 1
    assert d.get_indegree(4) == 1
    assert len(d) == 3
    assert 2 not in d
