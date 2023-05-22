"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import functools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eolearn.core.graph import CyclicDependencyError, DirectedGraph


@pytest.fixture(name="test_graph")
def test_graph_fixture():
    return DirectedGraph({1: [2, 3], 2: [4], 3: [4]})


def test_is_edge():
    graph = DirectedGraph({1: [2]})
    assert graph.is_edge(1, 2)
    assert not graph.is_edge(2, 1)


def test_add_edge():
    graph = DirectedGraph()
    assert not graph.is_edge(1, 2)
    assert graph.get_outdegree(1) == 0

    graph.add_edge(1, 2)

    assert graph.get_outdegree(1) == 1
    assert graph.get_indegree(2) == 1
    assert graph.is_edge(1, 2)

    graph.add_edge(1, 2)

    assert graph.get_indegree(1) == 0
    assert graph.get_outdegree(1) == 2
    assert graph.get_indegree(2) == 2


def test_del_edge():
    graph = DirectedGraph({1: [2]})
    assert graph.is_edge(1, 2)
    graph.del_edge(1, 2)
    assert not graph.is_edge(1, 2)

    graph = DirectedGraph({1: [2, 2]})
    assert graph.is_edge(1, 2)
    graph.del_edge(1, 2)
    assert graph.is_edge(1, 2)
    graph.del_edge(1, 2)
    assert not graph.is_edge(1, 2)


def test_neigbors(test_graph):
    assert test_graph.get_neighbors(1) == [2, 3]
    assert test_graph.get_neighbors(2) == [4]
    assert test_graph.get_neighbors(3) == [4]
    assert test_graph.get_neighbors(4) == []


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
    assert test_graph.get_vertices() == {1, 2, 3, 4}

    graph = DirectedGraph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    assert graph.get_vertices() == {1, 2, 3, 4}


def test_add_vertex():
    graph = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})
    graph.add_vertex(5)
    assert 5 in graph
    assert graph.get_indegree(5) == 0
    assert graph.get_outdegree(5) == 0
    assert graph.get_neighbors(5) == []


def test_del_vertex():
    graph = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})

    assert graph.get_outdegree(1) == 2
    assert graph.get_indegree(4) == 2
    assert len(graph) == 4
    assert 2 in graph

    graph.del_vertex(2)

    assert graph.get_outdegree(1) == 1
    assert graph.get_indegree(4) == 1
    assert len(graph) == 3
    assert 2 not in graph


@given(
    st.lists(
        st.tuples(st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10)).filter(
            lambda p: p[0] != p[1]
        ),
        min_size=1,
        max_size=110,
    )
)
def test_resolve_dependencies(edges):
    graph = DirectedGraph.from_edges(edges)

    if DirectedGraph._is_cyclic(graph):  # noqa[SLF001]
        with pytest.raises(CyclicDependencyError):
            graph.topologically_ordered_vertices()
    else:
        vertex_position = {vertex: i for i, vertex in enumerate(graph.topologically_ordered_vertices())}
        assert functools.reduce(lambda p, q: p and q, [vertex_position[u] < vertex_position[v] for u, v in edges])
