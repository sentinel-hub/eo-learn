"""
The module implements the core graph data structure. It is used, for example, to model dependencies among tasks in the
eoworkflow module.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import collections
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple


class CyclicDependencyError(ValueError):
    """This error is raised when trying to get a topological ordering of a `DirectedGraph`."""


class DirectedGraph:
    """A directed graph using adjacency-list representation. The graph is multi-edge.

    Constructs a new graph from an adjacency list. If adjacency_dict is None, an empty graph is constructed.

    :param adjacency_dict: A dictionary mapping vertices to lists of neighbors
    """

    def __init__(self, adjacency_dict: Optional[Dict[object, List[object]]] = None):
        self._adj_dict = (
            collections.defaultdict(list, adjacency_dict) if adjacency_dict else collections.defaultdict(list)
        )
        self._indegrees = self._make_indegrees_dict()
        self._vertices = set(self._adj_dict.keys()) | {v for neighs in self._adj_dict.values() for v in neighs}

    def __len__(self) -> int:
        """Returns the number of vertices in the graph."""
        return len(self._vertices)

    def __contains__(self, vertex) -> bool:
        """True if `vertex` is a vertex of the graph. False otherwise.

        :param vertex: Vertex
        """
        return vertex in self._vertices

    def __iter__(self):
        """Returns iterator over the vertices of the graph."""
        return iter(self._vertices)

    def _make_indegrees_dict(self):
        indegrees = collections.defaultdict(int)

        for u_vertex in self._adj_dict:
            for v_vertex in self._adj_dict[u_vertex]:
                indegrees[v_vertex] += 1

        return indegrees

    def get_indegrees(self) -> Dict[Any, int]:
        """Returns a dictionary containing in-degrees of vertices of the graph."""
        return dict(self._indegrees)

    def get_indegree(self, vertex) -> int:
        """Returns the in-degree of the vertex.

        The in-degree is the number of vertices `vertex'` such that `vertex' -> vertex` is an edge of the graph.

        :param vertex: Vertex
        """
        return self._indegrees[vertex]

    def get_outdegrees(self):
        """
        :return: dictionary of out-degrees, see get_outdegree
        """
        return {vertex: len(self._adj_dict[vertex]) for vertex in self._adj_dict}

    def get_outdegree(self, vertex) -> int:
        """Returns the out-degree of the vertex.

        The out-degree is the number of vertices `vertex'` such that `vertex -> vertex'` is an edge of the graph.

        :param vertex: Vertex
        """
        return len(self._adj_dict[vertex])

    def get_adj_dict(self) -> Dict[Any, list]:
        """
        :return: adj_dict
        """
        return {vertex: copy.copy(neighbours) for vertex, neighbours in self._adj_dict.items()}

    def get_vertices(self) -> set:
        """Returns the set of vertices of the graph."""
        return set(self._vertices)

    def add_edge(self, u_vertex, v_vertex):
        """Adds the edge `u_vertex -> v_vertex` to the graph if the edge is not already present.

        :param u_vertex: Vertex
        :param v_vertex: Vertex
        :return: `True` if a new edge was added. `False` otherwise.
        """
        self._vertices.add(u_vertex)
        self._vertices.add(v_vertex)
        self._indegrees[v_vertex] += 1
        self._adj_dict[u_vertex].append(v_vertex)

    def del_edge(self, u_vertex, v_vertex) -> bool:
        """Removes the edge `u_vertex -> v_vertex` from the graph if the edge is present.

        :param u_vertex: Vertex
        :param v_vertex: Vertex
        :return: `True` if the existing edge was removed. `False` otherwise.
        """
        if self.is_edge(u_vertex, v_vertex):
            self._indegrees[v_vertex] -= 1
            self._adj_dict[u_vertex].remove(v_vertex)
            return True

        return False

    def add_vertex(self, vertex) -> bool:
        """Adds a new vertex to the graph if not present.

        :param vertex: Vertex
        :return: `True` if `vertex` added and not yet present. `False` otherwise.
        """
        if vertex not in self._vertices:
            self._vertices.add(vertex)
            return True

        return False

    def del_vertex(self, vertex) -> bool:
        """Removes the vertex `vertex` and all incident edges from the graph.

        **Note** that this is an expensive operation that should be avoided!

        Running time is O(V+E)

        :param vertex: Vertex to be removed from graph
        :return: `True` if `vertex` was removed from the graph. `False` otherwise.
        """
        if vertex not in self._vertices:
            return False

        for v_vertex in self._adj_dict[vertex]:
            self._indegrees[v_vertex] -= 1

        for v_vertex in self._vertices:
            if vertex in self._adj_dict[v_vertex]:
                self._adj_dict[v_vertex].remove(vertex)

        self._vertices.remove(vertex)
        return True

    def is_edge(self, u_vertex, v_vertex) -> bool:
        """True if `u_vertex -> v_vertex` is an edge of the graph. False otherwise."""
        return v_vertex in self._adj_dict[u_vertex]

    def get_neighbors(self, vertex) -> list:
        """Returns the set of successor vertices of `vertex`."""
        return copy.copy(self._adj_dict[vertex])

    @staticmethod
    def from_edges(edges: Sequence[Tuple[object, object]]) -> "DirectedGraph":
        """Return DirectedGraph created from edges.
        :param edges: Pairs of objects that describe all the edges of the graph
        :return: DirectedGraph
        """
        dag = DirectedGraph()
        for _u, _v in edges:
            dag.add_edge(_u, _v)
        return dag

    @staticmethod
    def _is_cyclic(graph: "DirectedGraph") -> bool:
        """True if the directed graph contains a cycle. False otherwise.

        The algorithm is naive, running in O(V^2) time, and not intended for serious use! For production purposes on
        larger graphs consider implementing Tarjan's O(V+E)-time algorithm instead.
        """
        # pylint: disable=invalid-name
        vertices = graph.get_vertices()
        for w in vertices:
            stack = [w]
            seen = set()
            while stack:
                u = stack.pop()
                seen.add(u)
                for v in graph.get_neighbors(u):
                    if v == w:
                        return True
                    if v not in seen:
                        stack.append(v)
        return False

    def topologically_ordered_vertices(self) -> list:
        """Computes an ordering `<` of vertices so that for any two vertices `v` and `v'` we have that if `v˙ depends
        on `v'` then `v' < v`. In words, all dependencies of a vertex precede the vertex in this ordering.

        :return: A list of topologically ordered dependencies
        """
        in_degrees = self.get_indegrees()

        independent_vertices = collections.deque([v for v in self._vertices if self.get_indegree(v) == 0])
        topological_order = []
        while independent_vertices:
            v_vertex = independent_vertices.popleft()
            topological_order.append(v_vertex)

            for u_vertex in self._adj_dict[v_vertex]:
                in_degrees[u_vertex] -= 1
                if in_degrees[u_vertex] == 0:
                    independent_vertices.append(u_vertex)

        if len(topological_order) != len(self):
            raise CyclicDependencyError("Nodes form a cyclic graph, cannot produce a topologically ordered list.")

        return topological_order
