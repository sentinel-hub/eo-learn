"""
The module implements the core graph data structure. It is used, for example, to model dependencies among tasks in the
eoworkflow module.
"""

import collections


class NoSuchVertexError(ValueError):
    pass


class DirectedGraph:
    """
    A directed graph using adjacency-list representation.
    """

    def __init__(self, adjacency_dict=None):
        """
        Constructs a new graph from an adjacency list; if None, an empty graph  is constructed.
        :param adjacency_dict: A dictionary mapping vertices to lists neighbors
        """
        self.adj_dict = collections.defaultdict(list, adjacency_dict) if adjacency_dict else collections.defaultdict(
            list)
        self.indegrees = DirectedGraph._get_indegrees(self.adj_dict)
        self._vertices = set(self.adj_dict.keys()) | set([v for neighs in self.adj_dict.values() for v in neighs])

    def __len__(self):
        """
        :return: The number of vertices in the graph.
        """
        return len(self.vertices())

    def __contains__(self, vertex):
        """
        Tells whether ``vertex`` is a vertex of the graph.
        :param vertex: Vertex
        :return: ``True`` if ``vertex`` is a vertex of the graph, ``False`` otherwise
        """
        return vertex in self._vertices

    def __getitem__(self, vertex):
        """
        :param vertex: Vertex of the graph
        :return: Returns the list of ``v`` such that ``u -> v`` in the graph
        """
        return self.adj_dict[vertex]

    def __iter__(self):
        """
        :return: Iterator over the vertices of the graph.
        """
        return iter(self.vertices())

    def get_indegrees(self):
        """
        **Note** that if ``u`` is not in graph then ``indegrees[u]=0`` due to ``defaultdict`` behavior!

        :return: Dict containing in-degrees of vertices of the graph.
        """
        return self.indegrees

    def get_indegree(self, vertex):
        """
        The in-degree of the vertex
        :param vertex: Vertex
        :return: The number of vertices ``vertex'`` such that ``vertex' -> vertex`` is an edge of the graph.
        """
        return self.indegrees[vertex]

    def get_outdegree(self, vertex):
        """
        The out-degree of the vertex
        :param vertex: Vertex
        :return: The number of vertices ``vertex'`` such that ``vertex -> vertex'`` is an edge of the graph
        """
        return len(self.adj_dict[vertex])

    def get_adj_dict(self):
        return self.adj_dict

    def get_outdegrees(self):
        return {vertex: len(self.adj_dict[vertex]) for vertex in self.adj_dict}

    def add_edge(self, u_vertex, v_vertex):
        """
        Adds the edge ``u_vertex -> v_vertex`` to the graph if the edge is not already present.
        :param u_vertex: Vertex
        :param v_vertex: Vertex
        :return: ``True`` if a new edge was added, ``False`` otherwise
        """
        self._vertices.add(u_vertex)
        self._vertices.add(v_vertex)
        if not self.is_edge(u_vertex, v_vertex):
            self.indegrees[v_vertex] += 1
            self.adj_dict[u_vertex].append(v_vertex)
            return True

        return False

    def del_edge(self, u_vertex, v_vertex):
        """
        Removes the edge ``u_vertex -> v_vertex`` from the graph if the edge is present.
        :param u_vertex: Vertex
        :param v_vertex: Vertex
        :return: ``True`` if the existing edge was removed, ``False`` otherwise
        """
        if self.is_edge(u_vertex, v_vertex):
            self.indegrees[v_vertex] -= 1
            self.adj_dict[u_vertex].remove(v_vertex)
            return True

        return False

    def add_vertex(self, vertex):
        """
        Adds a new vertex to the graph (if not already present).

        :param vertex: Vertex
        :return: ``True`` if ``vertex`` added and not yet present; ``False`` otherwise.
        """
        if vertex not in self._vertices:
            self._vertices.add(vertex)
            return True

        return False

    def del_vertex(self, vertex):
        """
        Removes the vertex ``vertex`` and all incident edges from the graph.

        **Note** that this is an expensive operation that should be avoided!

        Running time is O(V+E)

        :param vertex: Vertex
        :return: ``True`` if ``vertex`` was removed from the graph; ``False`` otherwise
        """
        for v_vertex in self.adj_dict[vertex]:
            self.indegrees[v_vertex] -= 1

        for v_vertex in self.vertices():
            if vertex in self.adj_dict[v_vertex]:
                self.adj_dict[v_vertex].remove(vertex)

        if vertex in self._vertices:
            self._vertices.remove(vertex)
            return True

        return False

    def is_edge(self, u_vertex, v_vertex):
        """
        :param u_vertex: Vertex
        :param v_vertex: Vertex
        :return: ``True`` if ``u_vertex -> v_vertex`` is an edge of the graph, ``False`` otherwise
        """
        return v_vertex in self.adj_dict[u_vertex]

    def neighbors(self, vertex):
        """
        The set of all vertex at distance (exactly) 1 from ``vertex``
        :param vertex: Vertex
        :return: The set of ``vertex'`` such that ``vertex -> vertex'`` is an edge of the graph.
        """
        return self.adj_dict[vertex]

    def vertices(self):
        """
        :return: The set of vertices of the graph.
        """
        return self._vertices

    @staticmethod
    def from_edges(edges):
        dag = DirectedGraph()
        for _u, _v in edges:
            dag.add_edge(_u, _v)
        return dag

    @staticmethod
    def _get_indegrees(adj_dict):
        in_degs = collections.defaultdict(int)

        for u_vertex in adj_dict:
            for v_vertex in adj_dict[u_vertex]:
                in_degs[v_vertex] += 1

        return in_degs

    @staticmethod
    def _is_cyclic(dag):
        """
        Tests whether the directed graph dag contains a cycle. The algorithm is naive, running in O(V^2) time, and
        not intended for serious use!

        For production purposes on larger graphs consider implementing Tarjan's O(V+E)-time algorithm instead.

        :param dag: A directed graph
        :type dag: DirectedGraph
        :return: ``True`` if dag contains a cycle, ``False`` otherwise
        :rtype: bool
        """
        # pylint: disable=invalid-name
        vertices = dag.vertices()
        for w in vertices:
            stack = [w]
            seen = set()
            while stack:
                u = stack.pop()
                seen.add(u)
                for v in dag[u]:
                    if v == w:
                        return True
                    if v not in seen:
                        stack.append(v)
        return False
