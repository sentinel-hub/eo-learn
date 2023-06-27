"""
Visualization of EOWorkflow

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from typing import Sequence

from graphviz import Digraph

from eolearn.core import EONode


class EOWorkflowVisualization:
    """Class handling EOWorkflow visualization"""

    def __init__(self, nodes: Sequence[EONode]):
        """
        :param nodes: A sequence of topologically ordered workflow nodes
        """
        self.nodes = nodes

    def dependency_graph(self, filename: str | None = None) -> Digraph:
        """Visualize the computational graph.

        :param filename: Filename of the output image together with file extension. Supported formats: `png`, `jpg`,
            `pdf`, ... . Check `graphviz` Python package for more options
        :return: The DOT representation of the computational graph, with some more formatting
        """
        dot = self.get_dot()
        dot.attr(rankdir="LR")  # Show graph from left to right

        if filename is not None:
            file_name, file_format = filename.rsplit(".", 1)

            dot.render(filename=file_name, format=file_format, cleanup=True)

        return dot

    def get_dot(self) -> Digraph:
        """Generates the DOT description of the underlying computational graph.

        :return: The DOT representation of the computational graph
        """
        dot = Digraph(format="png")

        node_uid_to_dot_name = self._get_node_uid_to_dot_name_mapping(self.nodes)

        for node in self.nodes:
            for input_node in node.inputs:
                dot.edge(node_uid_to_dot_name[input_node.uid], node_uid_to_dot_name[node.uid])
        return dot

    @staticmethod
    def _get_node_uid_to_dot_name_mapping(nodes: Sequence[EONode]) -> dict[str, str]:
        """Creates mapping between EONode classes and names used in DOT graph. To do that, it has to collect nodes with
        the same name and assign them different indices."""
        dot_name_to_nodes: dict[str, list[EONode]] = {}
        for node in nodes:
            dot_name_to_nodes[node.get_name()] = dot_name_to_nodes.get(node.get_name(), [])
            dot_name_to_nodes[node.get_name()].append(node)

        node_to_dot_name = {}
        for _, same_name_nodes in dot_name_to_nodes.items():
            if len(same_name_nodes) == 1:
                node = same_name_nodes[0]
                node_to_dot_name[node.uid] = node.get_name()
            else:
                for idx, node in enumerate(same_name_nodes):
                    node_to_dot_name[node.uid] = node.get_name(idx + 1)

        return node_to_dot_name
