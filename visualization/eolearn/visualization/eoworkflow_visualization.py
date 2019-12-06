"""
Visualization of EOWorkflow

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from graphviz import Digraph


class EOWorkflowVisualization:
    """ Class handling EOWorkflow visualization
    """
    def __init__(self, workflow):
        """
        :param workflow: An instance of a workflow
        :type workflow: EOWorkflow
        """
        self.workflow = workflow

    def dependency_graph(self, filename=None):
        """ Visualize the computational graph.

        :param filename: Filename of the output image together with file extension. Supported formats: `png`, `jpg`,
            `pdf`, ... . Check `graphviz` Python package for more options
        :type filename: str
        :return: The DOT representation of the computational graph, with some more formatting
        :rtype: Digraph
        """
        dot = self.get_dot()
        dot.attr(rankdir='LR')  # Show graph from left to right

        if filename is not None:
            file_name, file_format = filename.rsplit('.', 1)

            dot.render(filename=file_name, format=file_format, cleanup=True)

        return dot

    def get_dot(self):
        """Generates the DOT description of the underlying computational graph.

        :return: The DOT representation of the computational graph
        :rtype: Digraph
        """
        dot = Digraph(format='png')

        dep_to_dot_name = self._get_dep_to_dot_name_mapping(self.workflow.ordered_dependencies)

        for dep in self.workflow.ordered_dependencies:
            for input_task in dep.inputs:
                dot.edge(dep_to_dot_name[self.workflow.uuid_dict[input_task.private_task_config.uuid]],
                         dep_to_dot_name[dep])
        return dot

    @staticmethod
    def _get_dep_to_dot_name_mapping(dependencies):
        """ Creates mapping between Dependency classes and names used in DOT graph
        """
        dot_name_to_deps = {}
        for dep in dependencies:
            dot_name = dep.name

            if dot_name not in dot_name_to_deps:
                dot_name_to_deps[dot_name] = [dep]
            else:
                dot_name_to_deps[dot_name].append(dep)

        dep_to_dot_name = {}
        for dot_name, deps in dot_name_to_deps.items():
            if len(deps) == 1:
                dep_to_dot_name[deps[0]] = dot_name
                continue

            for idx, dep in enumerate(deps):
                dep_to_dot_name[dep] = dot_name + str(idx)

        return dep_to_dot_name
