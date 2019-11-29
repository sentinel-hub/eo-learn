"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import unittest

from graphviz import Digraph

from eolearn.core import EOTask, EOWorkflow, Dependency, WorkflowResults, LinearWorkflow


class FooTask(EOTask):
    def execute(self, *eopatch):
        return eopatch


class TestGraph(unittest.TestCase):

    def setUp(self):
        task1 = FooTask()
        task2 = FooTask()
        task3 = FooTask()

        self.workflow = EOWorkflow(dependencies=[
            Dependency(task=task1, inputs=[]),
            Dependency(task=task2, inputs=[]),
            Dependency(task=task3, inputs=[task1, task2])
        ])

    def test_graph_nodes_and_edges(self):
        dot = self.workflow.get_dot()
        self.assertTrue(isinstance(dot, Digraph))

        digraph = self.workflow.dependency_graph()
        self.assertTrue(isinstance(digraph, Digraph))


if __name__ == '__main__':
    unittest.main()
