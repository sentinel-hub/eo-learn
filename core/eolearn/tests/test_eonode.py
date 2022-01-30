"""
Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Matej Batič, Miha Kadunc, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from eolearn.core import EOTask, EONode, linearly_connect_tasks, OutputTask


class InputTask(EOTask):
    def execute(self, *, val=None):
        return val


class DivideTask(EOTask):
    def execute(self, x, y, *, z=0):
        return x / y + z


class Inc(EOTask):
    def execute(self, x, *, d=1):
        return x + d


def test_nodes_different_uids():
    uids = set()
    for _ in range(5000):
        node = EONode(Inc())
        uids.add(node.uid)

    assert len(uids) == 5000, "Different nodes should have different uids."


def test_get_dependencies():
    input_node1 = EONode(InputTask())
    input_node2 = EONode(InputTask(), name="some name")
    divide_node1 = EONode(DivideTask(), inputs=(input_node1, input_node2), name="some name")
    divide_node2 = EONode(DivideTask(), inputs=(divide_node1, input_node2), name="some name")
    output_node = EONode(OutputTask(name="output"), inputs=[divide_node2])
    all_nodes = {input_node1, input_node2, divide_node1, divide_node2, output_node}

    assert len(output_node.get_dependencies()) == len(all_nodes), "Wrong number of nodes returned"

    assert all_nodes == set(output_node.get_dependencies())


def test_linearly_connect_tasks():
    tasks = [InputTask(), Inc(), (Inc(), "special inc"), Inc(), OutputTask(name="out")]
    nodes = linearly_connect_tasks(*tasks)

    assert all(isinstance(node, EONode) for node in nodes), "Function does not return EONodes"

    assert len(tasks) == len(nodes), "Function returns incorrect number of nodes"

    pure_tasks = tasks[:]
    pure_tasks[2] = pure_tasks[2][0]
    assert all(task == node.task for task, node in zip(pure_tasks, nodes)), "Nodes do not contain correct tasks"

    for i, node in enumerate(nodes):
        previous_endpoint = () if i == 0 else (nodes[i - 1],)
        assert node.inputs == previous_endpoint, "Nodes are not linked linearly"

    assert nodes[2].name == "special inc", "Names are not handled correctly"
