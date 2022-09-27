"""
Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Matej Batič, Miha Kadunc, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import time

from eolearn.core import EONode, EOTask, OutputTask, linearly_connect_tasks


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


def test_hashing():
    _ = {EONode(Inc()): "Can be hashed!"}

    linear = EONode(Inc())
    for _ in range(5000):
        linear = EONode(Inc(), inputs=[linear])

    branch_1, branch_2 = EONode(Inc()), EONode(Inc())
    for _ in range(500):
        branch_1 = EONode(DivideTask(), inputs=(branch_1, branch_2))
        branch_2 = EONode(DivideTask(), inputs=(branch_2, EONode(Inc())))

    t_start = time.time()
    linear.__hash__()
    branch_1.__hash__()
    branch_2.__hash__()
    t_end = time.time()
    assert t_end - t_start < 5, "Assert hashing slows down for large workflows!"


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
