"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import concurrent.futures
import datetime as dt

import pytest
import numpy as np

from eolearn.core import (
    EOPatch,
    EOTask,
    EONode,
    EOWorkflow,
    OutputTask,
    WorkflowResults,
    FeatureType,
    InitializeFeatureTask,
    RemoveFeatureTask,
    CreateEOPatchTask,
)
from eolearn.core.eoworkflow import NodeStats


class CustomException(ValueError):
    pass


class InputTask(EOTask):
    def execute(self, *, val=None):
        return val


class DivideTask(EOTask):
    def execute(self, x, y, *, z=0):
        return x / y + z


class IncTask(EOTask):
    def execute(self, x, *, d=1):
        return x + d


class ExceptionTask(EOTask):
    def execute(self, *_, **__):
        raise CustomException


def test_workflow_arguments():
    input_node1 = EONode(InputTask())
    input_node2 = EONode(InputTask(), name="some name")
    divide_node = EONode(DivideTask(), inputs=(input_node1, input_node2), name="some name")
    output_node = EONode(OutputTask(name="output"), inputs=[divide_node])

    workflow = EOWorkflow([input_node1, input_node2, divide_node, output_node])

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        k2future = {
            k: executor.submit(workflow.execute, {input_node1: {"val": k**3}, input_node2: {"val": k**2}})
            for k in range(2, 100)
        }
        executor.shutdown()
        for k in range(2, 100):
            assert k2future[k].result().outputs["output"] == k

    result1 = workflow.execute({input_node1: {"val": 15}, input_node2: {"val": 3}})
    assert result1.outputs["output"] == 5

    result2 = workflow.execute({input_node1: {"val": 6}, input_node2: {"val": 3}})
    assert result2.outputs["output"] == 2

    result3 = workflow.execute({input_node1: {"val": 6}, input_node2: {"val": 3}, divide_node: {"z": 1}})
    assert result3.outputs[output_node.task.name] == 3


def test_get_nodes():
    in_node = EONode(InputTask())
    inc_node0 = EONode(IncTask(), inputs=[in_node])
    inc_node1 = EONode(IncTask(), inputs=[inc_node0])
    inc_node2 = EONode(IncTask(), inputs=[inc_node1])
    output_node = EONode(OutputTask(name="out"), inputs=[inc_node2])

    eow = EOWorkflow([in_node, inc_node0, inc_node1, inc_node2, output_node])

    returned_nodes = eow.get_nodes()

    assert [
        in_node,
        inc_node0,
        inc_node1,
        inc_node2,
        output_node,
    ] == returned_nodes, "Returned nodes differ from original nodes"

    arguments_dict = {in_node: {"val": 2}, inc_node0: {"d": 2}}
    workflow_res = eow.execute(arguments_dict)

    manual_res = []
    for _, node in enumerate(returned_nodes):
        manual_res = [node.task.execute(*manual_res, **arguments_dict.get(node, {}))]

    assert workflow_res.outputs["out"] == manual_res[0], "Manually running returned nodes produces different results."


def test_get_node_with_uid():
    in_node = EONode(InputTask())
    inc_node = EONode(IncTask(), inputs=[in_node])
    output_node = EONode(OutputTask(name="out"), inputs=[inc_node])

    eow = EOWorkflow([in_node, inc_node, output_node])

    assert all(node == eow.get_node_with_uid(node.uid) for node in (in_node, inc_node, output_node))
    assert eow.get_node_with_uid("nonexsitant") is None
    with pytest.raises(KeyError):
        eow.get_node_with_uid("nonexsitant", fail_if_missing=True)


@pytest.mark.parametrize(
    "faulty_parameters",
    [
        [InputTask(), IncTask(), IncTask()],
        EONode(InputTask()),
        [EONode(IncTask()), IncTask()],
        [EONode(IncTask()), (EONode(IncTask()), "name")],
        [EONode(IncTask()), (EONode(IncTask(), inputs=[EONode(IncTask())]))],
        [EONode(IncTask()), (EONode(IncTask()), IncTask())],
    ],
)
def test_input_exceptions(faulty_parameters):
    with pytest.raises(ValueError):
        EOWorkflow(faulty_parameters)


def test_bad_structure_exceptions():
    in_node = EONode(InputTask())
    inc_node0 = EONode(IncTask(), inputs=[in_node])
    inc_node1 = EONode(IncTask(), inputs=[inc_node0])
    inc_node2 = EONode(IncTask(), inputs=[inc_node1])
    output_node = EONode(OutputTask(name="out"), inputs=[inc_node2])

    # This one must work
    EOWorkflow([in_node, inc_node0, inc_node1, inc_node2, output_node])

    # Duplicated node
    with pytest.raises(ValueError):
        EOWorkflow([in_node, inc_node0, inc_node0, inc_node1, inc_node2, output_node])

    # Missing node
    with pytest.raises(ValueError):
        EOWorkflow([in_node, inc_node0, inc_node2, output_node])

    # Create circle (much more difficult now)
    super(EONode, inc_node0).__setattr__("inputs", (inc_node1,))
    with pytest.raises(ValueError):
        EOWorkflow([in_node, inc_node0, inc_node1, inc_node2, output_node])


def test_multiedge_workflow():
    in_node = EONode(InputTask())
    inc_node = EONode(IncTask(), inputs=[in_node])
    div_node = EONode(DivideTask(), inputs=[inc_node, inc_node])
    output_node = EONode(OutputTask(name="out"), inputs=[div_node])

    workflow = EOWorkflow([in_node, output_node, inc_node, div_node])
    arguments_dict = {in_node: {"val": 2}}
    workflow_res = workflow.execute(arguments_dict)

    assert workflow_res.outputs["out"] == 1


def test_workflow_copying_eopatches():
    feature1 = FeatureType.DATA, "data1"
    feature2 = FeatureType.DATA, "data2"

    create_node = EONode(CreateEOPatchTask())
    init_node = EONode(
        InitializeFeatureTask([feature1, feature2], shape=(2, 4, 4, 3), init_value=1),
        inputs=[create_node],
    )
    remove_node1 = EONode(RemoveFeatureTask([feature1]), inputs=[init_node])
    remove_node2 = EONode(RemoveFeatureTask([feature2]), inputs=[init_node])
    output_node1 = EONode(OutputTask(name="out1"), inputs=[remove_node1])
    output_node2 = EONode(OutputTask(name="out2"), inputs=[remove_node2])

    workflow = EOWorkflow([create_node, init_node, remove_node1, remove_node2, output_node1, output_node2])
    results = workflow.execute()

    eop1 = results.outputs["out1"]
    eop2 = results.outputs["out2"]

    assert eop1 == EOPatch(data={"data2": np.ones((2, 4, 4, 3), dtype=np.uint8)})
    assert eop2 == EOPatch(data={"data1": np.ones((2, 4, 4, 3), dtype=np.uint8)})


def test_workflows_reusing_nodes():

    in_node = EONode(InputTask())
    node1 = EONode(IncTask(), inputs=[in_node])
    node2 = EONode(IncTask(), inputs=[node1])
    out_node = EONode(OutputTask(name="out"), inputs=[node2])
    input_args = {in_node: {"val": 2}, node2: {"d": 2}}

    original = EOWorkflow([in_node, node1, node2, out_node])
    node_reuse = EOWorkflow([in_node, node1, node2, out_node])

    assert original.execute(input_args).outputs["out"] == node_reuse.execute(input_args).outputs["out"]


def test_workflow_results():
    input_node = EONode(InputTask())
    output_node = EONode(OutputTask(name="out"), inputs=[input_node])
    workflow = EOWorkflow([input_node, output_node])

    results = workflow.execute({input_node: {"val": 10}})

    assert isinstance(results, WorkflowResults)
    assert results.outputs == {"out": 10}

    results_without_outputs = results.drop_outputs()
    assert results_without_outputs.outputs == {}
    assert id(results_without_outputs) != id(results)

    assert isinstance(results.start_time, dt.datetime)
    assert isinstance(results.end_time, dt.datetime)
    assert results.start_time < results.end_time < dt.datetime.now()

    assert isinstance(results.stats, dict)
    assert len(results.stats) == 2
    for node in [input_node, output_node]:
        stats_uid = node.uid
        assert isinstance(results.stats.get(stats_uid), NodeStats)


def test_workflow_from_endnodes():
    input_node1 = EONode(InputTask())
    input_node2 = EONode(InputTask(), name="some name")
    divide_node = EONode(DivideTask(), inputs=(input_node1, input_node2), name="some name")
    output_node = EONode(OutputTask(name="out"), inputs=[divide_node])

    regular_workflow = EOWorkflow([input_node1, input_node2, divide_node, output_node])
    endnode_workflow = EOWorkflow.from_endnodes(output_node)

    assert isinstance(endnode_workflow, EOWorkflow)
    assert set(endnode_workflow.get_nodes()) == set(regular_workflow.get_nodes()), "Nodes are different"

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        regular_results = [
            executor.submit(regular_workflow.execute, {input_node1: {"val": k**3}, input_node2: {"val": k**2}})
            for k in range(2, 100)
        ]
        endnode_results = [
            executor.submit(endnode_workflow.execute, {input_node1: {"val": k**3}, input_node2: {"val": k**2}})
            for k in range(2, 100)
        ]
        executor.shutdown()
        assert all(
            x.result().outputs["out"] == y.result().outputs["out"] for x, y in zip(regular_results, endnode_results)
        )

    endnode_duplicates = EOWorkflow.from_endnodes(output_node, output_node, divide_node)
    assert set(endnode_duplicates.get_nodes()) == set(regular_workflow.get_nodes()), "Fails if endnodes are repeated"


def test_exception_handling():
    input_node = EONode(InputTask(), name="xyz")
    exception_node = EONode(ExceptionTask(), inputs=[input_node])
    increase_node = EONode(IncTask(), inputs=[exception_node])
    workflow = EOWorkflow([input_node, exception_node, increase_node])

    with pytest.raises(CustomException):
        workflow.execute()

    results = workflow.execute(raise_errors=False)

    assert results.outputs == {}
    assert results.error_node_uid == exception_node.uid
    assert len(results.stats) == 2

    for node in [input_node, exception_node]:
        node_stats = results.stats[node.uid]

        assert node_stats.node_uid == node.uid
        assert node_stats.node_name == node.name

        if node is exception_node:
            assert isinstance(node_stats.exception, CustomException)
            assert node_stats.exception_traceback.startswith("Traceback")
        else:
            assert node_stats.exception is None
            assert node_stats.exception_traceback is None
