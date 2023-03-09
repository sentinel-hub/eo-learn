"""
Test for module eoworkflow_tasks

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from eolearn.core import EONode, EOTask, EOWorkflow, FeatureType, LoadTask, OutputTask
from eolearn.core.eoworkflow_tasks import InputTask


class DummyTask(EOTask):
    def execute(self, *eopatches):
        return eopatches[0]


def test_input_task():
    """Test basic functionalities of InputTask"""
    task = InputTask(value=31)
    assert task.execute() == 31
    assert task.execute(value=42) == 42

    task = InputTask()
    assert task.execute() is None
    assert task.execute(value=42) == 42


def test_output_task(test_eopatch):
    """Tests basic functionalities of OutputTask"""
    task = OutputTask(name="my-task", features=[FeatureType.BBOX, (FeatureType.DATA, "NDVI")])

    assert task.name == "my-task"

    new_eopatch = task.execute(test_eopatch)
    assert id(new_eopatch) != id(test_eopatch)

    assert len(new_eopatch.get_features()) == 2
    assert new_eopatch.bbox == test_eopatch.bbox


def test_output_task_in_workflow(test_eopatch_path, test_eopatch):
    load = EONode(LoadTask(test_eopatch_path))
    output = EONode(OutputTask(name="result-name"), inputs=[load])

    workflow = EOWorkflow([load, output, EONode(DummyTask(), inputs=[load])])

    results = workflow.execute()

    assert len(results.outputs) == 1
    assert results.outputs["result-name"] == test_eopatch
