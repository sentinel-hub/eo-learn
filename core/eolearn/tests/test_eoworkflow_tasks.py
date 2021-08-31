"""
Test for module eoworkflow_tasks.py
"""
from eolearn.core import EOTask, EOWorkflow, FeatureType, LoadTask, OutputTask


class DummyTask(EOTask):

    def execute(self, *eopatches):
        return eopatches[0]


def test_output_task(test_eopatch):
    """ Tests basic functionalities of OutputTask
    """
    task = OutputTask(name='my-task', features=[FeatureType.BBOX, (FeatureType.DATA, 'NDVI')])

    assert task.name == 'my-task'

    new_eopatch = task.execute(test_eopatch)
    assert id(new_eopatch) != id(test_eopatch)

    assert len(new_eopatch.get_feature_list()) == 2
    assert new_eopatch.bbox == test_eopatch.bbox


def test_output_task_in_workflow(test_eopatch_path, test_eopatch):
    load = LoadTask(test_eopatch_path)
    output = OutputTask(name='result-name')

    workflow = EOWorkflow([
        (load, []),
        (output, [load]),
        (DummyTask(), [load])
    ])

    results = workflow.execute()

    assert len(results.outputs) == 1
    assert results.outputs['result-name'] == test_eopatch
