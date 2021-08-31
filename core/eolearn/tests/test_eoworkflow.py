"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import functools
import concurrent.futures
import datetime as dt

import ipdb
import pytest
import numpy as np
from hypothesis import given, strategies as st

from eolearn.core import (
    EOPatch, EOTask, EOWorkflow, Dependency, LinearWorkflow, OutputTask, WorkflowResults, FeatureType,
    InitializeFeatureTask, RemoveFeatureTask
)
from eolearn.core.eoworkflow import CyclicDependencyError, TaskStats
from eolearn.core.graph import DirectedGraph


class InputTask(EOTask):
    def execute(self, *, val=None):
        return val


class DivideTask(EOTask):
    def execute(self, x, y, *, z=0):
        return x / y + z


class Inc(EOTask):
    def execute(self, x, *, d=1):
        return x + d


class Pow(EOTask):
    def execute(self, x, *, n=2):
        return x ** n


def test_workflow_arguments():
    input_task1 = InputTask()
    input_task2 = InputTask()
    divide_task = DivideTask()
    output_task = OutputTask(name='output')

    workflow = EOWorkflow([
        (input_task1, []),
        (input_task2, [], 'some name'),
        Dependency(task=divide_task, inputs=[input_task1, input_task2], name='some name'),
        (output_task, [divide_task])
    ])

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        k2future = {
            k: executor.submit(
                workflow.execute, {input_task1: {'val': k ** 3}, input_task2: {'val': k ** 2}}
            ) for k in range(2, 100)
        }
        executor.shutdown()
        for k in range(2, 100):
            assert k2future[k].result().outputs['output'] == k

    result1 = workflow.execute({input_task1: {'val': 15}, input_task2: {'val': 3}})
    assert result1.outputs['output'] == 5

    result2 = workflow.execute({input_task1: {'val': 6}, input_task2: {'val': 3}})
    assert result2.outputs['output'] == 2

    result3 = workflow.execute({input_task1: {'val': 6}, input_task2: {'val': 3}, divide_task: {'z': 1}})
    assert result3.outputs[output_task.name] == 3


def test_linear_workflow():
    in_task, in_task_name = InputTask(), 'My input task'
    inc_task1 = Inc()
    inc_task2 = Inc()
    pow_task = Pow()
    output_task = OutputTask(name='output')

    eow = LinearWorkflow((in_task, in_task_name), inc_task1, inc_task2, pow_task, output_task)
    res = eow.execute({in_task: {'val': 2}, inc_task1: {'d': 2}, pow_task: {'n': 3}})
    assert res.outputs['output'] == (2 + 2 + 1) ** 3

    task_map = eow.get_tasks()
    assert in_task_name in task_map, f"A task with name '{in_task_name}' should be among tasks"
    assert task_map[in_task_name] == in_task, f"A task with name '{in_task_name}' should map into {in_task_name}"


def test_get_tasks():
    in_task = InputTask()
    inc_task0 = Inc()
    inc_task1 = Inc()
    inc_task2 = Inc()
    output_task = OutputTask(name='out')

    task_names = ['InputTask', 'Inc', 'Inc_1', 'Inc_2', 'OutputTask']
    eow = LinearWorkflow(in_task, inc_task0, inc_task1, inc_task2, output_task)

    returned_tasks = eow.get_tasks()

    assert sorted(task_names) == sorted(returned_tasks), 'Returned tasks differ from original tasks'

    arguments_dict = {in_task: {'val': 2}, inc_task0: {'d': 2}}
    workflow_res = eow.execute(arguments_dict)

    manual_res = []
    for _, task in enumerate(returned_tasks.values()):
        manual_res = [task.execute(*manual_res, **arguments_dict.get(task, {}))]

    assert workflow_res.outputs['out'] == manual_res[0], 'Manually running returned tasks produces different results.'


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10),
            st.integers(min_value=0, max_value=10)
        ).filter(lambda p: p[0] != p[1]),
        min_size=1, max_size=110
    )
)
def test_resolve_dependencies(edges):
    dag = DirectedGraph.from_edges(edges)

    if DirectedGraph._is_cyclic(dag):
        with pytest.raises(CyclicDependencyError):
            EOWorkflow._schedule_dependencies(dag)
    else:
        vertex_position = {vertex: i for i, vertex in enumerate(EOWorkflow._schedule_dependencies(dag))}
        assert functools.reduce(lambda P, Q: P and Q, [vertex_position[u] < vertex_position[v] for u, v in edges])


@pytest.mark.parametrize(
    'faulty_parameters',
    [
        (None,),
        (InputTask(), 'a string'),
        (InputTask(), ('something', InputTask())),
        (InputTask(), 'name', 'something else'),
        ('task', 'name')
    ]
)
def test_exceptions(faulty_parameters):
    with pytest.raises(ValueError):
        LinearWorkflow(*faulty_parameters)


def test_workflow_copying_eopatches():
    feature1 = FeatureType.DATA, 'data1'
    feature2 = FeatureType.DATA, 'data2'

    init_task = InitializeFeatureTask(
        [feature1, feature2],
        shape=(2, 4, 4, 3),
        init_value=1
    )
    remove_task1 = RemoveFeatureTask(feature1)
    remove_task2 = RemoveFeatureTask(feature2)
    output_task1 = OutputTask(name='out1')
    output_task2 = OutputTask(name='out2')

    workflow = EOWorkflow([
        (init_task, []),
        (remove_task1, [init_task]),
        (remove_task2, [init_task]),
        (output_task1, [remove_task1]),
        (output_task2, [remove_task2]),
    ])
    results = workflow.execute({init_task: (EOPatch(),)})

    eop1 = results.outputs['out1']
    eop2 = results.outputs['out2']

    assert eop1 == EOPatch(data={'data2': np.ones((2, 4, 4, 3), dtype=np.uint8)})
    assert eop2 == EOPatch(data={'data1': np.ones((2, 4, 4, 3), dtype=np.uint8)})


def test_workflows_sharing_tasks():

    in_task = InputTask()
    task1 = Inc()
    task2 = Inc()
    out_task = OutputTask(name='out')
    input_args = {in_task: {'val': 2}, task2: {'d': 2}}

    original = EOWorkflow([(in_task, []), (task1, in_task), (task2, task1), (out_task, task2)])
    task_reuse = EOWorkflow([(in_task, []), (task1, in_task), (task2, task1), (out_task, task2)])

    assert original.execute(input_args).outputs['out'] == task_reuse.execute(input_args).outputs['out']


def test_workflow_results():
    input_task = InputTask()
    output_task = OutputTask(name='out')
    workflow = LinearWorkflow(input_task, output_task)

    results = workflow.execute({input_task: {'val': 10}})

    assert isinstance(results, WorkflowResults)
    assert results.outputs == {'out': 10}

    assert isinstance(results.start_time, dt.datetime)
    assert isinstance(results.end_time, dt.datetime)
    assert results.start_time < results.end_time < dt.datetime.now()

    assert isinstance(results.stats, dict)
    assert len(results.stats) == 2
    for task in [input_task, output_task]:
        stats_uid = task.private_task_config.uid
        assert isinstance(results.stats.get(stats_uid), TaskStats)
