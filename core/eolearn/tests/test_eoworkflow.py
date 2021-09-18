"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import logging
import functools
import concurrent.futures

from hypothesis import given, strategies as st

from eolearn.core import EOTask, EOWorkflow, Dependency, WorkflowResults, LinearWorkflow
from eolearn.core.eoworkflow import CyclicDependencyError, _UniqueIdGenerator
from eolearn.core.graph import DirectedGraph


logging.basicConfig(level=logging.INFO)


class InputTask(EOTask):
    def execute(self, *, val=None):
        return val


class DivideTask(EOTask):
    def execute(self, x, y, *, z=0):
        return x / y + z


class AddTask(EOTask):
    def execute(self, x, y):
        return x + y


class MulTask(EOTask):
    def execute(self, x, y):
        return x * y


class Inc(EOTask):
    def execute(self, x, *, d=1):
        return x + d


class Pow(EOTask):
    def execute(self, x, *, n=2):
        return x ** n


class DummyTask(EOTask):
    def execute(self):
        return 42


class TestEOWorkflow(unittest.TestCase):

    def test_workflow_arguments(self):
        input_task1 = InputTask()
        input_task2 = InputTask()
        divide_task = DivideTask()

        workflow = EOWorkflow([
            (input_task1, []),
            (input_task2, [], 'some name'),
            Dependency(task=divide_task, inputs=[input_task1, input_task2], name='some name')
        ])

        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            k2future = {
                k: executor.submit(
                    workflow.execute,
                    {
                        input_task1: {'val': k ** 3},
                        input_task2: {'val': k ** 2}
                    }
                ) for k in range(2, 100)
            }
            executor.shutdown()
            for k in range(2, 100):
                future = k2future[k]
                self.assertEqual(future.result()[divide_task], k)

        result1 = workflow.execute({
            input_task1: {'val': 15},
            input_task2: {'val': 3}
        })
        self.assertEqual(result1[divide_task], 5)

        result2 = workflow.execute({
            input_task1: {'val': 6},
            input_task2: {'val': 3}
        })
        self.assertEqual(result2[divide_task], 2)

        result3 = workflow.execute({
            input_task1: {'val': 6},
            input_task2: {'val': 3},
            divide_task: {'z': 1}
        })

        self.assertEqual(result3[divide_task], 3)

    def test_linear_workflow(self):
        in_task = InputTask()
        in_task_name = 'My input task'
        inc_task = Inc()
        pow_task = Pow()
        eow = LinearWorkflow((in_task, in_task_name), inc_task, inc_task, pow_task)
        res = eow.execute({
            in_task: {'val': 2},
            inc_task: {'d': 2},  # Note that this will assign value only to one instance of Inc task
            pow_task: {'n': 3}
        })
        self.assertEqual(res[pow_task], (2 + 2 + 1) ** 3)

        task_map = eow.get_tasks()
        self.assertTrue(in_task_name in task_map, f"A task with name '{in_task_name}' should be amongst tasks")
        self.assertEqual(task_map[in_task_name], in_task,
                         f"A task with name '{in_task_name}' should map into {in_task}")

    def test_get_tasks(self):
        in_task = InputTask()
        inc_task = Inc()

        task_names = ['InputTask', 'Inc', 'Inc_1', 'Inc_2']
        eow = LinearWorkflow(in_task, inc_task, inc_task, inc_task)

        returned_tasks = eow.get_tasks()

        # check if tasks are present
        self.assertEqual(sorted(task_names), sorted(returned_tasks))

        # check if tasks still work
        arguments_dict = {
            in_task: {'val': 2},
            inc_task: {'d': 2}
        }

        res_workflow = eow.execute(arguments_dict)
        res_workflow_value = list(res_workflow.values())

        res_tasks_values = []
        for idx, task in enumerate(returned_tasks.values()):
            res_tasks_values = [task.execute(*res_tasks_values, **arguments_dict.get(task, {}))]

        self.assertEqual(res_workflow_value, res_tasks_values)

    def test_trivial_workflow(self):
        task = DummyTask()
        dep = Dependency(task, [])
        workflow = EOWorkflow([dep])

        result = workflow.execute()

        self.assertTrue(isinstance(result, WorkflowResults))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result.keys()), 1)
        self.assertEqual(len(result.values()), 1)
        items = list(result.items())
        self.assertEqual(len(items), 1)
        self.assertTrue(isinstance(items[0][0], EOTask))
        self.assertEqual(items[0][1], 42)
        self.assertEqual(result[dep], 42)

        expected_repr = 'WorkflowResults(\n  Dependency(DummyTask):\n    42\n)'
        self.assertEqual(repr(result), expected_repr)

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=10),
                st.integers(min_value=0, max_value=10)
            ).filter(
                lambda p: p[0] != p[1]
            ),
            min_size=1,
            max_size=110
        )
    )
    def test_resolve_dependencies(self, edges):
        dag = DirectedGraph.from_edges(edges)
        if DirectedGraph._is_cyclic(dag):
            with self.assertRaises(CyclicDependencyError):
                _ = EOWorkflow._schedule_dependencies(dag)
        else:
            ver2pos = {u: i for i, u in enumerate(EOWorkflow._schedule_dependencies(dag))}
            self.assertTrue(functools.reduce(
                lambda P, Q: P and Q,
                [ver2pos[u] < ver2pos[v] for u, v in edges]
            ))

    def test_exceptions(self):

        for params in [(None,),
                       (InputTask(), 'a string'),
                       (InputTask(), ('something', InputTask())),
                       ((InputTask(), 'name', 'something else'),),
                       (('task', 'name'),)]:
            with self.assertRaises(ValueError):
                LinearWorkflow(*params)


class TestUniqueIdGenerator(unittest.TestCase):
    def test_exceeding_max_uuids(self):
        _UniqueIdGenerator.MAX_UUIDS = 10

        id_gen = _UniqueIdGenerator()
        for _ in range(_UniqueIdGenerator.MAX_UUIDS):
            id_gen.get_next()

        with self.assertRaises(MemoryError):
            id_gen.get_next()


if __name__ == '__main__':
    unittest.main()
