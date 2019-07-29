import unittest
import logging
import functools
import concurrent.futures
from io import StringIO

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

        workflow = EOWorkflow(dependencies=[
            Dependency(task=input_task1, inputs=[]),
            Dependency(task=input_task2, inputs=[]),
            Dependency(task=divide_task, inputs=[input_task1, input_task2])
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
        inc_task = Inc()
        pow_task = Pow()
        eow = LinearWorkflow(in_task, inc_task, pow_task)
        res = eow.execute({
            in_task: {'val': 2},
            inc_task: {'d': 2},
            pow_task: {'n': 3}
        })
        self.assertEqual(res[pow_task], (2+2)**3)

    def test_get_tasks(self):
        in_task = InputTask()
        inc_task = Inc()
        pow_task = Pow()

        task_names = ['InputTask', 'Inc', 'Pow']
        workflow_tasks = [in_task, inc_task, pow_task]
        eow = LinearWorkflow(*workflow_tasks)

        returned_tasks = eow.get_tasks()

        # check if tasks are present
        for task_name in task_names:
            self.assertIn(task_name, returned_tasks.keys())

        # check if tasks still work
        arguments_dict = {
            in_task: {'val': 2},
            inc_task: {'d': 2},
            pow_task: {'n': 3}
        }

        res_workflow = eow.execute(arguments_dict)
        res_workflow_value = [res_workflow[key] for key in res_workflow.keys()][0]

        for idx, task in enumerate(workflow_tasks):
            if idx == 0:
                res_tasks_value = task.execute(**arguments_dict[task])
            else:
                res_tasks_value = task.execute(res_tasks_value, **arguments_dict[task])

        self.assertEqual(res_workflow_value, res_tasks_value)

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


class TestGraph(unittest.TestCase):

    def setUp(self):
        input_task1 = InputTask()
        input_task2 = InputTask()
        divide_task = DivideTask()

        self.workflow = EOWorkflow(dependencies=[
            Dependency(task=input_task1, inputs=[]),
            Dependency(task=input_task2, inputs=[]),
            Dependency(task=divide_task, inputs=[input_task1, input_task2])
        ])

    def test_graph_nodes_and_edges(self):
        dot = self.workflow.get_dot()
        dot_file = StringIO()
        dot_file.write(dot.source)
        dot_file.seek(0)

        digraph = self.workflow.dependency_graph()


class TestWorkflowResults(unittest.TestCase):
    pass


class TestUniqueIdGenerator(unittest.TestCase):
    def test_exceeding_max_uuids(self):
        _UniqueIdGenerator.MAX_UUIDS = 10

        id_gen = _UniqueIdGenerator()
        for _ in range(_UniqueIdGenerator.MAX_UUIDS):
            id_gen.next()

        with self.assertRaises(MemoryError):
            id_gen.next()


if __name__ == '__main__':
    unittest.main()
