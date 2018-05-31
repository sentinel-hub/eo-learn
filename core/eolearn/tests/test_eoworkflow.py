import unittest
import logging
import functools

from hypothesis import given, strategies as st

from eolearn.core.eotask import EOTask
from eolearn.core.eoworkflow import EOWorkflow, Dependency, _UniqueIdGenerator, CyclicDependencyError
from eolearn.core.graph import DirectedGraph


logging.basicConfig(level=logging.DEBUG)


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
            Dependency(transform=input_task1, inputs=[]),
            Dependency(transform=input_task2, inputs=[]),
            Dependency(transform=divide_task, inputs=[input_task1, input_task2])
        ])

        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as e:
            r = {
                k: e.submit(
                    workflow.execute,
                    {
                        input_task1: {'val': k ** 3},
                        input_task2: {'val': k ** 2}
                    }
                ) for k in range(2, 100)
            }
            e.shutdown()
            for k in range(2, 100):
                print("ID {}: t({}, {}) = {}".format(k, k**3, k**2, r[k].result()[divide_task]))
                self.assertEqual(r[k].result()[divide_task], k)

        result1 = workflow.execute({
            input_task1: {'val': 15},
            input_task2: {'val': 3}
        })

        self.assertEqual(result1[divide_task], 5)

        result2 = workflow.execute({
            input_task1: {'val': 6},
            input_task2: {'val': 3}
        })
        self.assertEqual(result2[divide_task.uuid], 2)

        result3 = workflow.execute({
            input_task1: {'val': 6},
            input_task2: {'val': 3},
            divide_task: {'z': 1}
        })

        self.assertEqual(result3[divide_task.uuid], 3)

    def test_linear_workflow(self):
        in_task = InputTask()
        inc = Inc()
        pow = Pow()
        eow = EOWorkflow.make_linear_workflow(in_task, inc, pow)
        res = eow.execute({
            in_task: {'val': 2},
            inc: {'d': 2},
            pow: {'n': 3}
        })

        self.assertEqual(res[pow.uuid], (2+2)**3)

    def test_trivial_workflow(self):
        t = DummyTask()
        workflow = EOWorkflow(dependencies=[
            Dependency(transform=t, inputs=[])
        ])

        result = workflow.execute()
        self.assertEqual(result[t.uuid], 42)

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
                _ = EOWorkflow._resolve_dependencies(dag)
        else:
            ver2pos = {u: i for i, u in enumerate(EOWorkflow._resolve_dependencies(dag))}
            self.assertTrue(functools.reduce(
                lambda P, Q: P and Q,
                [ver2pos[u] < ver2pos[v] for u, v in edges]
            ))


class TestWorkflowResult(unittest.TestCase):
    pass


class TestUniqueIdGenerator(unittest.TestCase):
    def test_fails_after_exceeding_max_uuids(self):
        _UniqueIdGenerator.MAX_UUIDS = 10

        id_gen = _UniqueIdGenerator()
        for _ in range(_UniqueIdGenerator.MAX_UUIDS):
            id_gen.next()

        with self.assertRaises(MemoryError):
            id_gen.next()


if __name__ == '__main__':
    unittest.main()
