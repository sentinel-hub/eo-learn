import unittest
import logging
import tempfile

from eolearn.core import EOTask, EOWorkflow, Dependency, EOExecutor


logging.basicConfig(level=logging.DEBUG)


class ExampleTask(EOTask):

    def execute(self):
        pass


class RaiserErrorTask(EOTask):

    def execute(self):
        raise Exception()


class TestEOExecutor(unittest.TestCase):

    def test_execution_logs(self):
        task = ExampleTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(task=task, inputs=[]),
        ])

        execution_args = [
            {'arg1': 1},
            {'arg1': 2}
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            executor = EOExecutor(workflow, execution_args, tmpdirname)
            executor.run()

            self.assertEqual(len(executor.execution_logs), 2)

    def test_execution_stats(self):
        task = ExampleTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(task=task, inputs=[]),
        ])

        execution_args = [
            {'arg1': 1},
            {'arg1': 2}
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            executor = EOExecutor(workflow, execution_args, tmpdirname)
            executor.run()

            self.assertEqual(len(executor.execution_stats), 2)

    def test_execution_errors(self):
        task = RaiserErrorTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(task=task, inputs=[]),
        ])

        execution_args = [
            {'arg1': 1}
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            executor = EOExecutor(workflow, execution_args, tmpdirname)
            executor.run()

            self.assertTrue('error' in executor.execution_stats[0])

    def test_report_creation(self):
        task = ExampleTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(task=task, inputs=[]),
        ])

        execution_args = [
            {'arg1': 1}
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            executor = EOExecutor(workflow, execution_args, tmpdirname)
            executor.run()

            self.assertIsNotNone(executor.make_report())


if __name__ == '__main__':
    unittest.main()
