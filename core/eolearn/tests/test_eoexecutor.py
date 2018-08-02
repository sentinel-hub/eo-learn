import unittest
import logging
import os
import shutil

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
            Dependency(transform=task, inputs=[]),
        ])

        executions_args = [
            {'arg1': 1},
            {'arg1': 2}
        ]

        out_dir = 'dir'
        executor = EOExecutor(workflow, executions_args, out_dir)
        executor.run()

        self.assertEqual(len(executor.executions_logs), 2)
        shutil.rmtree(out_dir)

    def test_execution_info(self):
        task = ExampleTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(transform=task, inputs=[]),
        ])

        executions_args = [
            {'arg1': 1},
            {'arg1': 2}
        ]

        out_dir = 'dir'
        executor = EOExecutor(workflow, executions_args, out_dir)
        executor.run()

        self.assertEqual(len(executor.executions_info), 2)
        shutil.rmtree(out_dir)

    def test_execution_error(self):
        task = RaiserErrorTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(transform=task, inputs=[]),
        ])

        executions_args = [
            {'arg1': 1}
        ]

        out_dir = 'dir'
        executor = EOExecutor(workflow, executions_args, out_dir)
        executor.run()

        self.assertTrue('error' in executor.executions_info[0])
        shutil.rmtree(out_dir)

    def test_report_making(self):
        task = ExampleTask()

        workflow = EOWorkflow(dependencies=[
            Dependency(transform=task, inputs=[]),
        ])

        executions_args = [
            {'arg1': 1}
        ]

        out_dir = 'dir'
        executor = EOExecutor(workflow, executions_args, out_dir)
        executor.run()
        executor.make_report()

        report_path = os.path.join(out_dir, 'report.html')
        self.assertTrue(os.path.exists(report_path))
        shutil.rmtree(out_dir)  # TODO: fix removing folders


if __name__ == '__main__':
    unittest.main()
