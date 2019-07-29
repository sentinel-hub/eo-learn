import unittest
import os
import logging
import tempfile
import datetime

from eolearn.core import EOTask, EOWorkflow, Dependency, EOExecutor


logging.basicConfig(level=logging.DEBUG)


class ExampleTask(EOTask):

    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.info('Info statement of Example task with kwargs: %s', kwargs)
        my_logger.warning('Warning statement of Example task with kwargs: %s', kwargs)
        my_logger.debug('Debug statement of Example task with kwargs: %s', kwargs)

        if 'arg1' in kwargs and kwargs['arg1'] is None:
            raise Exception


class TestEOExecutor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        task = ExampleTask()
        cls.workflow = EOWorkflow([(task, []),
                                   Dependency(task=ExampleTask(), inputs=[task, task])])

        cls.execution_args = [
            {task: {'arg1': 1}},
            {},
            {task: {'arg1': 3, 'arg3': 10}},
            {task: {'arg1': None}}
        ]

    def test_execution_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executor = EOExecutor(self.workflow, self.execution_args, save_logs=True, logs_folder=tmp_dir_name)
            executor.run()

            self.assertEqual(len(executor.execution_logs), 4)
            for log in executor.execution_logs:
                self.assertTrue(len(log.split()) >= 3)

    def test_execution_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executor = EOExecutor(self.workflow, self.execution_args, logs_folder=tmp_dir_name)
            executor.run(workers=2)

            self.assertEqual(len(executor.execution_stats), 4)
            for stats in executor.execution_stats:
                for time_stat in ['start_time', 'end_time']:
                    self.assertTrue(time_stat in stats and isinstance(stats[time_stat], datetime.datetime))

    def test_execution_errors(self):
        for multiprocess in [True, False]:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                executor = EOExecutor(self.workflow, self.execution_args, logs_folder=tmp_dir_name)
                executor.run(workers=5, multiprocess=multiprocess)

                for idx, stats in enumerate(executor.execution_stats):
                    if idx != 3:
                        self.assertFalse('error' in stats, 'Workflow {} should be executed without errors'.format(idx))
                    else:
                        self.assertTrue('error' in stats and stats['error'],
                                        'This workflow should be executed with an error')

                self.assertEqual(executor.get_successful_executions(), [0, 1, 2])
                self.assertEqual(executor.get_failed_executions(), [3])

    def test_report_creation(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executor = EOExecutor(self.workflow, self.execution_args, logs_folder=tmp_dir_name, save_logs=True)
            executor.run(workers=10)
            executor.make_report()

            self.assertTrue(os.path.exists(executor.get_report_filename()), 'Execution report was not created')


if __name__ == '__main__':
    unittest.main()
