"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import os
import logging
import tempfile
import datetime

from eolearn.core import EOTask, EOWorkflow, Dependency, EOExecutor, WorkflowResults


logging.basicConfig(level=logging.DEBUG)


class ExampleTask(EOTask):

    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.info('Info statement of Example task with kwargs: %s', kwargs)
        my_logger.warning('Warning statement of Example task with kwargs: %s', kwargs)
        my_logger.debug('Debug statement of Example task with kwargs: %s', kwargs)

        if 'arg1' in kwargs and kwargs['arg1'] is None:
            raise Exception


class FooTask(EOTask):

    @staticmethod
    def execute(*_, **__):
        return 42


class TestEOExecutor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.task = ExampleTask()
        cls.final_task = FooTask()
        cls.workflow = EOWorkflow([(cls.task, []),
                                   Dependency(task=cls.final_task, inputs=[cls.task, cls.task])])

        cls.execution_args = [
            {cls.task: {'arg1': 1}},
            {},
            {cls.task: {'arg1': 3, 'arg3': 10}},
            {cls.task: {'arg1': None}}
        ]

    def test_execution_logs(self):
        for execution_names in [None, [4, 'x', 'y', 'z']]:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                executor = EOExecutor(self.workflow, self.execution_args, save_logs=True, logs_folder=tmp_dir_name,
                                      execution_names=execution_names)
                executor.run()

                self.assertEqual(len(executor.execution_logs), 4)
                for log in executor.execution_logs:
                    self.assertTrue(len(log.split()) >= 3)

                log_filenames = sorted(os.listdir(executor.report_folder))
                self.assertEqual(len(log_filenames), 4)

                if execution_names:
                    for name, log_filename in zip(execution_names, log_filenames):
                        self.assertTrue(log_filename == 'eoexecution-{}.log'.format(name))

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

    def test_execution_results(self):
        for return_results in [True, False]:

            executor = EOExecutor(self.workflow, self.execution_args)
            results = executor.run(workers=2, multiprocess=True, return_results=return_results)

            if return_results:
                self.assertTrue(isinstance(results, list))

                for idx, workflow_results in enumerate(results):
                    if idx == 3:
                        self.assertEqual(workflow_results, None)
                    else:
                        self.assertTrue(isinstance(workflow_results, WorkflowResults))
                        self.assertEqual(workflow_results[self.final_task], 42)
                        self.assertTrue(self.task not in workflow_results)
            else:
                self.assertEqual(results, None)

    def test_exceptions(self):

        with self.assertRaises(ValueError):
            EOExecutor(self.workflow, {})

        with self.assertRaises(ValueError):
            EOExecutor(self.workflow, self.execution_args, execution_names={1, 2, 3, 4})
        with self.assertRaises(ValueError):
            EOExecutor(self.workflow, self.execution_args, execution_names=['a', 'b'])


if __name__ == '__main__':
    unittest.main()
