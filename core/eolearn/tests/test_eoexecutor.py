"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import os
import logging
import tempfile
import datetime
import concurrent.futures
import multiprocessing
import time

from eolearn.core import (
    EOTask, EOWorkflow, Dependency, EOExecutor, WorkflowResults, execute_with_mp_lock, LinearWorkflow
)


logging.basicConfig(level=logging.DEBUG)


class ExampleTask(EOTask):

    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.debug('Debug statement of Example task with kwargs: %s', kwargs)
        my_logger.info('Info statement of Example task with kwargs: %s', kwargs)
        my_logger.warning('Warning statement of Example task with kwargs: %s', kwargs)
        my_logger.critical('Super important log')

        if 'arg1' in kwargs and kwargs['arg1'] is None:
            raise Exception


class FooTask(EOTask):

    @staticmethod
    def execute(*_, **__):
        return 42


class KeyboardExceptionTask(EOTask):

    @staticmethod
    def execute(*_, **__):
        raise KeyboardInterrupt


class CustomLogFilter(logging.Filter):
    """ A custom filter that keeps only logs with level warning or critical
    """
    def filter(self, record):
        return record.levelno >= logging.WARNING


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

    def test_execution_logs_single_process(self):
        self._run_and_test_execution(workers=1, multiprocess=True, filter_logs=False)
        self._run_and_test_execution(workers=1, multiprocess=False, filter_logs=True)

    def test_execution_logs_multiprocess(self):
        self._run_and_test_execution(workers=5, multiprocess=True, filter_logs=False)
        self._run_and_test_execution(workers=3, multiprocess=True, filter_logs=True)

    def test_execution_logs_multithread(self):
        self._run_and_test_execution(workers=3, multiprocess=False, filter_logs=False)
        self._run_and_test_execution(workers=2, multiprocess=False, filter_logs=True)

    def _run_and_test_execution(self, workers, multiprocess, filter_logs):
        for execution_names in [None, [4, 'x', 'y', 'z']]:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                executor = EOExecutor(self.workflow, self.execution_args, save_logs=True,
                                      logs_folder=tmp_dir_name,
                                      logs_filter=CustomLogFilter() if filter_logs else None,
                                      execution_names=execution_names)
                executor.run(workers=workers, multiprocess=multiprocess)

                self.assertEqual(len(executor.execution_logs), 4)
                for log in executor.execution_logs:
                    self.assertTrue(len(log.split()) >= 3)

                log_filenames = sorted(os.listdir(executor.report_folder))
                self.assertEqual(len(log_filenames), 4)

                if execution_names:
                    for name, log_filename in zip(execution_names, log_filenames):
                        self.assertTrue(log_filename == f'eoexecution-{name}.log')

                log_path = os.path.join(executor.report_folder, log_filenames[0])
                with open(log_path, 'r') as fp:
                    line_count = len(fp.readlines())
                    expected_line_count = 2 if filter_logs else 12
                    self.assertEqual(line_count, expected_line_count)

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
                        self.assertFalse('error' in stats, f'Workflow {idx} should be executed without errors')
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

    def test_keyboardInterrupt(self):
        exeption_task = KeyboardExceptionTask()
        workflow = LinearWorkflow(exeption_task)
        execution_args = []
        for _ in range(10):
            execution_args.append({exeption_task: {'arg1': 1}})

        run_args = [{'workers':1},
                    {'workers':3, 'multiprocess':True},
                    {'workers':3, 'multiprocess':False}]
        for arg in run_args:
            self.assertRaises(KeyboardInterrupt, EOExecutor(workflow, execution_args).run, **arg)


class TestExecuteWithMultiprocessingLock(unittest.TestCase):

    WORKERS = 5

    @staticmethod
    def logging_function(_=None):
        """ Logs start, sleeps for 0.5s, logs end
        """
        logging.info(multiprocessing.current_process().name)
        time.sleep(0.5)
        logging.info(multiprocessing.current_process().name)

    def test_with_lock(self):
        with tempfile.NamedTemporaryFile() as fp:
            logger = logging.getLogger()
            handler = logging.FileHandler(fp.name)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.WORKERS) as pool:
                pool.map(execute_with_mp_lock, [self.logging_function] * self.WORKERS)

            handler.close()
            logger.removeHandler(handler)

            with open(fp.name, 'r') as log_file:
                lines = log_file.read().strip('\n ').split('\n')

            self.assertEqual(len(lines), 2 * self.WORKERS)
            for idx in range(self.WORKERS):
                self.assertTrue(lines[2 * idx], lines[2 * idx + 1])
            for idx in range(1, self.WORKERS):
                self.assertNotEqual(lines[2 * idx - 1], lines[2 * idx])

    def test_without_lock(self):
        with tempfile.NamedTemporaryFile() as fp:
            logger = logging.getLogger()
            handler = logging.FileHandler(fp.name)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.WORKERS) as pool:
                pool.map(self.logging_function, [None] * self.WORKERS)

            handler.close()
            logger.removeHandler(handler)

            with open(fp.name, 'r') as log_file:
                lines = log_file.read().strip('\n ').split('\n')

            self.assertEqual(len(lines), 2 * self.WORKERS)
            self.assertEqual(len(set(lines[: self.WORKERS])), self.WORKERS, msg='All processes should start')
            self.assertEqual(len(set(lines[self.WORKERS:])), self.WORKERS, msg='All processes should finish')


if __name__ == '__main__':
    unittest.main()
