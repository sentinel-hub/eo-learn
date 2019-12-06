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

from eolearn.core import EOTask


logging.basicConfig(level=logging.DEBUG)


class TestException(BaseException):
    def __init__(self, param1, param2):
        # accept two parameters as opposed to BaseException, which just accepts one
        super().__init__()
        self.param1 = param1
        self.param2 = param2


class ExceptionTestingTask(EOTask):
    def __init__(self, task_arg):
        self.task_arg = task_arg

    def execute(self, exec_param):
        # try raising a subclassed exception with an unsupported __init__ arguments signature
        if self.task_arg == 'test_exception':
            raise TestException(1, 2)

        # try raising a subclassed exception with an unsupported __init__ arguments signature without initializing it
        if self.task_arg == 'test_exception_fail':
            raise TestException

        # raise one of the standard errors
        if self.task_arg == 'value_error':
            raise ValueError('Testing value error.')

        return self.task_arg + ' ' + exec_param


class TestEOTask(unittest.TestCase):
    class PlusOneTask(EOTask):

        @staticmethod
        def execute(x):
            return x + 1

    def test_call_equals_transform(self):
        t = self.PlusOneTask()
        self.assertEqual(t(1), t.execute(1), msg="t(x) should given the same result as t.execute(x)")


class TestCompositeTask(unittest.TestCase):
    class MultTask(EOTask):

        def __init__(self, num):
            self.num = num

        def execute(self, x):
            return (x + 1) * self.num

    def test_chained(self):
        composite = self.MultTask(1) * self.MultTask(2) * self.MultTask(3)

        for i in range(5):
            self.assertEqual(composite(i), 6 * i + 9)

    def test_execution_handling(self):
        task = ExceptionTestingTask('test_exception')
        self.assertRaises(TestException, task, 'test')

        task = ExceptionTestingTask('success')
        self.assertEqual(task('test'), 'success test')

        for parameter, exception_type in [('test_exception_fail', TypeError), ('value_error', ValueError)]:
            task = ExceptionTestingTask(parameter)
            self.assertRaises(exception_type, task, 'test')
            try:
                task('test')
            except exception_type as exception:
                message = str(exception)
                self.assertTrue(message.startswith('During execution of task ExceptionTestingTask: '))


if __name__ == '__main__':
    unittest.main()
