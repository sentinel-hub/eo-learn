"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import copy

import pytest

from eolearn.core import EOTask


class TwoParamException(BaseException):
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
            raise TwoParamException(1, 2)

        # try raising a subclassed exception with an unsupported __init__ arguments signature without initializing it
        if self.task_arg == 'test_exception_fail':
            raise TwoParamException

        # raise one of the standard errors
        if self.task_arg == 'value_error':
            raise ValueError('Testing value error.')

        return self.task_arg + ' ' + exec_param


class PlusOneTask(EOTask):
    @staticmethod
    def execute(x):
        return x + 1


class PlusConstSquaredTask(EOTask):
    def __init__(self, const):
        self.const = const

    def execute(self, x):
        return (x + self.const)**2


class SelfRecursiveTask(EOTask):
    def __init__(self, x, *args, **kwargs):
        self.recursive = self
        self.arg_x = x
        self.args = args
        self.kwargs = kwargs

    def execute(self, _):
        return self.arg_x


def test_call_equals_execute():
    task = PlusOneTask()
    assert task(1) == task.execute(1), 't(x) should given the same result as t.execute(x)'
    task = PlusConstSquaredTask(20)
    assert task(14) == task.execute(14), 't(x) should given the same result as t.execute(x)'


@pytest.mark.parametrize(
    ('parameter', 'exception_type'), [('test_exception_fail', TypeError), ('value_error', ValueError)]
)
def test_execution_handling(parameter, exception_type):
    task = ExceptionTestingTask('test_exception')
    with pytest.raises(TwoParamException):
        task('test')

    task = ExceptionTestingTask('success')
    assert task('test') == 'success test'

    task = ExceptionTestingTask(parameter)
    with pytest.raises(exception_type):
        task('test')

    try:
        task('test')
    except exception_type as exception:
        assert str(exception).startswith('During execution of task ExceptionTestingTask: ')
