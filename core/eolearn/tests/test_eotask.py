"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from eolearn.core import EOTask


class PlusOneTask(EOTask):
    @staticmethod
    def execute(x):
        return x + 1


class PlusConstSquaredTask(EOTask):
    def __init__(self, const):
        self.const = const

    def execute(self, x):
        return (x + self.const) ** 2


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
    assert task(1) == task.execute(1), "t(x) should given the same result as t.execute(x)"
    task = PlusConstSquaredTask(20)
    assert task(14) == task.execute(14), "t(x) should given the same result as t.execute(x)"
