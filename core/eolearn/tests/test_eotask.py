"""
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import logging

from eolearn.core import EOTask


logging.basicConfig(level=logging.DEBUG)


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


if __name__ == '__main__':
    unittest.main()
