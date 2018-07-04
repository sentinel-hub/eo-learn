import unittest
import logging

from eolearn.core import eotask
from eolearn.core.eodata import EOPatch, FeatureType

import numpy as np


logging.basicConfig(level=logging.DEBUG)


class TestEOTask(unittest.TestCase):
    class PlusOneTask(eotask.EOTask):
        def execute(self, x):
            return x + 1

    def test_call_equals_transform(self):
        t = self.PlusOneTask()
        self.assertEqual(t(1), t.execute(1), msg="t(x) should given the same result as t.execute(x)")


class TestEOChainedTask(unittest.TestCase):
    class AddOneTask(eotask.EOTask):
        def execute(self, x):
            return x+1

    def test_chained(self):
        add_two = eotask.EOChainedTask([
            self.AddOneTask(),
            self.AddOneTask()
        ])

        for i in range(5):
            self.assertEqual(add_two(i), i + 2)


if __name__ == '__main__':
    unittest.main()
