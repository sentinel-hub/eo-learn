"""
Credits:
Copyright (c) 2018-2019 Mark Bogataj, Filip Koprivec (Jožef Stefan Institute)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
from eolearn.core import EOPatch, FeatureType
from eolearn.features import AdaptiveThresholdMethod, SimpleThresholdMethod, ThresholdType, \
    Thresholding, OperatorEdgeDetection, SobelOperator, ScharrOperator, ScharrFourierOperator, \
    Prewitt3Operator, Prewitt4Operator, RobertsCrossOperator, KayyaliOperator, KirschOperator


import os.path


class TestEdgeDetectionTasks(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    def test_thresholding_edge_detection(self):
        patch = EOPatch.load(self.TEST_PATCH_FILENAME)
        task = Thresholding((FeatureType.DATA, "random"), [0,2,1])

        task.execute(patch)

    def test_operator_edge_detection(self):
        patch = EOPatch.load(self.TEST_PATCH_FILENAME)
        task = KirschOperator((FeatureType.DATA, "ndvi"))

        task.execute(patch)


if __name__ == '__main__':
    unittest.main()
