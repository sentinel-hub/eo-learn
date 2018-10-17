import unittest
import logging
import numpy as np

from eolearn.core.eodata import EOPatch, FeatureType

from eolearn.ml_tools import MorphologicalOperations, MorphologicalStructFactory, MorphologicalFilterTask


logging.basicConfig(level=logging.DEBUG)


class TestEOPatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.patch = EOPatch()

        mask = np.random.randint(20, size=(10, 100, 100, 3))
        timeless_mask = np.random.randint(20, 50, size=(100, 100, 5))

        cls.mask_name = 'mask'
        cls.timeless_mask_name = 'timeless_mask'
        cls.patch.add_feature(FeatureType.MASK, cls.mask_name, value=mask)
        cls.patch.add_feature(FeatureType.MASK_TIMELESS, cls.timeless_mask_name, value=timeless_mask)

    def test_postprocessing(self):
        for morph_operation in MorphologicalOperations:
            with self.subTest(msg='Test case {}'.format(morph_operation.name)):
                for feature_type, feature_name in [(FeatureType.MASK, self.mask_name),
                                                   (FeatureType.MASK_TIMELESS, self.timeless_mask_name)]:
                    for struct_elem in [None, MorphologicalStructFactory.get_disk(5),
                                        MorphologicalStructFactory.get_rectangle(5, 6)]:

                        task = MorphologicalFilterTask((feature_type, feature_name), morph_operation, struct_elem)
                        self.patch = task.execute(self.patch)


if __name__ == '__main__':
    unittest.main()
