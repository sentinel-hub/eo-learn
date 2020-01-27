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
