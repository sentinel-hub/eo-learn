import unittest
import logging

from eolearn.core import EOPatch, FeatureType

from eolearn.coregistration import InterpolationType, ECCRegistration

import numpy as np

logging.basicConfig(level=logging.DEBUG)


class TestEOPatch(unittest.TestCase):

    def test_registration(self):
        # Set up a dummy EOPatch to test execution of registration
        bands = np.zeros((2, 20, 20, 1))
        bands[1] = np.arange(400).reshape(1, 20, 20, 1) / 400
        bands[0] = bands[1]
        bands[1, 5:15, 5:15, :] = .5
        bands[0, 7:17, 5:15, :] = .5
        mask = np.ones((2, 20, 20, 1), dtype=np.int16)
        ndvi = np.ones((2, 20, 20, 1))
        dem = np.ones((20, 20, 1))

        eop = EOPatch()
        eop.add_feature(FeatureType.DATA, 'bands', value=bands)
        eop.add_feature(FeatureType.DATA, 'ndvi', value=ndvi)
        eop.add_feature(FeatureType.MASK, 'cm', value=mask)
        eop.add_feature(FeatureType.DATA_TIMELESS, 'dem', value=dem)

        reg = ECCRegistration((FeatureType.DATA, 'bands'), valid_mask_feature='cm',
                              interpolation_type=InterpolationType.NEAREST,
                              apply_to_features={
                                  FeatureType.DATA: {'bands', 'ndvi'},
                                  FeatureType.MASK: {'cm'}
                              })
        reop = reg.execute(eop)

        self.assertEqual(eop.data['bands'].shape, reop.data['bands'].shape,
                         msg='Shapes of .data[''bands''] do not match')
        self.assertEqual(eop.data['ndvi'].shape, reop.data['ndvi'].shape,
                         msg='Shapes of .data[''ndvi''] do not match')
        self.assertEqual(eop.mask['cm'].shape, reop.mask['cm'].shape,
                         msg='Shapes of .mask[''cm''] do not match')
        self.assertEqual(eop.data_timeless['dem'].shape, reop.data_timeless['dem'].shape,
                         msg='Shapes of .data[''bands''] do not match')
        self.assertFalse(np.allclose(eop.data['bands'], reop.data['bands']),
                         msg='Registration did not warp .data[''bands'']')
        self.assertFalse(np.allclose(eop.data['ndvi'], reop.data['ndvi']),
                         msg='Registration did not warp .data[''ndvi'']')
        self.assertFalse(np.allclose(eop.mask['cm'], reop.mask['cm']),
                         msg='Registration did not warp .mask[''cm'']')
        self.assertTrue(np.allclose(eop.data_timeless['dem'], reop.data_timeless['dem']),
                        msg='Registration did warp data_timeless')


if __name__ == '__main__':
    unittest.main()
