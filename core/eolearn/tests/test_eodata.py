import unittest
import logging
import os
import shutil
import datetime

from eolearn.core.eodata import EOPatch, FeatureType

import numpy as np


logging.basicConfig(level=logging.DEBUG)


class TestEOPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir('./test_outputs')

    def test_add_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands)

        self.assertTrue(np.array_equal(eop.data['bands'], bands), msg="Data numpy array not stored")

    def test_get_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands)

        eop_bands = eop.get_feature(FeatureType.DATA, 'bands')

        self.assertTrue(np.array_equal(eop_bands, bands), msg="Data numpy array not returned properly")

    def test_remove_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands)
        eop.add_feature(attr_type=FeatureType.DATA, field='bands_copy', value=bands)

        self.assertTrue('bands_copy' in eop.data.keys(), msg="Data numpy array not stored")
        self.assertTrue('bands_copy' in eop.features[FeatureType.DATA], msg="Feature key not stored")

        eop.remove_feature(attr_type=FeatureType.DATA, field='bands_copy')
        self.assertFalse('bands_copy' in eop.data.keys(), msg="Data numpy array not removed")
        self.assertFalse('bands_copy' in eop.features[FeatureType.DATA], msg="Feature key not removed")
        self.assertTrue('bands' in eop.data.keys(), msg="Data numpy array not stored after removal of other feature")

    def test_check_dims(self):
        bands_2d = np.arange(3*3).reshape(3, 3)
        bands_3d = np.arange(3*3*3).reshape(3, 3, 3)
        with self.assertRaises(ValueError):
            EOPatch(data={'bands': bands_2d})
        eop = EOPatch()
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands_2d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.MASK, field='mask', value=bands_2d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='bands_timeless', value=bands_2d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.MASK_TIMELESS, field='mask_timeless', value=bands_2d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.LABEL, field='label', value=bands_3d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.SCALAR, field='scalar', value=bands_3d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.LABEL_TIMELESS, field='label_timeless', value=bands_2d)
        with self.assertRaises(ValueError):
            eop.add_feature(attr_type=FeatureType.SCALAR_TIMELESS, field='scalar_timeless', value=bands_2d)

    def test_concatenate(self):
        eop1 = EOPatch()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop1.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands1)

        eop2 = EOPatch()
        bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
        eop2.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands2)

        eop = EOPatch.concatenate(eop1, eop2)

        self.assertTrue(np.array_equal(eop.data['bands'], np.concatenate((bands1, bands2), axis=0)),
                        msg="Array mismatch")

    def test_get_features(self):
        eop = EOPatch()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands1)
        self.assertEqual(eop.features[FeatureType.DATA]['bands'], (2, 3, 3, 2))

    def test_concatenate_prohibit_key_mismatch(self):
        eop1 = EOPatch()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop1.add_feature(attr_type=FeatureType.DATA, field='bands', value=bands1)

        eop2 = EOPatch()
        bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
        eop2.add_feature(attr_type=FeatureType.DATA, field='measurements', value=bands2)

        with self.assertRaises(ValueError):
            EOPatch.concatenate(eop1, eop2)

    def test_concatenate_leave_out_timeless_mismatched_keys(self):
        eop1 = EOPatch()
        mask1 = np.arange(3*3*2).reshape(3, 3, 2)
        eop1.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask1', value=mask1)
        eop1.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask', value=5*mask1)

        eop2 = EOPatch()
        mask2 = np.arange(3*3*2).reshape(3, 3, 2)
        eop2.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask2', value=mask2)
        eop2.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask', value=5*mask1)  # add mask1 to eop2

        eop = EOPatch.concatenate(eop1, eop2)

        self.assertTrue('mask1' not in eop.data_timeless)
        self.assertTrue('mask2' not in eop.data_timeless)

        self.assertTrue('mask' in eop.data_timeless)

    def test_concatenate_leave_out_keys_with_mismatched_value(self):
        mask = np.arange(3*3*2).reshape(3, 3, 2)

        eop1 = EOPatch()
        eop1.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask', value=mask)
        eop1.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='nask', value=3*mask)

        eop2 = EOPatch()
        eop2.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask', value=mask)
        eop2.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='nask', value=5*mask)

        eop = EOPatch.concatenate(eop1, eop2)

        self.assertTrue('mask' in eop.data_timeless)
        self.assertFalse('nask' in eop.data_timeless)

    def test_equals(self):
        eop1 = EOPatch(data={'bands': np.arange(2*3*3*2).reshape(2, 3, 3, 2)})
        eop2 = EOPatch(data={'bands': np.arange(2*3*3*2).reshape(2, 3, 3, 2)})

        self.assertTrue(eop1 == eop2)

        eop1.add_feature(FeatureType.DATA_TIMELESS, field='dem', value=np.arange(3*3*2).reshape(3, 3, 2))

        self.assertFalse(eop1 == eop2)

    def test_save_load(self):
        eop1 = EOPatch()
        mask1 = np.arange(3*3*2).reshape(3, 3, 2)
        eop1.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask1', value=mask1)
        eop1.add_feature(attr_type=FeatureType.DATA_TIMELESS, field='mask', value=5 * mask1)

        eop1.save('./test_outputs/eop1/')

        eop2 = EOPatch.load('./test_outputs/eop1')

        self.assertEqual(eop1, eop2)

    def test_timestamp_consolidation(self):
        # 10 frames
        timestamps = [datetime.datetime(2017, 1, 1, 10, 4, 7),
                      datetime.datetime(2017, 1, 4, 10, 14, 5),
                      datetime.datetime(2017, 1, 11, 10, 3, 51),
                      datetime.datetime(2017, 1, 14, 10, 13, 46),
                      datetime.datetime(2017, 1, 24, 10, 14, 7),
                      datetime.datetime(2017, 2, 10, 10, 1, 32),
                      datetime.datetime(2017, 2, 20, 10, 6, 35),
                      datetime.datetime(2017, 3, 2, 10, 0, 20),
                      datetime.datetime(2017, 3, 12, 10, 7, 6),
                      datetime.datetime(2017, 3, 15, 10, 12, 14)]

        data = np.random.rand(10, 100, 100, 3)
        mask = np.random.randint(0, 2, (10, 100, 100, 1))
        mask_timeless = np.random.randint(10, 20, (100, 100, 1))
        scalar = np.random.rand(10, 1)

        eop = EOPatch(timestamp=timestamps,
                      data={'DATA': data},
                      mask={'MASK': mask},
                      scalar={'SCALAR': scalar},
                      mask_timeless={'MASK_TIMELESS': mask_timeless})

        good_timestamps = timestamps.copy()
        del good_timestamps[0]
        del good_timestamps[-1]
        good_timestamps.append(datetime.datetime(2017, 12, 1))

        removed_frames = eop.consolidate_timestamps(good_timestamps)

        self.assertEqual(good_timestamps[:-1], eop.timestamp)
        self.assertEqual(len(removed_frames), 2)
        self.assertTrue(timestamps[0] in removed_frames)
        self.assertTrue(timestamps[-1] in removed_frames)
        self.assertTrue(np.array_equal(data[1:-1, ...], eop.data['DATA']))
        self.assertTrue(np.array_equal(mask[1:-1, ...], eop.mask['MASK']))
        self.assertTrue(np.array_equal(scalar[1:-1, ...], eop.scalar['SCALAR']))
        self.assertTrue(np.array_equal(mask_timeless, eop.mask_timeless['MASK_TIMELESS']))

    @classmethod
    def tearDown(cls):
        shutil.rmtree('./test_outputs/', ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
