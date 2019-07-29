import unittest
import logging
import os
import datetime
import numpy as np
import tempfile
import itertools

from geopandas import GeoSeries, GeoDataFrame

from eolearn.core import EOPatch, FeatureType, FeatureTypeSet, OverwritePermission, FileFormat, BBox, CRS

logging.basicConfig(level=logging.DEBUG)


class TestEOPatchFeatureTypes(unittest.TestCase):

    PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/TestEOPatch')

    def test_loading_valid(self):
        eop = EOPatch.load(self.PATCH_FILENAME)

        repr_str = eop.__repr__()
        self.assertTrue(isinstance(repr_str, str) and len(repr_str) > 0,
                        msg='EOPatch __repr__ must return non-empty string')

    def test_numpy_feature_types(self):
        eop = EOPatch()

        data_examples = []
        for size in range(6):
            for dtype in [np.float32, np.float64, np.float, np.uint8, np.int64, np.bool]:
                data_examples.append(np.zeros((2, ) * size, dtype=dtype))

        for feature_type in FeatureTypeSet.RASTER_TYPES:
            valid_count = 0

            for data in data_examples:
                try:
                    eop[feature_type]['TEST'] = data
                    valid_count += 1
                except ValueError:
                    pass

            self.assertEqual(valid_count, 6,  # 3 * (2 - feature_type.is_discrete()),
                             msg='Feature type {} should take only a specific type of data'.format(feature_type))

    def test_vector_feature_types(self):
        eop = EOPatch()

        invalid_entries = [
            {}, [], 0, None
        ]

        for feature_type in FeatureTypeSet.VECTOR_TYPES:
            for entry in invalid_entries:
                with self.assertRaises(ValueError,
                                       msg='Invalid entry {} for {} should raise an error'.format(entry, feature_type)):
                    eop[feature_type]['TEST'] = entry

        crs_test = {'init': 'epsg:4326'}
        geo_test = GeoSeries([BBox((1, 2, 3, 4), crs=CRS.WGS84).get_geometry()], crs=crs_test)

        eop.vector_timeless['TEST'] = geo_test
        self.assertTrue(isinstance(eop.vector_timeless['TEST'], GeoDataFrame),
                        'GeoSeries should be parsed into GeoDataFrame')
        self.assertTrue(hasattr(eop.vector_timeless['TEST'], 'geometry'), 'Feature should have geometry attribute')
        self.assertEqual(eop.vector_timeless['TEST'].crs, crs_test, 'GeoDataFrame should still contain the crs')

        with self.assertRaises(ValueError, msg='Should fail because there is no TIMESTAMP column'):
            eop.vector['TEST'] = geo_test

    def test_bbox_feature_type(self):
        eop = EOPatch()
        invalid_entries = [
            0, list(range(4)), tuple(range(5)), {}, set(), [1, 2, 4, 3, 4326, 3], 'BBox'
        ]

        for entry in invalid_entries:
            with self.assertRaises((ValueError, TypeError),
                                   msg='Invalid bbox entry {} should raise an error'.format(entry)):
                eop.bbox = entry

    def test_timestamp_feature_type(self):
        eop = EOPatch()
        invalid_entries = [
            [datetime.datetime(2017, 1, 1, 10, 4, 7), None, datetime.datetime(2017, 1, 11, 10, 3, 51)],
            'something',
            datetime.datetime(2017, 1, 1, 10, 4, 7)
        ]

        valid_entries = [
            ['2018-01-01', '15.2.1992'],
            (datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.date(2017, 1, 11))
        ]

        for entry in invalid_entries:
            with self.assertRaises((ValueError, TypeError),
                                   msg='Invalid timestamp entry {} should raise an error'.format(entry)):
                eop.timestamp = entry

        for entry in valid_entries:
            eop.timestamp = entry


class TestEOPatch(unittest.TestCase):

    def test_add_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.data['bands'] = bands

        self.assertTrue(np.array_equal(eop.data['bands'], bands), msg="Data numpy array not stored")

    def test_simplified_feature_operations(self):
        bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
        feature = FeatureType.DATA, 'TEST-BANDS'
        eop = EOPatch()

        eop[feature] = bands
        self.assertTrue(np.array_equal(eop[feature], bands), msg="Data numpy array not stored")

    def test_rename_feature(self):
        bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.data['bands'] = bands

        eop.rename_feature(FeatureType.DATA, 'bands', 'new_bands')

        self.assertTrue('new_bands' in eop.data)

    def test_rename_feature_missing(self):
        bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.data['bands'] = bands

        with self.assertRaises(BaseException,
                               msg='Should fail because there is no `missing_bands` feature in the EOPatch.'):
            eop.rename_feature(FeatureType.DATA, 'missing_bands', 'new_bands')

    def test_get_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

        eop = EOPatch()
        eop.data['bands'] = bands

        eop_bands = eop.get_feature(FeatureType.DATA, 'bands')

        self.assertTrue(np.array_equal(eop_bands, bands), msg="Data numpy array not returned properly")

    def test_remove_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        names = ['bands1', 'bands2', 'bands3']

        eop = EOPatch()
        eop.add_feature(FeatureType.DATA, names[0], bands)
        eop.data[names[1]] = bands
        eop[FeatureType.DATA][names[2]] = bands

        for feature_name in names:
            self.assertTrue(feature_name in eop.data, "Feature {} was not added to EOPatch".format(feature_name))
            self.assertTrue(np.array_equal(eop.data[feature_name], bands), "Data of feature {} is "
                                                                           "incorrect".format(feature_name))

        eop.remove_feature(FeatureType.DATA, names[0])
        del eop.data[names[1]]
        del eop[FeatureType.DATA][names[2]]
        for feature_name in names:
            self.assertFalse(feature_name in eop.data, msg="Feature {} should be deleted from "
                                                           "EOPatch".format(feature_name))

    def test_concatenate(self):
        eop1 = EOPatch()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop1.data['bands'] = bands1

        eop2 = EOPatch()
        bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
        eop2.data['bands'] = bands2

        eop = eop1 + eop2

        self.assertTrue(np.array_equal(eop.data['bands'], np.concatenate((bands1, bands2), axis=0)),
                        msg="Array mismatch")

    def test_concatenate_different_key(self):
        eop1 = EOPatch()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop1.data['bands'] = bands1

        eop2 = EOPatch()
        bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
        eop2.data['measurements'] = bands2

        eop = eop1 + eop2
        self.assertTrue('bands' in eop.data and 'measurements' in eop.data,
                        'Failed to concatenate different features')

    def test_concatenate_timeless(self):
        eop1 = EOPatch()
        mask1 = np.arange(3*3*2).reshape(3, 3, 2)
        eop1.data_timeless['mask1'] = mask1
        eop1.data_timeless['mask'] = 5 * mask1

        eop2 = EOPatch()
        mask2 = np.arange(3*3*2).reshape(3, 3, 2)
        eop2.data_timeless['mask2'] = mask2
        eop2.data_timeless['mask'] = 5 * mask1  # add mask1 to eop2

        eop = EOPatch.concatenate(eop1, eop2)

        for name in ['mask', 'mask1', 'mask2']:
            self.assertTrue(name in eop.data_timeless)
        self.assertTrue(np.array_equal(eop.data_timeless['mask'], 5 * mask1), "Data with same values should stay "
                                                                              "the same")

    def test_concatenate_missmatched_timeless(self):
        mask = np.arange(3*3*2).reshape(3, 3, 2)

        eop1 = EOPatch()
        eop1.data_timeless['mask'] = mask
        eop1.data_timeless['nask'] = 3 * mask

        eop2 = EOPatch()
        eop2.data_timeless['mask'] = mask
        eop2.data_timeless['nask'] = 5 * mask

        with self.assertRaises(ValueError):
            _ = eop1 + eop2

    def test_equals(self):
        eop1 = EOPatch(data={'bands': np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)})
        eop2 = EOPatch(data={'bands': np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)})
        self.assertEqual(eop1, eop2)

        eop1.data['bands'][1, ...] = np.nan
        self.assertNotEqual(eop1, eop2)

        eop2.data['bands'][1, ...] = np.nan
        self.assertEqual(eop1, eop2)

        eop1.data['bands'] = np.reshape(eop1.data['bands'], (2, 3, 2, 3))
        self.assertNotEqual(eop1, eop2)

        eop2.data['bands'] = np.reshape(eop2.data['bands'], (2, 3, 2, 3))
        eop1.data['bands'] = eop1.data['bands'].astype(np.float16)
        self.assertNotEqual(eop1, eop2)

        del eop1.data['bands']
        del eop2.data['bands']
        self.assertEqual(eop1, eop2)

        eop1.data_timeless['dem'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)

        self.assertNotEqual(eop1, eop2)

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


class TestSavingLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        eopatch = EOPatch()
        mask = np.zeros((3, 3, 2), dtype=np.int16)
        eopatch.data_timeless['mask'] = mask
        eopatch.timestamp = [datetime.datetime(2017, 1, 1, 10, 4, 7),
                             datetime.datetime(2017, 1, 4, 10, 14, 5)]
        eopatch.meta_info['something'] = 'nothing'
        eopatch.scalar['my scalar with spaces'] = np.array([[1, 2, 3]])

        cls.eopatch = eopatch

    def test_saving_at_exiting_file(self):
        with tempfile.NamedTemporaryFile() as fp:
            with self.assertRaises(OSError):
                self.eopatch.save(fp.name)

    def test_saving_in_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name)

    def test_saving_in_non_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            with open(os.path.join(tmp_dir_name, 'foo.txt'), 'w'):
                pass

            with self.assertWarns(UserWarning):
                self.eopatch.save(tmp_dir_name)

            self.eopatch.save(tmp_dir_name, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    def test_overwriting_non_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name)
            self.eopatch.save(tmp_dir_name, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)
            self.eopatch.save(tmp_dir_name, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

            add_eopatch = EOPatch()
            add_eopatch.data['some data'] = np.empty((2, 3, 3, 2))
            add_eopatch.save(tmp_dir_name, overwrite_permission=OverwritePermission.ADD_ONLY)
            with self.assertRaises(ValueError):
                add_eopatch.save(tmp_dir_name, overwrite_permission=OverwritePermission.ADD_ONLY)

            new_eopatch = EOPatch.load(tmp_dir_name, lazy_loading=False, mmap=True)
            self.assertEqual(new_eopatch, self.eopatch + add_eopatch)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name)
            eopatch2 = EOPatch.load(tmp_dir_name, mmap=False)
            self.assertEqual(self.eopatch, eopatch2)

            eopatch2.save(tmp_dir_name, file_format='pkl', overwrite_permission=1)
            eopatch2 = EOPatch.load(tmp_dir_name)
            self.assertEqual(self.eopatch, eopatch2)

            eopatch2.save(tmp_dir_name, file_format='npy', overwrite_permission=1)
            eopatch2 = EOPatch.load(tmp_dir_name, lazy_loading=False, mmap=False)
            self.assertEqual(self.eopatch, eopatch2)

            features = {FeatureType.DATA_TIMELESS: {'mask'}, FeatureType.TIMESTAMP: ...}
            eopatch2.save(tmp_dir_name, file_format=FileFormat.NPY, features=features,
                          compress_level=3, overwrite_permission=1)
            _ = EOPatch.load(tmp_dir_name, lazy_loading=True, mmap=False, features=features)
            eopatch2 = EOPatch.load(tmp_dir_name, lazy_loading=True, mmap=False)
            self.assertEqual(self.eopatch, eopatch2)

    def test_different_formats_equality(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name, file_format=FileFormat.PICKLE, compress_level=4)
            eopatch1 = EOPatch.load(tmp_dir_name, lazy_loading=False)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name, file_format='npy')
            eopatch2 = EOPatch.load(tmp_dir_name, lazy_loading=False, mmap=True)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name, file_format='npy', compress_level=9)
            eopatch3 = EOPatch.load(tmp_dir_name, lazy_loading=False)

        patches = [self.eopatch, eopatch1, eopatch2, eopatch3]

        for eopatch1, eopatch2 in itertools.combinations(patches, 2):
            self.assertEqual(eopatch1, eopatch2)

    def test_feature_names_case_sensitivity(self):
        eopatch = EOPatch()
        mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
        eopatch.data_timeless['mask'] = mask
        eopatch.data_timeless['Mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name, self.assertRaises(OSError):
            eopatch.save(tmp_dir_name, file_format='npy')

    def test_invalid_characters(self):
        eopatch = EOPatch()
        eopatch.data_timeless['mask.npy'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)

        with tempfile.TemporaryDirectory() as tmp_dir_name, self.assertRaises(ValueError):
            eopatch.save(tmp_dir_name, file_format='npy')

    def test_overwrite_failure(self):
        # basic patch
        eopatch = EOPatch()
        mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
        eopatch.data_timeless['mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name, self.assertRaises(BaseException):
            eopatch.save(tmp_dir_name, file_format='npy')

            # load original patch
            eopatch_before = EOPatch.load(tmp_dir_name, mmap=False)

            # force exception during saving (case sensitivity), backup is reloaded
            eopatch.data_timeless['Mask'] = mask
            eopatch.save(tmp_dir_name, file_format='npy', overwrite_permission=2)
            eopatch_after = EOPatch.load(tmp_dir_name)

            # should be equal
            self.assertEqual(eopatch_before, eopatch_after)

    def test_overwrite_success(self):
        # basic patch
        eopatch = EOPatch()
        mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
        eopatch.data_timeless['mask1'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eopatch.save(tmp_dir_name, file_format='npy')

            # load original patch
            eopatch_before = EOPatch.load(tmp_dir_name, mmap=False)

            # update original patch
            eopatch.data_timeless['mask2'] = mask
            eopatch.save(tmp_dir_name, file_format='npy', overwrite_permission=True)
            eopatch_after = EOPatch.load(tmp_dir_name, mmap=False)

            # should be different
            self.assertNotEqual(eopatch_before, eopatch_after)


if __name__ == '__main__':
    unittest.main()
