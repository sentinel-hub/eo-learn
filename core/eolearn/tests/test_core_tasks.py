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
import datetime
import os
import copy
import numpy as np

from eolearn.core import (
    EOPatch, FeatureType, CRS, CopyTask, DeepCopyTask, AddFeatureTask, RemoveFeatureTask, RenameFeatureTask,
    DuplicateFeatureTask, InitializeFeatureTask, MoveFeatureTask, MergeFeatureTask, MapFeatureTask, ZipFeatureTask,
    ExtractBandsTask, CreateEOPatchTask, MergeEOPatchesTask
)


logging.basicConfig(level=logging.DEBUG)


class TestCoreTasks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '../../../example_data', 'TestEOPatch')
        cls.patch = EOPatch()

        cls.patch.data['bands'] = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        cls.patch.mask_timeless['mask'] = np.arange(3*3*2).reshape(3, 3, 2)
        cls.patch.scalar['values'] = np.arange(10*5).reshape(10, 5)
        cls.patch.timestamp = [datetime.datetime(2017, 1, 1, 10, 4, 7),
                               datetime.datetime(2017, 1, 4, 10, 14, 5),
                               datetime.datetime(2017, 1, 11, 10, 3, 51),
                               datetime.datetime(2017, 1, 14, 10, 13, 46),
                               datetime.datetime(2017, 1, 24, 10, 14, 7),
                               datetime.datetime(2017, 2, 10, 10, 1, 32),
                               datetime.datetime(2017, 2, 20, 10, 6, 35),
                               datetime.datetime(2017, 3, 2, 10, 0, 20),
                               datetime.datetime(2017, 3, 12, 10, 7, 6),
                               datetime.datetime(2017, 3, 15, 10, 12, 14)]
        cls.patch.bbox = (324.54, 546.45, 955.4, 63.43, 3857)
        cls.patch.meta_info['something'] = np.random.rand(10, 1)

    def test_copy(self):
        patch_copy = CopyTask().execute(self.patch)

        self.assertEqual(self.patch, patch_copy, 'Copied patch is different')

        patch_copy.data['new'] = np.arange(1).reshape(1, 1, 1, 1)
        self.assertFalse('new' in self.patch.data, 'Dictionary of features was not copied')

        patch_copy.data['bands'][0, 0, 0, 0] += 1
        self.assertTrue(np.array_equal(self.patch.data['bands'], patch_copy.data['bands']),
                        'Data should not be copied')

    def test_deepcopy(self):
        patch_deepcopy = DeepCopyTask().execute(self.patch)

        self.assertEqual(self.patch, patch_deepcopy, 'Deep copied patch is different')

        patch_deepcopy.data['new'] = np.arange(1).reshape(1, 1, 1, 1)
        self.assertFalse('new' in self.patch.data, 'Dictionary of features was not copied')

        patch_deepcopy.data['bands'][0, 0, 0, 0] += 1
        self.assertFalse(np.array_equal(self.patch.data['bands'], patch_deepcopy.data['bands']),
                         'Data should be copied')

    def test_partial_copy(self):
        partial_copy = DeepCopyTask(features=[(FeatureType.MASK_TIMELESS, 'mask'),
                                              FeatureType.BBOX]).execute(self.patch)
        expected_patch = EOPatch(mask_timeless=self.patch.mask_timeless, bbox=self.patch.bbox)
        self.assertEqual(partial_copy, expected_patch, 'Partial copying was not successful')

        partial_deepcopy = DeepCopyTask(features=[FeatureType.TIMESTAMP,
                                                  (FeatureType.SCALAR, 'values')]).execute(self.patch)
        expected_patch = EOPatch(scalar=self.patch.scalar, timestamp=self.patch.timestamp)
        self.assertEqual(partial_deepcopy, expected_patch, 'Partial deep copying was not successful')

    def test_add_rename_remove_feature(self):
        cloud_mask = np.arange(10).reshape(5, 2, 1, 1)
        feature_name = 'CLOUD MASK'
        new_feature_name = 'CLM'

        patch = copy.deepcopy(self.patch)

        patch = AddFeatureTask((FeatureType.MASK, feature_name))(patch, cloud_mask)
        self.assertTrue(np.array_equal(patch.mask[feature_name], cloud_mask), 'Feature was not added')

        patch = RenameFeatureTask((FeatureType.MASK, feature_name, new_feature_name))(patch)
        self.assertTrue(np.array_equal(patch.mask[new_feature_name], cloud_mask), 'Feature was not renamed')
        self.assertFalse(feature_name in patch[FeatureType.MASK], 'Old feature still exists')

        patch = RemoveFeatureTask((FeatureType.MASK, new_feature_name))(patch)
        self.assertFalse(feature_name in patch.mask, 'Feature was not removed')

        patch = RemoveFeatureTask(FeatureType.MASK_TIMELESS)(patch)
        self.assertEqual(len(patch.mask_timeless), 0, 'mask_timeless features were not removed')

        patch = RemoveFeatureTask((FeatureType.MASK, ...))(patch)
        self.assertEqual(len(patch.mask), 0, 'mask features were not removed')

    def test_duplicate_feature(self):
        mask_data = np.arange(10).reshape(5, 2, 1, 1)
        feature_name = 'MASK1'
        duplicate_name = 'MASK2'

        patch = AddFeatureTask((FeatureType.MASK, feature_name))(self.patch, mask_data)

        duplicate_task = DuplicateFeatureTask((FeatureType.MASK, feature_name, duplicate_name))
        patch = duplicate_task(patch)

        self.assertTrue(duplicate_name in patch.mask, 'Feature was not duplicated. Name not found.')
        self.assertEqual(id(patch.mask['MASK1']), id(patch.mask['MASK2']))
        self.assertTrue(np.array_equal(patch.mask[duplicate_name], mask_data),
                        'Feature was not duplicated correctly. Data does not match.')

        with self.assertRaises(ValueError, msg='Expected a ValueError when creating an already exising feature.'):
            patch = duplicate_task(patch)

        duplicate_names = {'D1', 'D2'}
        feature_list = [(FeatureType.MASK, 'MASK1', 'D1'), (FeatureType.MASK, 'MASK2', 'D2')]
        patch = DuplicateFeatureTask(feature_list).execute(patch)

        self.assertTrue(duplicate_names.issubset(patch.mask), 'Duplicating multiple features failed.')

        patch = DuplicateFeatureTask((FeatureType.MASK, 'MASK1', 'DEEP'), deep_copy=True)(patch)
        self.assertNotEqual(id(patch.mask['MASK1']), id(patch.mask['DEEP']))
        self.assertTrue(np.array_equal(patch.mask['MASK1'], patch.mask['DEEP']),
                        'Feature was not duplicated correctly. Data does not match.')

        # Duplicating MASK1 three times into D3, D4, D5 doesn't work, because EOTask.feature_gen
        # returns a dict containing only ('MASK1', 'D5') duplication

        # duplicate_names = {'D3', 'D4', 'D5'}
        # feature_list = [(FeatureType.MASK, 'MASK1', new) for new in duplicate_names]
        # patch = DuplicateFeatureTask(feature_list).execute(patch)

        # self.assertTrue(duplicate_names.issubset(patch.mask),
        #                 'Duplicating single feature multiple times failed.')

    def test_initialize_feature(self):
        patch = DeepCopyTask()(self.patch)

        init_val = 123
        shape = (5, 10, 10, 3)
        compare_data = np.ones(shape) * init_val

        patch = InitializeFeatureTask((FeatureType.MASK, 'test'), shape=shape, init_value=init_val)(patch)
        self.assertEqual(patch.mask['test'].shape, shape)
        self.assertTrue(np.array_equal(patch.mask['test'], compare_data))

        failmsg = 'Expected a ValueError when trying to initialize a feature with a wrong shape dmensions.'
        with self.assertRaises(ValueError, msg=failmsg):
            patch = InitializeFeatureTask((FeatureType.MASK_TIMELESS, 'wrong'), shape=shape, init_value=init_val)(patch)

        init_val = 123
        shape = (10, 10, 3)
        compare_data = np.ones(shape) * init_val

        patch = InitializeFeatureTask((FeatureType.MASK_TIMELESS, 'test'), shape=shape, init_value=init_val)(patch)
        self.assertEqual(patch.mask_timeless['test'].shape, shape)
        self.assertTrue(np.array_equal(patch.mask_timeless['test'], compare_data))

        fail_msg = 'Expected a ValueError when trying to initialize a feature with a wrong shape dmensions.'
        with self.assertRaises(ValueError, msg=fail_msg):
            patch = InitializeFeatureTask((FeatureType.MASK, 'wrong'), shape=shape, init_value=init_val)(patch)

        init_val = 123
        shape = (5, 10, 10, 3)
        compare_data = np.ones(shape) * init_val
        new_names = {'F1', 'F2', 'F3'}

        patch = InitializeFeatureTask({FeatureType.MASK: new_names}, shape=shape, init_value=init_val)(patch)
        fail_msg = "Failed to initialize new features from a shape tuple."
        self.assertTrue(new_names < set(patch.mask), msg=fail_msg)
        self.assertTrue(all(patch.mask[key].shape == shape for key in new_names))
        self.assertTrue(all(np.array_equal(patch.mask[key], compare_data) for key in new_names))

        patch = InitializeFeatureTask({FeatureType.DATA: new_names}, shape=(FeatureType.DATA, 'bands'))(patch)
        fail_msg = "Failed to initialize new features from an existing feature."
        self.assertTrue(new_names < set(patch.data), msg=fail_msg)
        self.assertTrue(all(patch.data[key].shape == patch.data['bands'].shape for key in new_names))

        self.assertRaises(ValueError, InitializeFeatureTask, {FeatureType.DATA: new_names}, 1234)

    def test_move_feature(self):
        patch_src = EOPatch()
        patch_dst = EOPatch()

        shape = (10, 5, 5, 3)
        size = np.product(shape)

        shape_timeless = (5, 5, 3)
        size_timeless = np.product(shape_timeless)

        data = [np.random.randint(0, 100, size).reshape(*shape) for i in range(3)] + \
               [np.random.randint(0, 100, size_timeless).reshape(*shape_timeless) for i in range(2)]

        features = [(FeatureType.DATA, 'D1'),
                    (FeatureType.DATA, 'D2'),
                    (FeatureType.MASK, 'M1'),
                    (FeatureType.MASK_TIMELESS, 'MTless1'),
                    (FeatureType.MASK_TIMELESS, 'MTless2')]

        for feat, dat in zip(features, data):
            patch_src = AddFeatureTask(feat)(patch_src, dat)

        patch_dst = MoveFeatureTask(features)(patch_src, patch_dst)

        for i, feature in enumerate(features):
            self.assertTrue(id(data[i]) == id(patch_dst[feature]))
            self.assertTrue(np.array_equal(data[i], patch_dst[feature]))

        patch_dst = EOPatch()
        patch_dst = MoveFeatureTask(features, deep_copy=True)(patch_src, patch_dst)

        for i, feature in enumerate(features):
            self.assertTrue(id(data[i]) != id(patch_dst[feature]))
            self.assertTrue(np.array_equal(data[i], patch_dst[feature]))

        features = [(FeatureType.MASK_TIMELESS, ...)]
        patch_dst = EOPatch()
        patch_dst = MoveFeatureTask(features)(patch_src, patch_dst)

        self.assertTrue(FeatureType.MASK_TIMELESS in patch_dst.get_features())
        self.assertFalse(FeatureType.DATA in patch_dst.get_features())

        self.assertTrue('MTless1' in patch_dst.get_feature(FeatureType.MASK_TIMELESS))
        self.assertTrue('MTless2' in patch_dst.get_feature(FeatureType.MASK_TIMELESS))

    def test_merge_features(self):
        patch = EOPatch()

        shape = (10, 5, 5, 3)
        size = np.product(shape)

        shape_timeless = (5, 5, 3)
        size_timeless = np.product(shape_timeless)

        data = [np.random.randint(0, 100, size).reshape(*shape) for _ in range(3)] + \
               [np.random.randint(0, 100, size_timeless).reshape(*shape_timeless) for _ in range(2)]

        features = [(FeatureType.DATA, 'D1'),
                    (FeatureType.DATA, 'D2'),
                    (FeatureType.MASK, 'M1'),
                    (FeatureType.MASK_TIMELESS, 'MTless1'),
                    (FeatureType.MASK_TIMELESS, 'MTless2')]

        for feat, dat in zip(features, data):
            patch = AddFeatureTask(feat)(patch, dat)

        patch = MergeFeatureTask(features[:3], (FeatureType.MASK, 'merged'))(patch)
        patch = MergeFeatureTask(features[3:], (FeatureType.MASK_TIMELESS, 'merged_timeless'))(patch)

        expected = np.concatenate([patch[f] for f in features[:3]], axis=-1)

        self.assertTrue(np.array_equal(patch.mask['merged'], expected))

    def test_zip_features(self):
        patch = EOPatch.load(self.data_path)

        merge = ZipFeatureTask({FeatureType.DATA: ['CLP', 'NDVI', 'BANDS-S2-L1C']}, # input features
                               (FeatureType.DATA, 'MERGED'),                        # output feature
                               lambda *f: np.concatenate(f, axis=-1))

        patch = merge(patch)

        expected = np.concatenate([patch.data['CLP'], patch.data['NDVI'], patch.data['BANDS-S2-L1C']], axis=-1)
        self.assertTrue(np.array_equal(patch.data['MERGED'], expected))

        zip_fail = ZipFeatureTask({FeatureType.DATA: ['CLP', 'NDVI']}, (FeatureType.DATA, 'MERGED'))
        self.assertRaises(NotImplementedError, zip_fail, patch)

    def test_map_features(self):
        patch = EOPatch.load(self.data_path)

        move = MapFeatureTask({FeatureType.DATA: ['CLP', 'NDVI', 'BANDS-S2-L1C']},
                              {FeatureType.DATA: ['CLP2', 'NDVI2', 'BANDS-S2-L1C2']}, copy.deepcopy)

        patch = move(patch)

        self.assertTrue(np.array_equal(patch.data['CLP'], patch.data['CLP2']))
        self.assertTrue(np.array_equal(patch.data['NDVI'], patch.data['NDVI2']))
        self.assertTrue(np.array_equal(patch.data['BANDS-S2-L1C'], patch.data['BANDS-S2-L1C2']))

        self.assertTrue(id(patch.data['CLP']) != id(patch.data['CLP2']))
        self.assertTrue(id(patch.data['NDVI']) != id(patch.data['NDVI2']))
        self.assertTrue(id(patch.data['BANDS-S2-L1C']) != id(patch.data['BANDS-S2-L1C2']))

        map_fail = MapFeatureTask({FeatureType.DATA: ['CLP', 'NDVI']}, {FeatureType.DATA: ['CLP2', 'NDVI2', ]})
        self.assertRaises(NotImplementedError, map_fail, patch)

        f_in, f_out = {FeatureType.DATA: ['CLP', 'NDVI']}, {FeatureType.DATA: ['CLP2']}
        self.assertRaises(ValueError, MapFeatureTask, f_in, f_out)

    def test_extract_bands(self):
        patch = EOPatch.load(self.data_path)

        bands = [2, 4, 8]
        move_bands = ExtractBandsTask((FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA, 'MOVED_BANDS'), bands)
        patch = move_bands(patch)
        self.assertTrue(np.array_equal(patch.data['MOVED_BANDS'], patch.data['REFERENCE_SCENES'][..., bands]))

        bands = [2, 4, 16]
        move_bands = ExtractBandsTask((FeatureType.DATA, 'REFERENCE_SCENES'), (FeatureType.DATA, 'MOVED_BANDS'), bands)
        self.assertRaises(ValueError, move_bands, patch)

    def test_create_eopatch(self):
        data = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
        bbox = [5.60, 52.68, 5.75, 52.63, CRS.WGS84]

        patch = CreateEOPatchTask()(data={'bands': data}, bbox=bbox)
        self.assertTrue(np.array_equal(patch.data['bands'], data))

    def test_kwargs(self):
        patch = EOPatch()
        shape = (3, 5, 5, 2)

        data1 = np.random.randint(0, 5, size=shape)
        data2 = np.random.randint(0, 5, size=shape)

        patch[(FeatureType.DATA, 'D1')] = data1
        patch[(FeatureType.DATA, 'D2')] = data2

        task0 = MapFeatureTask((FeatureType.DATA, 'D1'),
                               (FeatureType.DATA_TIMELESS, 'NON_ZERO'),
                               np.count_nonzero,
                               axis=0)

        task1 = MapFeatureTask((FeatureType.DATA, 'D1'),
                               (FeatureType.DATA_TIMELESS, 'MAX1'),
                               np.max, axis=0)

        task2 = ZipFeatureTask({FeatureType.DATA: ['D1', 'D2']},
                               (FeatureType.DATA, 'MAX2'),
                               np.maximum,
                               dtype=np.float32)

        patch = task0(patch)
        patch = task1(patch)
        patch = task2(patch)

        self.assertTrue(np.array_equal(patch[(FeatureType.DATA_TIMELESS, 'NON_ZERO')], np.count_nonzero(data1, axis=0)))
        self.assertTrue(np.array_equal(patch[(FeatureType.DATA_TIMELESS, 'MAX1')], np.max(data1, axis=0)))
        self.assertTrue(np.array_equal(patch[(FeatureType.DATA, 'MAX2')], np.maximum(data1, data2)))

    def test_merge_eopatches(self):
        task = MergeEOPatchesTask(time_dependent_op='mean', timeless_op='concatenate')

        patch = EOPatch.load(self.data_path)
        del patch.data['REFERENCE_SCENES']  # wrong time dimension

        task.execute(patch, patch, patch)


if __name__ == '__main__':
    unittest.main()
