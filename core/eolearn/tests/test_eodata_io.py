"""
Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import unittest
import logging
import datetime
import numpy as np
import tempfile

import fs
from fs.errors import CreateFailed
from fs.tempfs import TempFS
from geopandas import GeoDataFrame

from sentinelhub import BBox, CRS
from eolearn.core import EOPatch, FeatureType, OverwritePermission

logging.basicConfig(level=logging.DEBUG)


class TestSavingLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        eopatch = EOPatch()
        mask = np.zeros((3, 3, 2), dtype=np.int16)
        eopatch.data_timeless['mask'] = mask
        eopatch.timestamp = [datetime.datetime(2017, 1, 1, 10, 4, 7),
                             datetime.datetime(2017, 1, 4, 10, 14, 5)]
        eopatch.meta_info['something'] = 'nothing'
        eopatch.bbox = BBox((1, 2, 3, 4), CRS.WGS84)
        eopatch.scalar['my scalar with spaces'] = np.array([[1, 2, 3]])
        eopatch.vector['my-df'] = GeoDataFrame({
            'values': [1],
            'TIMESTAMP': [datetime.datetime(2017, 1, 1, 10, 4, 7)],
            'geometry': [eopatch.bbox.geometry]
        }, crs={'init': eopatch.bbox.crs.epsg})

        cls.eopatch = eopatch

    def test_saving_to_a_file(self):
        with tempfile.NamedTemporaryFile() as fp:
            with self.assertRaises(CreateFailed):
                self.eopatch.save(fp.name)

    def test_saving_in_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name)

            self.eopatch.save(fs.path.combine(tmp_dir_name, 'new-subfolder'))

    def test_saving_in_non_empty_folder(self):
        with TempFS() as temp_fs:
            empty_file = 'foo.txt'

            with temp_fs.open(empty_file, 'w'):
                pass

            self.eopatch.save(temp_fs.root_path)
            self.assertTrue(temp_fs.exists(empty_file))

            self.eopatch.save('/', overwrite_permission=OverwritePermission.OVERWRITE_PATCH, filesystem=temp_fs)
            self.assertFalse(temp_fs.exists(empty_file))

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

            new_eopatch = EOPatch.load(tmp_dir_name, lazy_loading=False)
            self.assertEqual(new_eopatch, self.eopatch + add_eopatch)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.eopatch.save(tmp_dir_name)
            eopatch2 = EOPatch.load(tmp_dir_name)
            self.assertEqual(self.eopatch, eopatch2)

            eopatch2.save(tmp_dir_name, overwrite_permission=1)
            eopatch2 = EOPatch.load(tmp_dir_name)
            self.assertEqual(self.eopatch, eopatch2)

            eopatch2.save(tmp_dir_name, overwrite_permission=1)
            eopatch2 = EOPatch.load(tmp_dir_name, lazy_loading=False)
            self.assertEqual(self.eopatch, eopatch2)

            features = {FeatureType.DATA_TIMELESS: {'mask'}, FeatureType.TIMESTAMP: ...}
            eopatch2.save(tmp_dir_name, features=features,
                          compress_level=3, overwrite_permission=1)
            _ = EOPatch.load(tmp_dir_name, lazy_loading=True, features=features)
            eopatch2 = EOPatch.load(tmp_dir_name, lazy_loading=True, )
            self.assertEqual(self.eopatch, eopatch2)

    def test_feature_names_case_sensitivity(self):
        eopatch = EOPatch()
        mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
        eopatch.data_timeless['mask'] = mask
        eopatch.data_timeless['Mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name, self.assertRaises(IOError):
            eopatch.save(tmp_dir_name)

    def test_invalid_characters(self):
        eopatch = EOPatch()

        with self.assertRaises(ValueError):
            eopatch.data_timeless['mask.npy'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)

    def test_overwrite_failure(self):
        eopatch = EOPatch()
        mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
        eopatch.data_timeless['mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name, self.assertRaises(IOError):
            eopatch.save(tmp_dir_name)

            # load original patch
            eopatch_before = EOPatch.load(tmp_dir_name)

            # force exception during saving (case sensitivity), backup is reloaded
            eopatch.data_timeless['Mask'] = mask
            eopatch.save(tmp_dir_name, overwrite_permission=2)
            eopatch_after = EOPatch.load(tmp_dir_name)

            # should be equal
            self.assertEqual(eopatch_before, eopatch_after)

    def test_overwrite_success(self):
        # basic patch
        eopatch = EOPatch()
        mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
        eopatch.data_timeless['mask1'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eopatch.save(tmp_dir_name)

            # load original patch
            eopatch_before = EOPatch.load(tmp_dir_name)

            # update original patch
            eopatch.data_timeless['mask2'] = mask
            eopatch.save(tmp_dir_name, overwrite_permission=1)
            eopatch_after = EOPatch.load(tmp_dir_name)

            # should be different
            self.assertNotEqual(eopatch_before, eopatch_after)


if __name__ == '__main__':
    unittest.main()
