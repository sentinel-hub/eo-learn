import os
import unittest
import logging
import tempfile

import numpy as np

from sentinelhub import read_data
from sentinelhub.time_utils import datetime_to_iso

from eolearn.core import EOPatch, FeatureType
from eolearn.io import ExportToTiff, ImportFromTiff

logging.basicConfig(level=logging.DEBUG)


class TestExportAndImportTiff(unittest.TestCase):
    """ Testing if export and then import of the data preserves the data
    """

    class TestCase:
        """
        Container for each test case
        """

        def __init__(self, name, feature_type, data, bands=None, times=None, expected_times=None):
            self.name = name
            self.feature_type = feature_type
            self.data = data
            self.bands = bands
            self.times = times
            self.expected_times = expected_times

            if self.expected_times is None:
                self.expected_times = self.times

        def get_expected(self):
            """ Returns expected data at the end of export-import process
            """
            expected = self.data.copy()

            if isinstance(self.expected_times, tuple):
                expected = expected[self.expected_times[0]: self.expected_times[1] + 1, ...]
            elif isinstance(self.expected_times, list):
                expected = expected[self.expected_times, ...]

            if isinstance(self.bands, tuple):
                expected = expected[..., self.bands[0]: self.bands[1] + 1]
            elif isinstance(self.bands, list):
                expected = expected[..., self.bands]

            if expected.dtype == np.int64:
                expected = expected.astype(np.int32)

            return expected

        def get_expected_timestamp_size(self):
            if self.feature_type.is_timeless():
                return None
            return self.get_expected().shape[0]

    @classmethod
    def setUpClass(cls):

        cls.eopatch = EOPatch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '../../../example_data/TestEOPatch'))

        dates = cls.eopatch.timestamp
        scalar_array = np.arange(10 * 6, dtype=np.float).reshape(10, 6)
        mask_array = np.arange(5*3*2*1, dtype=np.uint16).reshape(5, 3, 2, 1)
        data_timeless_array = np.arange(3*2*5, dtype=np.float64).reshape(3, 2, 5)
        data_array = np.arange(10 * 3 * 2 * 6, dtype=np.float32).reshape(10, 3, 2, 6)

        cls.test_cases = [
            cls.TestCase('scalar_timeless', FeatureType.SCALAR_TIMELESS, np.arange(3)),
            cls.TestCase('scalar_timeless_list', FeatureType.SCALAR_TIMELESS, np.arange(5), bands=[3, 0, 2]),
            cls.TestCase('scalar_timeless_tuple', FeatureType.SCALAR_TIMELESS, np.arange(6), bands=(1, 4)),
            cls.TestCase('scalar_band_single_time_single', FeatureType.SCALAR, scalar_array, bands=[3], times=[7]),
            cls.TestCase('scalar_band_list_time_list', FeatureType.SCALAR, scalar_array,
                         bands=[2, 4, 1, 0], times=[1, 7, 0, 2, 3]),
            cls.TestCase('scalar_band_tuple_time_tuple', FeatureType.SCALAR, scalar_array, bands=(1, 4), times=(2, 8)),
            cls.TestCase('mask_timeless', FeatureType.MASK_TIMELESS, np.arange(3*3*1).reshape(3, 3, 1)),
            cls.TestCase('mask_single', FeatureType.MASK, mask_array, times=[4]),
            cls.TestCase('mask_list', FeatureType.MASK, mask_array, times=[4, 2]),
            cls.TestCase('mask_tuple_int', FeatureType.MASK, mask_array, times=(2, 4)),
            cls.TestCase('mask_tuple_datetime', FeatureType.MASK, mask_array, times=(dates[2], dates[4]),
                         expected_times=(2, 4)),
            cls.TestCase('mask_tuple_string', FeatureType.MASK, mask_array,
                         times=(datetime_to_iso(dates[2], only_date=False), datetime_to_iso(dates[4], only_date=False)),
                         expected_times=(2, 4)),
            cls.TestCase('data_timeless_band_list', FeatureType.DATA_TIMELESS, data_timeless_array, bands=[2, 4, 1, 0]),
            cls.TestCase('data_timeless_band_tuple', FeatureType.DATA_TIMELESS, data_timeless_array, bands=(1, 4)),
            cls.TestCase('data_band_list_time_list', FeatureType.DATA, data_array,
                         bands=[2, 4, 1, 0], times=[1, 7, 0, 2, 3]),
            cls.TestCase('data_band_tuple_time_tuple', FeatureType.DATA, data_array, bands=(1, 4), times=(2, 8)),
            cls.TestCase('', FeatureType.DATA, data_array),
        ]

    def test_export_import(self):
        for test_case in self.test_cases:
            with self.subTest(msg='Test case {}'.format(test_case.name)):

                self.eopatch[test_case.feature_type][test_case.name] = test_case.data

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tmp_file_name = 'temp_file.tiff'
                    feature = test_case.feature_type, test_case.name

                    export_task = ExportToTiff(feature, folder=tmp_dir_name,
                                               band_indices=test_case.bands, date_indices=test_case.times)
                    export_task.execute(self.eopatch, filename=tmp_file_name)

                    import_task = ImportFromTiff(feature, folder=tmp_dir_name,
                                                 timestamp_size=test_case.get_expected_timestamp_size())

                    expected_raster = test_case.get_expected()

                    new_eop = import_task.execute(filename=tmp_file_name)
                    old_eop = import_task.execute(self.eopatch, filename=tmp_file_name)

                    self.assertTrue(np.array_equal(expected_raster, new_eop[test_case.feature_type][test_case.name]),
                                    msg='Tiff imported into new EOPatch is not the same as expected')
                    self.assertTrue(np.array_equal(expected_raster, old_eop[test_case.feature_type][test_case.name]),
                                    msg='Tiff imported into old EOPatch is not the same as expected')
                    self.assertEqual(expected_raster.dtype, new_eop[test_case.feature_type][test_case.name].dtype,
                                     msg='Tiff imported into new EOPatch has different dtype as expected')

    def test_export2tiff_wrong_format(self):
        data = np.arange(10*3*2*6, dtype=float).reshape(10, 3, 2, 6)

        self.eopatch.data['data'] = data

        for bands, times in [([2, 'string', 1, 0], [1, 7, 0, 2, 3]),
                             ([2, 3, 1, 0], [1, 7, 'string', 2, 3])]:
            with tempfile.TemporaryDirectory() as tmp_dir_name, self.assertRaises(ValueError):
                tmp_file_name = 'temp_file.tiff'
                task = ExportToTiff((FeatureType.DATA, 'data'), folder=tmp_dir_name,
                                    band_indices=bands, date_indices=times, image_dtype=data.dtype)
                task.execute(self.eopatch, filename=tmp_file_name)


class TestImportTiff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eopatch = EOPatch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '../../../example_data/TestEOPatch'))

    def test_import_tiff_subset(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/import-tiff-test1.tiff')

        mask_feature = FeatureType.MASK_TIMELESS, 'TEST_TIF'
        mask_type, mask_name = mask_feature

        task = ImportFromTiff(mask_feature, path)
        task.execute(self.eopatch)

        tiff_img = read_data(path)

        self.assertTrue(np.array_equal(tiff_img[20: 53, 21: 54], self.eopatch[mask_type][mask_name][..., 0]),
                        msg='Imported tiff data should be the same as original')

    def test_import_tiff_intersecting(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/import-tiff-test2.tiff')

        mask_feature = FeatureType.MASK_TIMELESS, 'TEST_TIF'
        mask_type, mask_name = mask_feature
        no_data_value = 1.0

        task = ImportFromTiff(mask_feature, path, image_dtype=np.float64, no_data_value=no_data_value)
        task.execute(self.eopatch)

        tiff_img = read_data(path)

        self.assertTrue(np.array_equal(tiff_img[-6:, :3, :], self.eopatch[mask_type][mask_name][:6, -3:, :]),
                        msg='Imported tiff data should be the same as original')
        feature_dtype = self.eopatch[mask_type][mask_name].dtype
        self.assertEqual(feature_dtype, np.float64,
                         msg='Feature should have dtype numpy.float64 but {} found'.format(feature_dtype))

        self.eopatch[mask_type][mask_name][:6, -3:, :] = no_data_value
        unique_values = list(np.unique(self.eopatch[mask_type][mask_name][:6, -3:, :]))
        self.assertEqual(unique_values, [no_data_value],
                         msg='No data values should all be equal to {}'.format(no_data_value))


if __name__ == '__main__':
    unittest.main()
