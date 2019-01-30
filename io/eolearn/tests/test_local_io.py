import os
import unittest
import logging
import tempfile
import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.io import ExportToTiff
from sentinelhub.io_utils import read_data
from sentinelhub.time_utils import datetime_to_iso, iso_to_datetime

logging.basicConfig(level=logging.DEBUG)


class TestEOPatch(unittest.TestCase):

    PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/TestEOPatch')

    def test_export2tiff_mask_timeless(self):
        mask_timeless = np.arange(3*3*1).reshape(3, 3, 1)
        subset = mask_timeless.squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.mask_timeless['mask_timeless'] = mask_timeless

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.MASK_TIMELESS, 'mask_timeless'), folder=tmp_dir_name)
            task.execute(eop, filename=tmp_file_name)

            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_mask_single(self):
        mask = np.arange(5*3*3*1).reshape(5, 3, 3, 1)
        times = [4]
        subset = mask[times].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.mask['mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.MASK, 'mask'), folder=tmp_dir_name, date_indices=times)
            task.execute(eop, filename=tmp_file_name)

            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_mask_list(self):
        mask = np.arange(5*3*2*1).reshape(5, 3, 2, 1)
        times = [4, 2]
        subset = mask[times].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.mask['mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.MASK, 'mask'), folder=tmp_dir_name, date_indices=times)
            task.execute(eop, filename=tmp_file_name)

            # rasterio saves `bands` to the last dimension, move it up front
            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            raster = np.moveaxis(raster, -1, 0)

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_mask_tuple_int(self):
        mask = np.arange(5*3*2*1).reshape(5, 3, 2, 1)
        times = (2, 4)
        selection = np.arange(times[0], times[1]+1)
        subset = mask[selection].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.mask['mask'] = mask

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.MASK, 'mask'), folder=tmp_dir_name, date_indices=times)
            task.execute(eop, filename=tmp_file_name)

            # rasterio saves `bands` to the last dimension, move it up front
            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            raster = np.moveaxis(raster, -1, 0)

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_mask_tuple_datetime(self):
        mask = np.arange(5*3*2*1).reshape(5, 3, 2, 1)

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.mask['mask'] = mask
        dates = np.array(eop.timestamp)

        indices = [2, 4]
        times = (dates[indices[0]], dates[indices[1]])
        selection = np.arange(indices[0], indices[1] + 1)
        subset = mask[selection].squeeze()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.MASK, 'mask'), folder=tmp_dir_name, date_indices=times)
            task.execute(eop, filename=tmp_file_name)

            # rasterio saves `bands` to the last dimension, move it up front
            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            raster = np.moveaxis(raster, -1, 0)

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_mask_tuple_string(self):
        mask = np.arange(5*3*2*1).reshape(5, 3, 2, 1)

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.mask['mask'] = mask
        dates = np.array(eop.timestamp)

        indices = [2, 4]

        # day time gets floored
        times = (datetime_to_iso(dates[indices[0]]),
                 datetime_to_iso(dates[indices[1]]))

        selection = np.nonzero(np.where((dates >= iso_to_datetime(times[0])) &
                                        (dates <= iso_to_datetime(times[1])), dates, 0))

        subset = mask[selection].squeeze()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.MASK, 'mask'), folder=tmp_dir_name, date_indices=times)
            task.execute(eop, filename=tmp_file_name)

            # rasterio saves `bands` to the last dimension, move it up front
            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            raster = np.moveaxis(raster, -1, 0)

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_data_timeless_band_single(self):
        data_timeless = np.arange(3*2*5, dtype=float).reshape(3, 2, 5)
        bands = [2]
        subset = data_timeless[..., bands].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.data_timeless['data_timeless'] = data_timeless

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.DATA_TIMELESS, 'data_timeless'), folder=tmp_dir_name,
                                band_indices=bands, image_dtype=data_timeless.dtype)
            task.execute(eop, filename=tmp_file_name)

            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_data_timeless_band_list(self):
        data_timeless = np.arange(3*2*5, dtype=float).reshape(3, 2, 5)
        bands = [2, 4, 1, 0]
        subset = data_timeless[..., bands].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.data_timeless['data_timeless'] = data_timeless

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.DATA_TIMELESS, 'data_timeless'), folder=tmp_dir_name,
                                band_indices=bands, image_dtype=data_timeless.dtype)
            task.execute(eop, filename=tmp_file_name)

            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_data_timeless_band_tuple(self):
        data_timeless = np.arange(3*2*5, dtype=float).reshape(3, 2, 5)
        bands = (1, 3)
        selection = np.arange(bands[0], bands[1] + 1)
        subset = data_timeless[..., selection].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.data_timeless['data_timeless'] = data_timeless

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.DATA_TIMELESS, 'data_timeless'), folder=tmp_dir_name,
                                band_indices=bands, image_dtype=data_timeless.dtype)
            task.execute(eop, filename=tmp_file_name)

            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_data_band_single_time_single(self):
        data = np.arange(10*3*2*6, dtype=float).reshape(10, 3, 2, 6)
        bands = [3]
        times = [7]

        subset = data[times, ..., bands].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.data['data'] = data

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.DATA, 'data'), folder=tmp_dir_name,
                                band_indices=bands, date_indices=times, image_dtype=data.dtype)
            task.execute(eop, filename=tmp_file_name)

            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_data_band_list_time_list(self):
        data = np.arange(10*3*2*6, dtype=float).reshape(10, 3, 2, 6)
        bands = [2, 4, 1, 0]
        times = [1, 7, 0, 2, 3]

        subset = data[times][..., bands].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.data['data'] = data

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.DATA, 'data'), folder=tmp_dir_name,
                                band_indices=bands, date_indices=times, image_dtype=data.dtype)
            task.execute(eop, filename=tmp_file_name)

            # split times and bands in raster and mimic the initial shape
            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            raster = raster.reshape(raster.shape[0], raster.shape[1], len(bands), len(times))
            raster = np.moveaxis(raster, -1, 0)

            self.assertTrue(np.all(subset == raster))

    def test_export2tiff_data_band_tuple_time_tuple(self):
        data = np.arange(10*3*2*6, dtype=float).reshape(10, 3, 2, 6)
        bands = (1, 4)
        times = (2, 8)
        bands_selection = np.arange(bands[0], bands[1] + 1)
        times_selection = np.arange(times[0], times[1] + 1)

        subset = data[times_selection][..., bands_selection].squeeze()

        eop = EOPatch.load(self.PATCH_FILENAME)
        eop.data['data'] = data

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_name = 'temp_file.tiff'
            task = ExportToTiff((FeatureType.DATA, 'data'), folder=tmp_dir_name,
                                band_indices=bands, date_indices=times, image_dtype=data.dtype)
            task.execute(eop, filename=tmp_file_name)

            # split times and bands in raster and mimic the initial shape
            raster = read_data(os.path.join(tmp_dir_name, tmp_file_name))
            raster = raster.reshape(raster.shape[0], raster.shape[1], len(bands_selection), len(times_selection))
            raster = np.moveaxis(raster, -1, 0)

            self.assertTrue(np.all(subset == raster))


if __name__ == '__main__':
    unittest.main()
