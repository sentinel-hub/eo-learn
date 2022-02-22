"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2018-2019 William Ouellette (TomTom)
Copyright (c) 2019 Drew Bollinger (DevelopmentSeed)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy
import dataclasses
import datetime
import logging
import os
import tempfile
from typing import Optional, Union

import boto3
import numpy as np
import pytest
from conftest import TEST_EOPATCH_PATH
from fs.errors import ResourceNotFound
from moto import mock_s3
from numpy.testing import assert_array_equal

from sentinelhub import read_data
from sentinelhub.time_utils import serialize_time

from eolearn.core import EOPatch, FeatureType
from eolearn.io import ExportToTiffTask, ImportFromTiffTask

logging.basicConfig(level=logging.DEBUG)


BUCKET_NAME = "mocked-test-bucket"
PATH_ON_BUCKET = f"s3://{BUCKET_NAME}/some-folder"


@pytest.fixture(autouse=True)
def create_s3_bucket_fixture():
    with mock_s3():
        s3resource = boto3.resource("s3", region_name="eu-central-1")
        s3resource.create_bucket(Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        yield


@dataclasses.dataclass
class TiffTestCase:
    name: str
    feature_type: FeatureType
    data: np.ndarray
    bands: Optional[Union[tuple, list]] = None
    times: Optional[Union[tuple, list]] = None
    expected_times: Optional[Union[tuple, list]] = None

    def __post_init__(self):
        if self.expected_times is None:
            self.expected_times = self.times

    def get_expected(self):
        """Returns expected data at the end of export-import process"""
        expected = self.data.copy()

        if isinstance(self.expected_times, tuple):
            expected = expected[self.expected_times[0] : self.expected_times[1] + 1, ...]
        elif isinstance(self.expected_times, list):
            expected = expected[self.expected_times, ...]

        if isinstance(self.bands, tuple):
            expected = expected[..., self.bands[0] : self.bands[1] + 1]
        elif isinstance(self.bands, list):
            expected = expected[..., self.bands]

        if expected.dtype == np.int64:
            expected = expected.astype(np.int32)

        return expected

    def get_expected_timestamp_size(self):
        if self.feature_type.is_timeless():
            return None
        return self.get_expected().shape[0]


DATES = EOPatch.load(TEST_EOPATCH_PATH, lazy_loading=True).timestamp
SCALAR_ARRAY = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)
MASK_ARRAY = np.arange(5 * 3 * 2 * 1, dtype=np.uint16).reshape(5, 3, 2, 1)
DATA_TIMELESS_ARRAY = np.arange(3 * 2 * 5, dtype=np.float64).reshape(3, 2, 5)
DATA_ARRAY = np.arange(10 * 3 * 2 * 6, dtype=np.float32).reshape(10, 3, 2, 6)


TIFF_TEST_CASES = [
    TiffTestCase("scalar_timeless", FeatureType.SCALAR_TIMELESS, np.arange(3)),
    TiffTestCase("scalar_timeless_list", FeatureType.SCALAR_TIMELESS, np.arange(5), bands=[3, 0, 2]),
    TiffTestCase("scalar_timeless_tuple", FeatureType.SCALAR_TIMELESS, np.arange(6), bands=(1, 4)),
    TiffTestCase("scalar_band_single_time_single", FeatureType.SCALAR, SCALAR_ARRAY, bands=[3], times=[7]),
    TiffTestCase(
        "scalar_band_list_time_list", FeatureType.SCALAR, SCALAR_ARRAY, bands=[2, 4, 1, 0], times=[1, 7, 0, 2, 3]
    ),
    TiffTestCase("scalar_band_tuple_time_tuple", FeatureType.SCALAR, SCALAR_ARRAY, bands=(1, 4), times=(2, 8)),
    TiffTestCase("mask_timeless", FeatureType.MASK_TIMELESS, np.arange(3 * 3 * 1).reshape(3, 3, 1)),
    TiffTestCase("mask_single", FeatureType.MASK, MASK_ARRAY, times=[4]),
    TiffTestCase("mask_list", FeatureType.MASK, MASK_ARRAY, times=[4, 2]),
    TiffTestCase("mask_tuple_int", FeatureType.MASK, MASK_ARRAY, times=(2, 4)),
    TiffTestCase(
        "mask_tuple_datetime", FeatureType.MASK, MASK_ARRAY, times=(DATES[2], DATES[4]), expected_times=(2, 4)
    ),
    TiffTestCase(
        "mask_tuple_string",
        FeatureType.MASK,
        MASK_ARRAY,
        times=(serialize_time(DATES[2]), serialize_time(DATES[4])),
        expected_times=(2, 4),
    ),
    TiffTestCase("data_timeless_band_list", FeatureType.DATA_TIMELESS, DATA_TIMELESS_ARRAY, bands=[2, 4, 1, 0]),
    TiffTestCase("data_timeless_band_tuple", FeatureType.DATA_TIMELESS, DATA_TIMELESS_ARRAY, bands=(1, 4)),
    TiffTestCase("data_band_list_time_list", FeatureType.DATA, DATA_ARRAY, bands=[2, 4, 1, 0], times=[1, 7, 0, 2, 3]),
    TiffTestCase("data_band_tuple_time_tuple", FeatureType.DATA, DATA_ARRAY, bands=(1, 4), times=(2, 8)),
    TiffTestCase("data_normal", FeatureType.DATA, DATA_ARRAY),
]


@pytest.mark.parametrize("test_case", TIFF_TEST_CASES)
def test_export_import(test_case, test_eopatch):

    test_eopatch[test_case.feature_type][test_case.name] = test_case.data

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_file_name = "temp_file.tiff"
        tmp_file_name_reproject = "temp_file_4326.tiff"
        feature = test_case.feature_type, test_case.name

        export_task = ExportToTiffTask(
            feature, folder=tmp_dir_name, band_indices=test_case.bands, date_indices=test_case.times
        )
        export_task(test_eopatch, filename=tmp_file_name)

        export_task = ExportToTiffTask(
            feature,
            folder=tmp_dir_name,
            band_indices=test_case.bands,
            date_indices=test_case.times,
            crs="EPSG:4326",
            compress="lzw",
        )
        export_task(test_eopatch, filename=tmp_file_name_reproject)

        import_task = ImportFromTiffTask(
            feature, folder=tmp_dir_name, timestamp_size=test_case.get_expected_timestamp_size()
        )

        expected_raster = test_case.get_expected()

        new_eop = import_task(filename=tmp_file_name)
        old_eop = import_task(test_eopatch, filename=tmp_file_name)

        assert_array_equal(
            expected_raster,
            new_eop[test_case.feature_type][test_case.name],
            err_msg="Tiff imported into new EOPatch is not the same as expected",
        )
        assert_array_equal(
            expected_raster,
            old_eop[test_case.feature_type][test_case.name],
            err_msg="Tiff imported into old EOPatch is not the same as expected",
        )
        assert (
            expected_raster.dtype == new_eop[test_case.feature_type][test_case.name].dtype
        ), "Tiff imported into new EOPatch has different dtype as expected"


@pytest.mark.parametrize(
    "bands, times", [([2, "string", 1, 0], [1, 7, 0, 2, 3]), ([2, 3, 1, 0], [1, 7, "string", 2, 3])]
)
def test_export2tiff_wrong_format(bands, times, test_eopatch):
    test_eopatch.data["data"] = np.arange(10 * 3 * 2 * 6, dtype=float).reshape(10, 3, 2, 6)

    with tempfile.TemporaryDirectory() as tmp_dir_name, pytest.raises(ValueError):
        tmp_file_name = "temp_file.tiff"
        task = ExportToTiffTask(
            (FeatureType.DATA, "data"), folder=tmp_dir_name, band_indices=bands, date_indices=times, image_dtype=float
        )
        task.execute(test_eopatch, filename=tmp_file_name)


def test_export2tiff_wrong_feature(mocker, test_eopatch):
    mocker.patch("logging.Logger.warning")

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_file_name = "temp_file.tiff"
        feature = FeatureType.MASK_TIMELESS, "feature-not-present"

        export_task = ExportToTiffTask(feature, folder=tmp_dir_name, fail_on_missing=False)
        export_task(test_eopatch, filename=tmp_file_name)

        assert logging.Logger.warning.call_count == 1

        (val_err,), _ = logging.Logger.warning.call_args
        assert str(val_err) == "Feature feature-not-present of type FeatureType.MASK_TIMELESS was not found in EOPatch"

        with pytest.raises(ValueError):
            failing_export_task = ExportToTiffTask(feature, folder=tmp_dir_name, fail_on_missing=True)
            failing_export_task(test_eopatch, filename=tmp_file_name)


def test_export2tiff_separate_timestamps(test_eopatch):
    test_case = TIFF_TEST_CASES[-1]
    eopatch = copy.deepcopy(test_eopatch)
    eopatch[test_case.feature_type][test_case.name] = test_case.data
    eopatch.timestamp = test_eopatch.timestamp[: test_case.data.shape[0]]

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_file_name = "temp_file_*"
        tmp_file_name_reproject = "temp_file_4326_%Y%m%d.tif"
        feature = test_case.feature_type, test_case.name

        export_task = ExportToTiffTask(feature, band_indices=test_case.bands, date_indices=test_case.times)
        full_path = os.path.join(tmp_dir_name, tmp_file_name)
        export_task(eopatch, filename=full_path)

        for timestamp in eopatch.timestamp:
            expected_path = os.path.join(tmp_dir_name, timestamp.strftime("temp_file_%Y%m%dT%H%M%S.tif"))
            assert os.path.exists(expected_path), f"Path {expected_path} does not exist"

        full_path = os.path.join(tmp_dir_name, tmp_file_name_reproject)
        export_task = ExportToTiffTask(
            feature,
            folder=full_path,
            band_indices=test_case.bands,
            date_indices=test_case.times,
            crs="EPSG:4326",
            compress="lzw",
        )
        export_task(eopatch)

        for timestamp in eopatch.timestamp:
            expected_path = os.path.join(tmp_dir_name, timestamp.strftime(tmp_file_name_reproject))
            assert os.path.exists(expected_path), f"Path {expected_path} does not exist"


def test_import_tiff_subset(test_eopatch, example_data_path):
    path = os.path.join(example_data_path, "import-tiff-test1.tiff")
    mask_feature = FeatureType.MASK_TIMELESS, "TEST_TIF"

    task = ImportFromTiffTask(mask_feature, path)
    task(test_eopatch)

    tiff_img = read_data(path)

    assert_array_equal(
        tiff_img[20:53, 21:54],
        test_eopatch[mask_feature][..., 0],
        err_msg="Imported tiff data should be the same as original",
    )


def test_import_tiff_intersecting(test_eopatch, example_data_path):
    path = os.path.join(example_data_path, "import-tiff-test2.tiff")
    feature = FeatureType.DATA_TIMELESS, "TEST_TIF"
    no_data_value = 1.0

    task = ImportFromTiffTask(feature, path, image_dtype=np.float64, no_data_value=no_data_value)
    task(test_eopatch)

    tiff_img = read_data(path)

    assert_array_equal(
        tiff_img[-6:, :3, :],
        test_eopatch[feature][:6, -3:, :],
        err_msg="Imported tiff data should be the same as original",
    )
    feature_dtype = test_eopatch[feature].dtype
    assert feature_dtype == np.float64, f"Feature should have dtype numpy.float64 but {feature_dtype} found"

    test_eopatch[feature][:6, -3:, :] = no_data_value
    unique_values = list(np.unique(test_eopatch[feature][:6, -3:, :]))
    assert unique_values == [no_data_value], f"No data values should all be equal to {no_data_value}"


def test_timeless_feature(test_eopatch):
    feature = FeatureType.DATA_TIMELESS, "DEM"
    filename = "relative-path/my-filename.tiff"

    export_task = ExportToTiffTask(feature, folder=PATH_ON_BUCKET)
    import_task = ImportFromTiffTask(feature, folder=PATH_ON_BUCKET)

    export_task.execute(test_eopatch, filename=filename)
    new_eopatch = import_task.execute(test_eopatch, filename=filename)

    assert_array_equal(new_eopatch[feature], test_eopatch[feature])


def test_time_dependent_feature(test_eopatch):
    feature = FeatureType.DATA, "NDVI"
    filename_export = "relative-path/*.tiff"
    filename_import = [
        f'relative-path/{timestamp.strftime("%Y%m%dT%H%M%S")}.tiff' for timestamp in test_eopatch.timestamp
    ]

    export_task = ExportToTiffTask(feature, folder=PATH_ON_BUCKET)
    import_task = ImportFromTiffTask(feature, folder=PATH_ON_BUCKET, timestamp_size=68)

    export_task(test_eopatch, filename=filename_export)
    new_eopatch = import_task(filename=filename_import)

    assert_array_equal(new_eopatch[feature], test_eopatch[feature])

    test_eopatch.timestamp[-1] = datetime.datetime(2020, 10, 10)
    filename_import = [
        f'relative-path/{timestamp.strftime("%Y%m%dT%H%M%S")}.tiff' for timestamp in test_eopatch.timestamp
    ]

    with pytest.raises(ResourceNotFound):
        import_task(filename=filename_import)


def test_time_dependent_feature_with_timestamps(test_eopatch):
    feature = FeatureType.DATA, "NDVI"
    filename = "relative-path/%Y%m%dT%H%M%S.tiff"

    export_task = ExportToTiffTask(feature, folder=PATH_ON_BUCKET)
    import_task = ImportFromTiffTask(feature, folder=PATH_ON_BUCKET)

    export_task.execute(test_eopatch, filename=filename)
    new_eopatch = import_task(test_eopatch, filename=filename)

    assert_array_equal(new_eopatch[feature], test_eopatch[feature])
