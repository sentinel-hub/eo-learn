"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime
import os
import tempfile

import pytest
import numpy as np
import fs
from fs.errors import CreateFailed, ResourceNotFound
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from geopandas import GeoDataFrame
from moto import mock_s3
import boto3

from sentinelhub import BBox, CRS
from eolearn.core import EOPatch, FeatureType, OverwritePermission, SaveTask, LoadTask


@mock_s3
def _create_new_s3_fs():
    """Creates a new empty mocked s3 bucket. If one such bucket already exists it deletes it first."""
    bucket_name = "mocked-test-bucket"
    s3resource = boto3.resource("s3", region_name="eu-central-1")

    bucket = s3resource.Bucket(bucket_name)

    if bucket.creation_date:  # If bucket already exists
        for key in bucket.objects.all():
            key.delete()
        bucket.delete()

    s3resource.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})

    return S3FS(bucket_name=bucket_name)


FS_LOADERS = [TempFS, _create_new_s3_fs]


@pytest.fixture(name="eopatch")
def eopatch_fixture():
    eopatch = EOPatch()
    mask = np.zeros((3, 3, 2), dtype=np.int16)
    data = np.zeros((2, 3, 3, 2), dtype=np.int16)
    eopatch.data_timeless["mask"] = mask
    eopatch.data["data"] = data
    eopatch.timestamp = [datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.datetime(2017, 1, 4, 10, 14, 5)]
    eopatch.meta_info["something"] = "nothing"
    eopatch.meta_info["something-else"] = "nothing"
    eopatch.bbox = BBox((1, 2, 3, 4), CRS.WGS84)
    eopatch.scalar["my scalar with spaces"] = np.array([[1, 2, 3], [1, 2, 3]])
    eopatch.scalar_timeless["my timeless scalar with spaces"] = np.array([1, 2, 3])
    eopatch.vector["my-df"] = GeoDataFrame(
        {
            "values": [1, 2],
            "TIMESTAMP": [datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.datetime(2017, 1, 4, 10, 14, 5)],
            "geometry": [eopatch.bbox.geometry, eopatch.bbox.geometry],
        },
        crs=eopatch.bbox.crs.pyproj_crs(),
    )

    return eopatch


def test_saving_to_a_file(eopatch):
    with tempfile.NamedTemporaryFile() as fp:
        with pytest.raises(CreateFailed):
            eopatch.save(fp.name)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_saving_in_empty_folder(eopatch, fs_loader):
    with fs_loader() as temp_fs:

        if isinstance(temp_fs, TempFS):
            eopatch.save(temp_fs.root_path)
        else:
            eopatch.save("/", filesystem=temp_fs)
        assert temp_fs.exists("/data_timeless/mask.npy")

        subfolder = "new-subfolder"
        eopatch.save("new-subfolder", filesystem=temp_fs)
        assert temp_fs.exists(f"/{subfolder}/bbox.geojson")


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_saving_in_non_empty_folder(eopatch, fs_loader):
    with fs_loader() as temp_fs:
        empty_file = "foo.txt"

        with temp_fs.open(empty_file, "w"):
            pass

        eopatch.save("/", filesystem=temp_fs)
        assert temp_fs.exists(empty_file)

        eopatch.save("/", overwrite_permission=OverwritePermission.OVERWRITE_PATCH, filesystem=temp_fs)
        assert not temp_fs.exists(empty_file)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_overwriting_non_empty_folder(eopatch, fs_loader):
    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs)
        eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)
        eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

        add_eopatch = EOPatch()
        add_eopatch.data_timeless["some data"] = np.empty((3, 3, 2))
        add_eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.ADD_ONLY)
        with pytest.raises(ValueError):
            add_eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.ADD_ONLY)

        new_eopatch = EOPatch.load("/", filesystem=temp_fs, lazy_loading=False)
        assert new_eopatch == eopatch + add_eopatch


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_save_load(eopatch, fs_loader):
    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs)
        eopatch2 = EOPatch.load("/", filesystem=temp_fs)
        assert eopatch == eopatch2

        eopatch2.save("/", filesystem=temp_fs, overwrite_permission=1)
        eopatch2 = EOPatch.load("/", filesystem=temp_fs)
        assert eopatch == eopatch2

        eopatch2.save("/", filesystem=temp_fs, overwrite_permission=1)
        eopatch2 = EOPatch.load("/", filesystem=temp_fs, lazy_loading=False)
        assert eopatch == eopatch2

        features = {FeatureType.DATA_TIMELESS: ["mask"], FeatureType.TIMESTAMP: ...}
        eopatch2.save("/", filesystem=temp_fs, features=features, compress_level=3, overwrite_permission=1)
        eopatch2 = EOPatch.load("/", filesystem=temp_fs, lazy_loading=True)
        assert eopatch == eopatch2
        eopatch3 = EOPatch.load("/", filesystem=temp_fs, lazy_loading=True, features=features)
        assert eopatch != eopatch3


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_save_add_only_features(eopatch, fs_loader):
    features = [
        (FeatureType.DATA_TIMELESS, "mask"),
        FeatureType.MASK,
        FeatureType.VECTOR,
        (FeatureType.SCALAR, ...),
        (FeatureType.META_INFO, "something"),
        FeatureType.BBOX,
    ]

    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs, features=features, overwrite_permission=0)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_overwrite_failure(fs_loader):
    eopatch = EOPatch()
    mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
    eopatch.data_timeless["mask"] = mask
    eopatch.data_timeless["Mask"] = mask

    with fs_loader() as temp_fs, pytest.raises(IOError):
        eopatch.save("/", filesystem=temp_fs)

    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs, features=[(FeatureType.DATA_TIMELESS, "mask")], overwrite_permission=2)

        with pytest.raises(IOError):
            eopatch.save(
                "/", filesystem=temp_fs, features=[(FeatureType.DATA_TIMELESS, "Mask")], overwrite_permission=0
            )


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_save_and_load_tasks(eopatch, fs_loader):
    folder = "foo-folder"
    patch_folder = "patch-folder"
    with fs_loader() as temp_fs:
        temp_fs.makedir(folder)

        save_task = SaveTask(folder, filesystem=temp_fs, compress_level=9)
        load_task = LoadTask(folder, filesystem=temp_fs, lazy_loading=False)

        saved_eop = save_task(eopatch, eopatch_folder=patch_folder)
        bbox_path = fs.path.join(folder, patch_folder, "bbox.geojson.gz")
        assert temp_fs.exists(bbox_path)
        assert saved_eop == eopatch

        eop = load_task(eopatch_folder=patch_folder)
        assert eop == eopatch


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_fail_saving_nonexistent_feature(eopatch, fs_loader):
    features = [(FeatureType.DATA, "nonexistent")]
    with fs_loader() as temp_fs, pytest.raises(ValueError):
        eopatch.save("/", filesystem=temp_fs, features=features)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_fail_loading_nonexistent_feature(fs_loader):
    for features in [[(FeatureType.DATA, "nonexistent")], [(FeatureType.META_INFO, "nonexistent")]]:
        with fs_loader() as temp_fs, pytest.raises(IOError):
            EOPatch.load("/", filesystem=temp_fs, features=features)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_nonexistent_location(fs_loader):
    path = "./folder/subfolder/new-eopatch/"
    empty_eop = EOPatch()

    with fs_loader() as temp_fs:
        with pytest.raises(ResourceNotFound):
            EOPatch.load(path, filesystem=temp_fs)

        empty_eop.save(path, filesystem=temp_fs)

    with TempFS() as temp_fs:
        full_path = os.path.join(temp_fs.root_path, path)
        with pytest.raises(CreateFailed):
            EOPatch.load(full_path)

        load_task = LoadTask(full_path)
        with pytest.raises(CreateFailed):
            load_task.execute()

        empty_eop.save(full_path)
        assert os.path.exists(full_path)

    with TempFS() as temp_fs:
        full_path = os.path.join(temp_fs.root_path, path)
        save_task = SaveTask(full_path)
        save_task.execute(empty_eop)
        assert os.path.exists(full_path)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_cleanup_different_compression(fs_loader, eopatch):
    folder = "foo-folder"
    patch_folder = "patch-folder"
    with fs_loader() as temp_fs:
        temp_fs.makedir(folder)

        save_compressed_task = SaveTask(folder, filesystem=temp_fs, compress_level=9, overwrite_permission=1)
        save_noncompressed_task = SaveTask(folder, filesystem=temp_fs, compress_level=0, overwrite_permission=1)
        bbox_path = fs.path.join(folder, patch_folder, "bbox.geojson")
        compressed_bbox_path = bbox_path + ".gz"
        data_timeless_path = fs.path.join(folder, patch_folder, "data_timeless", "mask.npy")
        compressed_data_timeless_path = data_timeless_path + ".gz"

        save_compressed_task(eopatch, eopatch_folder=patch_folder)
        save_noncompressed_task(eopatch, eopatch_folder=patch_folder)
        assert temp_fs.exists(bbox_path)
        assert temp_fs.exists(data_timeless_path)
        assert not temp_fs.exists(compressed_bbox_path)
        assert not temp_fs.exists(compressed_data_timeless_path)

        save_compressed_task(eopatch, eopatch_folder=patch_folder)
        assert not temp_fs.exists(bbox_path)
        assert not temp_fs.exists(data_timeless_path)
        assert temp_fs.exists(compressed_bbox_path)
        assert temp_fs.exists(compressed_data_timeless_path)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
@pytest.mark.parametrize("folder_name", ["/", "foo", "foo/bar"])
def test_lazy_loading_plus_overwrite_patch(fs_loader, folder_name, eopatch):
    with fs_loader() as temp_fs:
        eopatch.save(folder_name, filesystem=temp_fs)

        lazy_eopatch = EOPatch.load(folder_name, filesystem=temp_fs, lazy_loading=True)
        lazy_eopatch.data["whatever"] = np.empty((2, 3, 3, 2))
        del lazy_eopatch[FeatureType.DATA_TIMELESS, "mask"]

        lazy_eopatch.save(folder_name, filesystem=temp_fs, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        assert temp_fs.exists(fs.path.join(folder_name, "data", "whatever.npy"))
        assert not temp_fs.exists(fs.path.join(folder_name, "data_timeless", "mask.npy"))
