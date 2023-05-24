"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import datetime
import os
import tempfile
import warnings
from typing import Any, Type

import fs
import geopandas as gpd
import numpy as np
import pytest
from fs.errors import CreateFailed, ResourceNotFound
from fs.tempfs import TempFS
from geopandas import GeoDataFrame
from moto import mock_s3
from shapely.geometry import Point

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureType, LoadTask, OverwritePermission, SaveTask, merge_eopatches
from eolearn.core.constants import TIMESTAMP_COLUMN
from eolearn.core.eodata_io import (
    FeatureIO,
    FeatureIOBBox,
    FeatureIOGeoDf,
    FeatureIOJson,
    FeatureIONumpy,
    FeatureIOTimestamps,
    walk_filesystem,
)
from eolearn.core.exceptions import EODeprecationWarning
from eolearn.core.types import FeaturesSpecification
from eolearn.core.utils.parsing import FeatureParser
from eolearn.core.utils.testing import assert_feature_data_equal, generate_eopatch

FS_LOADERS = [TempFS, pytest.lazy_fixture("create_mocked_s3fs")]

DUMMY_BBOX = BBox((0, 0, 1, 1), CRS.WGS84)


@pytest.fixture(name="_silence_warnings")
def _silence_warnings_fixture():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=EODeprecationWarning)
        yield


@pytest.fixture(name="eopatch")
def eopatch_fixture():
    eopatch = generate_eopatch(
        {
            FeatureType.DATA: ["data"],
            FeatureType.MASK_TIMELESS: ["mask"],
            FeatureType.SCALAR: ["my scalar with spaces"],
            FeatureType.SCALAR_TIMELESS: ["my timeless scalar with spaces"],
            FeatureType.META_INFO: ["something", "something-else"],
        }
    )
    eopatch.vector["my-df"] = GeoDataFrame(
        {
            "values": [1, 2],
            TIMESTAMP_COLUMN: [datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.datetime(2017, 1, 4, 10, 14, 5)],
            "geometry": [DUMMY_BBOX.geometry, DUMMY_BBOX.geometry],
        },
        crs=DUMMY_BBOX.crs.pyproj_crs(),
    )
    eopatch.vector_timeless["empty-vector"] = GeoDataFrame(
        {"values": [], "geometry": []}, crs=eopatch.bbox.crs.pyproj_crs()
    )

    return eopatch


def test_saving_to_a_file(eopatch):
    with tempfile.NamedTemporaryFile() as fp, pytest.raises(CreateFailed):
        eopatch.save(fp.name)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_saving_in_empty_folder(eopatch, fs_loader):
    with fs_loader() as temp_fs:
        if isinstance(temp_fs, TempFS):
            eopatch.save(temp_fs.root_path)
        else:
            eopatch.save("/", filesystem=temp_fs)
        assert temp_fs.exists("/mask_timeless/mask.npy")

        subfolder = "new-subfolder"
        eopatch.save("new-subfolder", filesystem=temp_fs)
        assert temp_fs.exists(f"/{subfolder}/bbox.geojson")


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
@pytest.mark.usefixtures("_silence_warnings")
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
@pytest.mark.usefixtures("_silence_warnings")
def test_overwriting_non_empty_folder(eopatch, fs_loader):
    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs)
        eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)
        eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

        add_eopatch = EOPatch(bbox=eopatch.bbox)
        add_eopatch.data_timeless["some data"] = np.empty((3, 3, 2))
        add_eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.ADD_ONLY)
        with pytest.raises(ValueError):
            add_eopatch.save("/", filesystem=temp_fs, overwrite_permission=OverwritePermission.ADD_ONLY)

        new_eopatch = EOPatch.load("/", filesystem=temp_fs, lazy_loading=False)
        assert new_eopatch == merge_eopatches(eopatch, add_eopatch)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
@pytest.mark.parametrize(
    ("save_features", "load_features"),
    [
        (..., ...),
        ([(FeatureType.DATA, ...), FeatureType.TIMESTAMPS], [(FeatureType.DATA, ...), FeatureType.TIMESTAMPS]),
        ([(FeatureType.DATA, "data"), FeatureType.TIMESTAMPS], [(FeatureType.DATA, ...)]),
        ([(FeatureType.DATA, "data"), FeatureType.TIMESTAMPS], ...),
    ],
)
def test_save_load_partial(
    eopatch: EOPatch, fs_loader, save_features: FeaturesSpecification, load_features: FeaturesSpecification
):
    with fs_loader() as temp_fs:
        eopatch.save("/", features=save_features, filesystem=temp_fs)
        loaded_eopatch = EOPatch.load("/", features=load_features, filesystem=temp_fs)

        # have to check features that have been saved and then loaded (double filtering)
        features_to_load = FeatureParser(load_features).get_features(eopatch)
        for feature in FeatureParser(save_features).get_features(eopatch):
            if feature in features_to_load:
                assert feature in loaded_eopatch


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_save_add_only_features(eopatch, fs_loader):
    features = [
        (FeatureType.MASK_TIMELESS, "mask"),
        FeatureType.MASK,
        FeatureType.VECTOR,
        (FeatureType.SCALAR, ...),
        (FeatureType.META_INFO, "something"),
        FeatureType.BBOX,
    ]

    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs, features=features, overwrite_permission=OverwritePermission.ADD_ONLY)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_bbox_always_saved(eopatch, fs_loader):
    with fs_loader() as temp_fs:
        eopatch.save("/", filesystem=temp_fs, features=[FeatureType.DATA])
        assert temp_fs.exists("/bbox.geojson")


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
@pytest.mark.usefixtures("_silence_warnings")
def test_overwrite_failure(fs_loader):
    eopatch = EOPatch(bbox=DUMMY_BBOX)
    mask = np.arange(3 * 3 * 2).reshape(3, 3, 2)
    eopatch.mask_timeless["mask"] = mask
    eopatch.mask_timeless["Mask"] = mask

    with fs_loader() as temp_fs, pytest.raises(IOError):
        eopatch.save("/", filesystem=temp_fs)

    with fs_loader() as temp_fs:
        eopatch.save(
            "/",
            filesystem=temp_fs,
            features=[(FeatureType.MASK_TIMELESS, "mask")],
            overwrite_permission=OverwritePermission.OVERWRITE_PATCH,
        )

        with pytest.raises(IOError):
            eopatch.save(
                "/",
                filesystem=temp_fs,
                features=[(FeatureType.MASK_TIMELESS, "Mask")],
                overwrite_permission=OverwritePermission.ADD_ONLY,
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
    """In the event of a path not existing all save actions should create the path, and loads should fail."""
    path = "./folder/subfolder/new-eopatch/"
    eopatch = EOPatch(bbox=DUMMY_BBOX)

    # IO on nonexistent path inside a temporary FS
    with fs_loader() as temp_fs:
        with pytest.raises(ResourceNotFound):
            EOPatch.load(path, filesystem=temp_fs)

        eopatch.save(path, filesystem=temp_fs)

    # IO on nonexistent path (no fs specified)
    with TempFS() as temp_fs:
        full_path = os.path.join(temp_fs.root_path, path)
        with pytest.raises(CreateFailed):
            EOPatch.load(full_path)

        load_task = LoadTask(full_path)
        with pytest.raises(CreateFailed):
            load_task.execute()

        eopatch.save(full_path)
        assert os.path.exists(full_path)

    # SaveTask on nonexistent path (no fs specified)
    with TempFS() as temp_fs:
        full_path = os.path.join(temp_fs.root_path, path)
        SaveTask(full_path).execute(eopatch)
        assert os.path.exists(full_path)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
def test_cleanup_different_compression(fs_loader, eopatch):
    folder = "foo-folder"
    patch_folder = "patch-folder"
    with fs_loader() as temp_fs:
        temp_fs.makedir(folder)

        save_compressed_task = SaveTask(
            folder, filesystem=temp_fs, compress_level=9, overwrite_permission="OVERWRITE_FEATURES"
        )
        save_noncompressed_task = SaveTask(
            folder, filesystem=temp_fs, compress_level=0, overwrite_permission="OVERWRITE_FEATURES"
        )
        bbox_path = fs.path.join(folder, patch_folder, "bbox.geojson")
        compressed_bbox_path = bbox_path + ".gz"
        mask_timeless_path = fs.path.join(folder, patch_folder, "mask_timeless", "mask.npy")
        compressed_mask_timeless_path = mask_timeless_path + ".gz"

        save_compressed_task(eopatch, eopatch_folder=patch_folder)
        save_noncompressed_task(eopatch, eopatch_folder=patch_folder)
        assert temp_fs.exists(bbox_path)
        assert temp_fs.exists(mask_timeless_path)
        assert not temp_fs.exists(compressed_bbox_path)
        assert not temp_fs.exists(compressed_mask_timeless_path)

        save_compressed_task(eopatch, eopatch_folder=patch_folder)
        assert not temp_fs.exists(bbox_path)
        assert not temp_fs.exists(mask_timeless_path)
        assert temp_fs.exists(compressed_bbox_path)
        assert temp_fs.exists(compressed_mask_timeless_path)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
@pytest.mark.parametrize("folder_name", ["/", "foo", "foo/bar"])
@pytest.mark.usefixtures("_silence_warnings")
def test_lazy_loading_plus_overwrite_patch(fs_loader, folder_name, eopatch):
    with fs_loader() as temp_fs:
        eopatch.save(folder_name, filesystem=temp_fs)

        lazy_eopatch = EOPatch.load(folder_name, filesystem=temp_fs, lazy_loading=True)
        lazy_eopatch.data["whatever"] = np.empty((2, 3, 3, 2))
        del lazy_eopatch[FeatureType.MASK_TIMELESS, "mask"]

        lazy_eopatch.save(folder_name, filesystem=temp_fs, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        assert temp_fs.exists(fs.path.join(folder_name, "data", "whatever.npy"))
        assert not temp_fs.exists(fs.path.join(folder_name, "mask_timeless", "mask.npy"))


@pytest.mark.parametrize(
    ("constructor", "data"),
    [
        (FeatureIONumpy, np.zeros(20)),
        (FeatureIONumpy, np.zeros((2, 3, 3, 2), dtype=np.int16)),
        (FeatureIONumpy, np.full((4, 5), fill_value=CRS.POP_WEB)),
        (FeatureIOGeoDf, gpd.GeoDataFrame({"col1": ["name1"], "geometry": [Point(1, 2)]}, crs="EPSG:3857")),
        (FeatureIOGeoDf, gpd.GeoDataFrame({"col1": ["name1"], "geometry": [Point(1, 2)]}, crs="EPSG:32733")),
        (
            FeatureIOGeoDf,
            gpd.GeoDataFrame(
                {
                    "values": [1, 2],
                    TIMESTAMP_COLUMN: [
                        datetime.datetime(2017, 1, 1, 10, 4, 7),
                        datetime.datetime(2017, 1, 4, 10, 14, 5),
                    ],
                    "geometry": [Point(1, 2), Point(2, 1)],
                },
                crs="EPSG:3857",
            ),
        ),
        (FeatureIOJson, {}),
        (FeatureIOJson, {"test": "test1", "test3": {"test": "test1"}}),
        (FeatureIOBBox, BBox((1, 2, 3, 4), CRS.WGS84)),
        (FeatureIOTimestamps, []),
        (FeatureIOTimestamps, [datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.datetime(2017, 1, 4, 10, 14, 5)]),
    ],
)
@pytest.mark.parametrize("compress_level", [0, 1])
def test_feature_io(constructor: Type[FeatureIO], data: Any, compress_level: int) -> None:
    """
    Tests verifying that FeatureIO subclasses correctly save, load, and lazy-load data.
    Test cases do not include subfolders, because subfolder management is currently done by the `save_eopatch` function.
    """

    file_extension = "." + str(constructor.get_file_format().extension)
    file_extension = file_extension if compress_level == 0 else file_extension + ".gz"
    file_name = "name"
    with TempFS("testing_file_sistem") as temp_fs:
        feat_io = constructor(file_name + file_extension, filesystem=temp_fs)
        constructor.save(data, temp_fs, file_name, compress_level)
        loaded_data = feat_io.load()
        assert_feature_data_equal(loaded_data, data)

        temp_fs.remove(file_name + file_extension)
        cache_data = feat_io.load()
        assert_feature_data_equal(loaded_data, cache_data)


@mock_s3
@pytest.mark.parametrize("fs_loader", FS_LOADERS)
@pytest.mark.parametrize(
    "features",
    [
        ...,
        [(FeatureType.DATA, "data"), FeatureType.TIMESTAMPS],
        [(FeatureType.META_INFO, "something"), (FeatureType.SCALAR_TIMELESS, ...)],
    ],
)
def test_walk_filesystem_interface(fs_loader, features, eopatch):
    with fs_loader() as temp_fs:
        io_kwargs = dict(path="./", filesystem=temp_fs, features=features)
        eopatch.save(**io_kwargs)
        loaded_eopatch = EOPatch.load(**io_kwargs)

        with pytest.warns(EODeprecationWarning):
            for ftype, fname, _ in walk_filesystem(temp_fs, io_kwargs["path"], features):
                feature_key = ftype if ftype.is_meta() else (ftype, fname)
                assert feature_key in loaded_eopatch
