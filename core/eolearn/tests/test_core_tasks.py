"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import copy
import pickle
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union

import numpy as np
import pytest
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from numpy.testing import assert_array_equal, assert_equal
from pytest import approx

from sentinelhub import CRS, BBox

from eolearn.core import (
    AddFeatureTask,
    CopyTask,
    CreateEOPatchTask,
    DeepCopyTask,
    DuplicateFeatureTask,
    EOPatch,
    ExtractBandsTask,
    FeatureType,
    InitializeFeatureTask,
    LoadTask,
    MapFeatureTask,
    MergeEOPatchesTask,
    MergeFeatureTask,
    MoveFeatureTask,
    RemoveFeatureTask,
    RenameFeatureTask,
    SaveTask,
    ZipFeatureTask,
)
from eolearn.core.core_tasks import ExplodeBandsTask
from eolearn.core.types import FeatureRenameSpec, FeatureSpec, FeaturesSpecification
from eolearn.core.utils.parsing import parse_features

DUMMY_BBOX = BBox((0, 0, 1, 1), CRS(3857))


@pytest.fixture(name="patch")
def patch_fixture() -> EOPatch:
    patch = EOPatch()
    patch.data["bands"] = np.arange(5 * 3 * 4 * 2).reshape(5, 3, 4, 2)
    patch.data["CLP"] = np.full((5, 3, 4, 1), 0.7)
    patch.data["CLP_S2C"] = np.zeros((5, 3, 4, 1), dtype=np.int64)
    patch.mask["CLM"] = np.full((5, 3, 4, 1), True)
    patch.mask_timeless["mask"] = np.arange(3 * 4 * 2).reshape(3, 4, 2)
    patch.mask_timeless["LULC"] = np.zeros((3, 4, 1), dtype=np.uint16)
    patch.mask_timeless["RANDOM_UINT8"] = np.random.randint(0, 100, size=(3, 4, 1), dtype=np.int8)
    patch.scalar["values"] = np.arange(10 * 5).reshape(10, 5)
    patch.scalar["CLOUD_COVERAGE"] = np.ones((10, 5))
    patch.timestamps = [
        datetime(2017, 1, 14, 10, 13, 46),
        datetime(2017, 2, 10, 10, 1, 32),
        datetime(2017, 2, 20, 10, 6, 35),
        datetime(2017, 3, 2, 10, 0, 20),
        datetime(2017, 3, 12, 10, 7, 6),
    ]
    patch.bbox = BBox((324.54, 546.45, 955.4, 63.43), CRS(3857))
    patch.meta_info["something"] = np.random.rand(10, 1)
    return patch


@pytest.mark.parametrize("task", [DeepCopyTask, CopyTask])
def test_copy(task: Type[CopyTask], patch: EOPatch) -> None:
    patch_copy = task().execute(patch)
    assert patch_copy == patch

    patch_copy.data["bands"][0, 0, 0, 0] += 1
    assert (patch_copy != patch) if task == DeepCopyTask else (patch_copy == patch)

    patch_copy.data["new"] = np.arange(1).reshape(1, 1, 1, 1)
    assert "new" not in patch.data


@pytest.mark.parametrize(
    "features",
    [
        [(FeatureType.MASK_TIMELESS, "mask"), (FeatureType.BBOX, None)],
        [(FeatureType.TIMESTAMP, None), (FeatureType.SCALAR, "values")],
    ],
)
@pytest.mark.parametrize("task", [DeepCopyTask, CopyTask])
def test_partial_copy(features: List[FeatureSpec], task: Type[CopyTask], patch: EOPatch) -> None:
    patch_copy = task(features=features)(patch)

    assert set(patch_copy.get_features()) == {(FeatureType.BBOX, None), *features}

    for feature in features:
        if isinstance(patch[feature], np.ndarray):
            assert_array_equal(patch_copy[feature], patch[feature])
        else:
            assert patch_copy[feature] == patch[feature]


def test_load_task(test_eopatch_path):
    full_load = LoadTask(test_eopatch_path)
    full_patch = full_load.execute(eopatch_folder=".")
    assert len(full_patch.get_features()) == 30

    partial_load = LoadTask(test_eopatch_path, features=[FeatureType.BBOX, FeatureType.MASK_TIMELESS])
    partial_patch = partial_load.execute(eopatch_folder=".")

    assert FeatureType.BBOX in partial_patch and FeatureType.TIMESTAMPS not in partial_patch

    load_more = LoadTask(test_eopatch_path, features=[FeatureType.TIMESTAMPS])
    upgraded_partial_patch = load_more.execute(partial_patch, eopatch_folder=".")
    assert FeatureType.BBOX in upgraded_partial_patch and FeatureType.TIMESTAMPS in upgraded_partial_patch
    assert FeatureType.DATA not in upgraded_partial_patch


def test_load_nothing():
    load = LoadTask("./some/fake/path")
    eopatch = load.execute(eopatch_folder=None)

    assert eopatch == EOPatch()


def test_save_nothing(patch):
    temp_path = "/some/fake/path"
    with TempFS() as temp_fs:
        save = SaveTask(temp_path, filesystem=temp_fs)
        output = save.execute(patch, eopatch_folder=None)

        assert not temp_fs.exists(temp_path)
        assert output == patch


@pytest.mark.parametrize("filesystem", [OSFS("."), S3FS("s3://fake-bucket/"), TempFS()])
@pytest.mark.parametrize("task_class", [LoadTask, SaveTask])
def test_io_task_pickling(filesystem, task_class):
    task = task_class("/", filesystem=filesystem)

    pickled_task = pickle.dumps(task)
    unpickled_task = pickle.loads(pickled_task)
    assert isinstance(unpickled_task, task_class)


@pytest.mark.parametrize(
    "feature, feature_data",
    [
        ((FeatureType.MASK, "CLOUD MASK"), np.arange(10).reshape(5, 2, 1, 1)),
        ((FeatureType.META_INFO, "something_else"), np.random.rand(10, 1)),
        ((FeatureType.TIMESTAMPS, None), [datetime(2022, 1, 1, 10, 4, 7), datetime(2022, 1, 4, 10, 14, 5)]),
    ],
)
def test_add_feature(feature: FeatureSpec, feature_data: Any) -> None:
    # this test should fail for bbox and timestamps after rework
    patch = EOPatch(bbox=DUMMY_BBOX)
    assert feature not in patch
    patch = AddFeatureTask(feature)(patch, feature_data)

    if isinstance(feature_data, np.ndarray):
        assert_array_equal(patch[feature], feature_data)
    else:
        assert patch[feature] == feature_data


def test_rename_feature(patch: EOPatch) -> None:
    f_type, f_name, f_new_name = FeatureType.DATA, "bands", "new_bands"
    assert (f_type, f_new_name) not in patch
    patch_copy = copy.deepcopy(patch)

    patch = RenameFeatureTask((f_type, f_name, f_new_name))(patch)
    assert_array_equal(patch[(f_type, f_new_name)], patch_copy[(f_type, f_name)])
    assert (f_type, f_name) not in patch, "Feature was not removed from patch. "


@pytest.mark.parametrize("feature", [(FeatureType.DATA, "bands"), (FeatureType.TIMESTAMPS, None)])
def test_remove_feature(feature: FeatureSpec, patch: EOPatch) -> None:
    patch_copy = copy.deepcopy(patch)
    assert feature in patch

    patch = RemoveFeatureTask(feature)(patch)
    assert feature not in patch

    del patch_copy[feature]
    assert patch == patch_copy


@pytest.mark.skip
def test_remove_fails(patch: EOPatch) -> None:
    with pytest.raises(ValueError):
        RemoveFeatureTask((FeatureType.BBOX, None))(patch)


@pytest.mark.parametrize(
    "feature_specification",
    [
        [(FeatureType.DATA, "bands", "bands2")],
        [(FeatureType.DATA, "bands", "bands2"), (FeatureType.MASK_TIMELESS, "mask", "mask2")],
        [(FeatureType.DATA, "bands", f"bands{i}") for i in range(5)],
    ],
)
@pytest.mark.parametrize("deep", [True, False])
def test_duplicate_feature(feature_specification: List[FeatureRenameSpec], deep: bool, patch: EOPatch) -> None:
    patch = DuplicateFeatureTask(feature_specification, deep)(patch)

    for f_type, f_name, f_dup_name in feature_specification:
        original_feature = (f_type, f_name)
        duplicated_feature = (f_type, f_dup_name)
        assert duplicated_feature in patch

        original_id = id(patch[original_feature])
        duplicated_id = id(patch[duplicated_feature])
        assert original_id != duplicated_id if deep else original_id == duplicated_id

        assert_array_equal(patch[original_feature], patch[duplicated_feature])


def test_duplicate_feature_fails(patch: EOPatch) -> None:
    with pytest.raises(ValueError):
        # Expected a ValueError when creating an already exising feature.
        DuplicateFeatureTask((FeatureType.DATA, "bands", "bands"))(patch)


@pytest.mark.parametrize(
    "init_val, shape, feature_spec",
    [
        (8, (5, 2, 6, 3), (FeatureType.MASK, "test")),
        (9, (1, 4, 3), (FeatureType.MASK_TIMELESS, "test")),
        (7, (5, 2, 7, 4), {FeatureType.MASK: ["F1", "F2", "F3"]}),
    ],
)
def test_initialize_feature(
    init_val: float, shape: Tuple[int, ...], feature_spec: FeaturesSpecification, patch: EOPatch
) -> None:
    expected_data = init_val * np.ones(shape)
    patch = InitializeFeatureTask(feature_spec, shape=shape, init_value=init_val)(patch)

    assert all([np.array_equal(patch[features], expected_data) for features in parse_features(feature_spec)])


@pytest.mark.parametrize(
    "init_val, shape, feature_spec",
    [
        (3, (FeatureType.DATA, "bands"), {FeatureType.MASK: ["F1", "F2", "F3"]}),
    ],
)
def test_initialize_feature_with_spec(
    init_val: float, shape: FeatureSpec, feature_spec: FeaturesSpecification, patch: EOPatch
) -> None:
    expected_data = init_val * np.ones(patch[shape].shape)

    patch = InitializeFeatureTask(feature_spec, shape=shape, init_value=init_val)(patch)
    assert all([np.array_equal(patch[features], expected_data) for features in parse_features(feature_spec)])


def test_initialize_feature_fails(patch: EOPatch) -> None:
    with pytest.raises(ValueError):
        # Expected a ValueError when trying to initialize a feature with a wrong shape dimensions.
        InitializeFeatureTask((FeatureType.MASK_TIMELESS, "wrong"), (5, 10, 10, 3), 123)(patch)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "features",
    [
        [(FeatureType.DATA, "bands")],
        [(FeatureType.DATA, "bands"), (FeatureType.MASK_TIMELESS, "mask")],
        [(FeatureType.DATA, "bands"), (FeatureType.BBOX, None)],
    ],
)
def test_move_feature(features: FeatureSpec, deep: bool, patch: EOPatch) -> None:
    patch_dst = MoveFeatureTask(features, deep_copy=deep)(patch, EOPatch(bbox=DUMMY_BBOX))

    for feat in features:
        assert feat in patch_dst

        original_id = id(patch[feat])
        duplicated_id = id(patch_dst[feat])
        assert original_id != duplicated_id if deep else original_id == duplicated_id

        if isinstance(patch[feat], np.ndarray):
            assert_array_equal(patch[feat], patch_dst[feat])
        else:
            assert patch[feat] == patch_dst[feat]


@pytest.mark.parametrize(
    "features_to_merge, feature, axis",
    [
        ([(FeatureType.DATA, "bands")], (FeatureType.DATA, "merged"), 0),
        ([(FeatureType.DATA, "bands"), (FeatureType.DATA, "CLP")], (FeatureType.DATA, "merged"), -1),
        ([(FeatureType.DATA, "CLP_S2C"), (FeatureType.DATA, "CLP")], (FeatureType.DATA, "merged"), 0),
        (
            [
                (FeatureType.MASK_TIMELESS, "RANDOM_UINT8"),
                (FeatureType.MASK_TIMELESS, "mask"),
                (FeatureType.MASK_TIMELESS, "LULC"),
            ],
            (FeatureType.MASK_TIMELESS, "merged_timeless"),
            -1,
        ),
    ],
)
def test_merge_features(axis: int, features_to_merge: List[FeatureSpec], feature: FeatureSpec, patch: EOPatch) -> None:
    patch = MergeFeatureTask(features_to_merge, feature, axis=axis)(patch)
    expected = np.concatenate([patch[f] for f in features_to_merge], axis=axis)

    assert_array_equal(patch[feature], expected)


@pytest.mark.parametrize(
    "features_to_zip, feature, function",
    [
        ([(FeatureType.DATA, "CLP"), (FeatureType.DATA, "bands")], (FeatureType.DATA, "ziped"), np.maximum),
        ([(FeatureType.DATA, "CLP"), (FeatureType.DATA, "bands")], (FeatureType.DATA, "ziped"), lambda a, b: a + b),
        (
            {FeatureType.MASK_TIMELESS: ["mask", "LULC", "RANDOM_UINT8"]},
            (FeatureType.MASK_TIMELESS, "ziped"),
            lambda a, b, c: a + b + c - 10,
        ),
    ],
)
def test_zip_features(
    features_to_zip: FeaturesSpecification, feature: FeatureSpec, function: Callable, patch: EOPatch
) -> None:
    expected = function(*[patch[f] for f in parse_features(features_to_zip)])
    patch = ZipFeatureTask(features_to_zip, feature, function)(patch)

    assert np.array_equal(patch[feature], expected)


def test_zip_features_fails(patch: EOPatch) -> None:
    with pytest.raises(NotImplementedError):
        ZipFeatureTask({FeatureType.DATA: ["CLP", "bands"]}, (FeatureType.DATA, "MERGED"))(patch)


@pytest.mark.parametrize(
    "input_features, output_features, map_function",
    [
        ({FeatureType.DATA: ["CLP", "bands"]}, {FeatureType.DATA: ["CLP_+3", "bands_+3"]}, lambda x: x + 3),
        ({FeatureType.MASK_TIMELESS: ["mask", "LULC"]}, {FeatureType.MASK_TIMELESS: ["mask2", "LULC2"]}, copy.deepcopy),
        ({FeatureType.DATA: ["CLP", "CLP_S2C"]}, {FeatureType.DATA: ["CLP_ceil", "CLP_S2C_ceil"]}, np.ceil),
    ],
)
def test_map_features(
    input_features: FeaturesSpecification,
    output_features: FeaturesSpecification,
    map_function: Callable,
    patch: EOPatch,
) -> None:
    original_patch = patch.copy(deep=True, features=input_features)
    mapped_patch = MapFeatureTask(input_features, output_features, map_function)(patch)

    for feature in parse_features(input_features):
        assert_array_equal(original_patch[feature], mapped_patch[feature]), "Task changed input data."

    for in_feature, out_feature in zip(parse_features(input_features), parse_features(output_features)):
        expected_output = map_function(mapped_patch[in_feature])
        assert_array_equal(mapped_patch[out_feature], expected_output)


@pytest.mark.parametrize("input_features, map_function", [({FeatureType.DATA: ["CLP", "bands"]}, lambda x: x + 3)])
def test_map_features_overwrite(input_features: FeaturesSpecification, map_function: Callable, patch: EOPatch) -> None:
    original_patch = patch.copy(deep=True, features=input_features)
    patch = MapFeatureTask(input_features, input_features, map_function)(patch)

    for in_feature in parse_features(input_features):
        expected_output = map_function(original_patch[in_feature])
        assert_array_equal(patch[in_feature], expected_output)


def test_map_features_fails(patch: EOPatch) -> None:
    with pytest.raises(NotImplementedError):
        MapFeatureTask({FeatureType.DATA: ["CLP", "NDVI"]}, {FeatureType.DATA: ["CLP2", "NDVI2"]})(patch)

    with pytest.raises(ValueError):
        MapFeatureTask({FeatureType.DATA: ["CLP", "NDVI"]}, {FeatureType.DATA: ["CLP2"]})


@pytest.mark.parametrize(
    "feature,  task_input",
    [
        ((FeatureType.DATA, "REFERENCE_SCENES"), {(FeatureType.DATA, "MOVED_BANDS"): [2, 4, 8]}),
        ((FeatureType.DATA, "REFERENCE_SCENES"), {(FeatureType.DATA, "MOVED_BANDS"): [2]}),
        ((FeatureType.DATA, "REFERENCE_SCENES"), {(FeatureType.DATA, "MOVED_BANDS"): (2,)}),
        ((FeatureType.DATA, "REFERENCE_SCENES"), {(FeatureType.DATA, "MOVED_BANDS"): 2}),
        (
            (FeatureType.DATA, "REFERENCE_SCENES"),
            {(FeatureType.DATA, "B01"): [0], (FeatureType.DATA, "B02"): [1], (FeatureType.DATA, "B02 & B03"): [1, 2]},
        ),
        ((FeatureType.DATA, "REFERENCE_SCENES"), {(FeatureType.DATA, "MOVED_BANDS"): []}),
    ],
)
def test_explode_bands(
    test_eopatch: EOPatch,
    feature: FeatureType,
    task_input: Dict[Tuple[FeatureType, str], Union[int, Iterable[int]]],
):
    move_bands = ExplodeBandsTask(feature, task_input)
    patch = move_bands(test_eopatch)
    assert all(new_feature in patch for new_feature in task_input)

    for new_feature, bands in task_input.items():
        if isinstance(bands, int):
            bands = [bands]
        assert_equal(patch[new_feature], test_eopatch[feature][..., bands])


def test_extract_bands(test_eopatch):
    bands = [2, 4, 8]
    move_bands = ExtractBandsTask((FeatureType.DATA, "REFERENCE_SCENES"), (FeatureType.DATA, "MOVED_BANDS"), bands)
    patch = move_bands(test_eopatch)
    assert np.array_equal(patch.data["MOVED_BANDS"], patch.data["REFERENCE_SCENES"][..., bands])

    old_value = patch.data["MOVED_BANDS"][0, 0, 0, 0]
    patch.data["MOVED_BANDS"][0, 0, 0, 0] += 1.0
    assert patch.data["REFERENCE_SCENES"][0, 0, 0, bands[0]] == old_value
    assert old_value + 1.0 == approx(patch.data["MOVED_BANDS"][0, 0, 0, 0])

    bands = [2, 4, 16]
    move_bands = ExtractBandsTask((FeatureType.DATA, "REFERENCE_SCENES"), (FeatureType.DATA, "MOVED_BANDS"), bands)
    with pytest.raises(ValueError):
        move_bands(patch)


def test_create_eopatch():
    data = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    bbox = BBox((5.60, 52.68, 5.75, 52.63), CRS.WGS84)

    patch = CreateEOPatchTask()(data={"bands": data}, bbox=bbox)
    assert np.array_equal(patch.data["bands"], data)


def test_kwargs():
    patch = EOPatch()
    shape = (3, 5, 5, 2)

    data1 = np.random.randint(0, 5, size=shape)
    data2 = np.random.randint(0, 5, size=shape)

    patch[(FeatureType.DATA, "D1")] = data1
    patch[(FeatureType.DATA, "D2")] = data2

    task0 = MapFeatureTask((FeatureType.DATA, "D1"), (FeatureType.DATA_TIMELESS, "NON_ZERO"), np.count_nonzero, axis=0)

    task1 = MapFeatureTask((FeatureType.DATA, "D1"), (FeatureType.DATA_TIMELESS, "MAX1"), np.max, axis=0)

    task2 = ZipFeatureTask({FeatureType.DATA: ["D1", "D2"]}, (FeatureType.DATA, "MAX2"), np.maximum, dtype=np.float32)

    patch = task0(patch)
    patch = task1(patch)
    patch = task2(patch)

    assert np.array_equal(patch[(FeatureType.DATA_TIMELESS, "NON_ZERO")], np.count_nonzero(data1, axis=0))
    assert np.array_equal(patch[(FeatureType.DATA_TIMELESS, "MAX1")], np.max(data1, axis=0))
    assert np.array_equal(patch[(FeatureType.DATA, "MAX2")], np.maximum(data1, data2))


def test_merge_eopatches(test_eopatch):
    task = MergeEOPatchesTask(time_dependent_op="max", timeless_op="concatenate")

    del test_eopatch.data["REFERENCE_SCENES"]  # wrong time dimension

    task.execute(test_eopatch, test_eopatch, test_eopatch)
