"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

import copy
import pickle
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union

import numpy as np
import pytest
from fs.base import FS
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from numpy.testing import assert_array_equal

from sentinelhub import CRS, BBox

from eolearn.core import (
    AddFeatureTask,
    CopyTask,
    CreateEOPatchTask,
    DeepCopyTask,
    DuplicateFeatureTask,
    EOPatch,
    EOTask,
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
from eolearn.core.utils.testing import PatchGeneratorConfig, assert_feature_data_equal, generate_eopatch

DUMMY_BBOX = BBox((0, 0, 1, 1), CRS(3857))
# ruff: noqa: NPY002


@pytest.fixture(name="patch")
def patch_fixture() -> EOPatch:
    patch = generate_eopatch(
        {
            FeatureType.DATA: ["bands", "CLP"],
            FeatureType.MASK: ["CLM"],
            FeatureType.MASK_TIMELESS: ["mask", "LULC", "RANDOM_UINT8"],
            FeatureType.SCALAR: ["values", "CLOUD_COVERAGE"],
            FeatureType.META_INFO: ["something"],
        }
    )
    patch.data["CLP_S2C"] = np.zeros_like(patch.data["CLP"])
    return patch


@pytest.fixture(name="eopatch_to_explode")
def eopatch_to_explode_fixture() -> EOPatch:
    return generate_eopatch((FeatureType.DATA, "bands"), config=PatchGeneratorConfig(depth_range=(8, 9)))


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
        [(FeatureType.TIMESTAMPS, None), (FeatureType.SCALAR, "values")],
    ],
)
@pytest.mark.parametrize("task", [DeepCopyTask, CopyTask])
def test_partial_copy(features: List[FeatureSpec], task: Type[CopyTask], patch: EOPatch) -> None:
    patch_copy = task(features=features)(patch)

    assert set(patch_copy.get_features()) == {(FeatureType.BBOX, None), *features}
    for feature in features:
        assert_feature_data_equal(patch[feature], patch_copy[feature])


def test_load_task(test_eopatch_path: str) -> None:
    full_patch = LoadTask(test_eopatch_path)(eopatch_folder=".")
    assert len(full_patch.get_features()) == 30

    partial_load = LoadTask(test_eopatch_path, features=[FeatureType.BBOX, FeatureType.MASK_TIMELESS])
    partial_patch = partial_load.execute(eopatch_folder=".")
    assert FeatureType.BBOX in partial_patch
    assert FeatureType.TIMESTAMPS not in partial_patch

    load_more = LoadTask(test_eopatch_path, features=[FeatureType.TIMESTAMPS])
    upgraded_partial_patch = load_more.execute(partial_patch, eopatch_folder=".")
    assert FeatureType.BBOX in upgraded_partial_patch
    assert FeatureType.TIMESTAMPS in upgraded_partial_patch
    assert FeatureType.DATA not in upgraded_partial_patch


@pytest.mark.parametrize("filesystem", [OSFS("."), S3FS("s3://fake-bucket/"), TempFS()])
@pytest.mark.parametrize("task_class", [LoadTask, SaveTask])
def test_io_task_pickling(filesystem: FS, task_class: Type[EOTask]) -> None:
    task = task_class("/", filesystem=filesystem)

    pickled_task = pickle.dumps(task)
    unpickled_task = pickle.loads(pickled_task)
    assert isinstance(unpickled_task, task_class)


@pytest.mark.parametrize(
    ("feature", "feature_data"),
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

    assert_feature_data_equal(patch[feature], feature_data)


def test_rename_feature(patch: EOPatch) -> None:
    f_type, f_name, f_new_name = FeatureType.DATA, "bands", "new_bands"
    assert (f_type, f_new_name) not in patch
    patch_copy = copy.deepcopy(patch)

    patch = RenameFeatureTask((f_type, f_name, f_new_name))(patch)
    assert_array_equal(patch[(f_type, f_new_name)], patch_copy[(f_type, f_name)])
    assert (f_type, f_name) not in patch, "Feature was not removed from patch. "


@pytest.mark.parametrize(
    "features", [(FeatureType.DATA, "bands"), [FeatureType.TIMESTAMPS, FeatureType.DATA, (FeatureType.MASK, "CLM")]]
)
def test_remove_feature(features: FeaturesSpecification, patch: EOPatch) -> None:
    original_patch = copy.deepcopy(patch)
    features_to_remove = parse_features(features, patch)
    assert all(feature in patch for feature in features_to_remove)

    patch = RemoveFeatureTask(features)(patch)
    for feature in original_patch.get_features():
        assert (feature not in patch) if feature in features_to_remove else (feature in patch)


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
    ("init_val", "shape", "feature_spec"),
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

    assert all(np.array_equal(patch[features], expected_data) for features in parse_features(feature_spec))


@pytest.mark.parametrize(
    ("init_val", "shape", "feature_spec"),
    [
        (3, (FeatureType.DATA, "bands"), {FeatureType.MASK: ["F1", "F2", "F3"]}),
    ],
)
def test_initialize_feature_with_spec(
    init_val: float, shape: FeatureSpec, feature_spec: FeaturesSpecification, patch: EOPatch
) -> None:
    expected_data = init_val * np.ones(patch[shape].shape)

    patch = InitializeFeatureTask(feature_spec, shape=shape, init_value=init_val)(patch)
    assert all(np.array_equal(patch[features], expected_data) for features in parse_features(feature_spec))


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

        assert_feature_data_equal(patch[feat], patch_dst[feat])


@pytest.mark.parametrize(
    ("features_to_merge", "feature", "axis"),
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
    ("input_features", "output_feature", "zip_function", "kwargs"),
    [
        ({FeatureType.DATA: ["CLP", "bands"]}, (FeatureType.DATA, "ziped"), np.maximum, {}),
        ({FeatureType.DATA: ["CLP", "bands"]}, (FeatureType.DATA, "ziped"), lambda a, b: a + b, {}),
        (
            {FeatureType.MASK_TIMELESS: ["mask", "LULC", "RANDOM_UINT8"]},
            (FeatureType.MASK_TIMELESS, "ziped"),
            lambda a, b, c: a + b + c - 10,
            {},
        ),
        (
            {FeatureType.DATA: ["bands", "CLP"]},
            (FeatureType.DATA, "feat_max"),
            np.floor_divide,
            {"dtype": np.float32},
        ),
        ({FeatureType.DATA: ["CLP", "bands"]}, (FeatureType.DATA, "feat_sum_+3"), lambda x, y, a: x + y + a, {"a": 3}),
    ],
)
def test_zip_features(
    input_features: FeaturesSpecification,
    output_feature: FeatureSpec,
    zip_function: Callable,
    kwargs: Dict[str, Any],
    patch: EOPatch,
) -> None:
    expected = zip_function(*[patch[feat] for feat in parse_features(input_features)], **kwargs)
    patch = ZipFeatureTask(input_features, output_feature, zip_function, **kwargs)(patch)

    assert_array_equal(patch[output_feature], expected)


def test_zip_features_fails(patch: EOPatch) -> None:
    with pytest.raises(NotImplementedError):
        ZipFeatureTask({FeatureType.DATA: ["CLP", "bands"]}, (FeatureType.DATA, "MERGED"))(patch)


@pytest.mark.parametrize(
    ("input_features", "output_features", "map_function", "kwargs"),
    [
        ({FeatureType.DATA: ["CLP", "bands"]}, {FeatureType.DATA: ["CLP_+3", "bands_+3"]}, lambda x: x + 3, {}),
        (
            {FeatureType.MASK_TIMELESS: ["mask", "LULC"]},
            {FeatureType.MASK_TIMELESS: ["mask2", "LULC2"]},
            copy.deepcopy,
            {},
        ),
        ({FeatureType.DATA: ["CLP", "CLP_S2C"]}, {FeatureType.DATA: ["CLP_ceil", "CLP_S2C_ceil"]}, np.ceil, {}),
        (
            {FeatureType.DATA: ["CLP", "bands"]},
            {FeatureType.DATA_TIMELESS: ["CLP_max", "bands_max"]},
            np.max,
            {"axis": -1},
        ),
        (
            {FeatureType.DATA: ["CLP", "bands"]},
            {FeatureType.DATA: ["CLP_+3", "bands_+3"]},
            lambda x, a: x + a,
            {"a": 3},
        ),
    ],
)
def test_map_features(
    input_features: FeaturesSpecification,
    output_features: FeaturesSpecification,
    map_function: Callable,
    kwargs: Dict[str, Any],
    patch: EOPatch,
) -> None:
    original_patch = patch.copy(deep=True, features=input_features)
    mapped_patch = MapFeatureTask(input_features, output_features, map_function, **kwargs)(patch)

    for feature in parse_features(input_features):
        assert_array_equal(original_patch[feature], mapped_patch[feature]), "Task changed input data."

    for in_feature, out_feature in zip(parse_features(input_features), parse_features(output_features)):
        expected_output = map_function(mapped_patch[in_feature], **kwargs)
        assert_array_equal(mapped_patch[out_feature], expected_output)


@pytest.mark.parametrize(("input_features", "map_function"), [({FeatureType.DATA: ["CLP", "bands"]}, lambda x: x + 3)])
def test_map_features_overwrite(input_features: FeaturesSpecification, map_function: Callable, patch: EOPatch) -> None:
    original_patch = patch.copy(deep=True, features=input_features)
    patch = MapFeatureTask(input_features, input_features, map_function)(patch)

    for in_feature in parse_features(input_features):
        expected_output = map_function(original_patch[in_feature])
        assert_array_equal(patch[in_feature], expected_output)


def test_map_features_fails(patch: EOPatch) -> None:
    with pytest.raises(NotImplementedError):
        MapFeatureTask((FeatureType.DATA, "CLP"), (FeatureType.DATA, "CLP2"))(patch)

    with pytest.raises(ValueError):
        MapFeatureTask({FeatureType.DATA: ["CLP", "NDVI"]}, {FeatureType.DATA: ["CLP2"]}, map_function=lambda x: x)


@pytest.mark.parametrize(
    ("input_feature", "kwargs"),
    [
        ((FeatureType.DATA, "bands"), {"axis": -1, "name": "fun_name", "bands": [4, 3, 2]}),
    ],
)
def test_map_kwargs_passing(input_feature: FeatureSpec, kwargs: Dict[str, Any], patch: EOPatch) -> None:
    def kwargs_map(_, *, some=3, **kwargs) -> tuple:
        return some, kwargs

    mapped_patch = MapFeatureTask(input_feature, (FeatureType.META_INFO, "kwargs"), kwargs_map, **kwargs)(patch)

    expected_output = kwargs_map(mapped_patch[input_feature], **kwargs)
    assert mapped_patch[(FeatureType.META_INFO, "kwargs")] == expected_output


@pytest.mark.parametrize(
    ("feature", "task_input"),
    [
        ((FeatureType.DATA, "bands"), {(FeatureType.DATA, "EXPLODED_BANDS"): [2, 4, 6]}),
        ((FeatureType.DATA, "bands"), {(FeatureType.DATA, "EXPLODED_BANDS"): [2]}),
        ((FeatureType.DATA, "bands"), {(FeatureType.DATA, "EXPLODED_BANDS"): (2,)}),
        ((FeatureType.DATA, "bands"), {(FeatureType.DATA, "EXPLODED_BANDS"): 2}),
        (
            (FeatureType.DATA, "bands"),
            {(FeatureType.DATA, "B01"): [0], (FeatureType.DATA, "B02"): [1], (FeatureType.DATA, "B02 & B03"): [1, 2]},
        ),
        ((FeatureType.DATA, "bands"), {(FeatureType.DATA, "EXPLODED_BANDS"): []}),
    ],
)
def test_explode_bands(
    eopatch_to_explode: EOPatch,
    feature: Tuple[FeatureType, str],
    task_input: Dict[Tuple[FeatureType, str], Union[int, Iterable[int]]],
) -> None:
    patch = ExplodeBandsTask(feature, task_input)(eopatch_to_explode)
    assert all(new_feature in patch for new_feature in task_input)

    for new_feature, bands in task_input.items():
        if isinstance(bands, int):
            bands = [bands]
        assert_array_equal(patch[new_feature], patch[feature][..., bands])


def test_extract_bands(eopatch_to_explode: EOPatch) -> None:
    bands = [2, 4, 6]
    patch = ExtractBandsTask((FeatureType.DATA, "bands"), (FeatureType.DATA, "EXTRACTED_BANDS"), bands)(
        eopatch_to_explode
    )
    assert_array_equal(patch.data["EXTRACTED_BANDS"], patch.data["bands"][..., bands])

    patch.data["EXTRACTED_BANDS"][0, 0, 0, 0] += 1
    assert patch.data["EXTRACTED_BANDS"][0, 0, 0, 0] != patch.data["bands"][0, 0, 0, bands[0]]


def test_extract_bands_fails(eopatch_to_explode: EOPatch) -> None:
    with pytest.raises(ValueError):
        # fails because band 16 does not exist
        ExtractBandsTask((FeatureType.DATA, "bands"), (FeatureType.DATA, "EXTRACTED_BANDS"), [2, 4, 16])(
            eopatch_to_explode
        )


@pytest.mark.parametrize(
    "features",
    [
        {"bbox": DUMMY_BBOX},
        {"data": {"bands": np.arange(0, 32).reshape(1, 4, 4, 2)}, "bbox": DUMMY_BBOX},
        {"data": {"bands": np.arange(0, 32).reshape(1, 4, 4, 2), "CLP": np.ones((1, 4, 4, 2))}, "bbox": DUMMY_BBOX},
    ],
)
def test_create_eopatch(features: Dict[str, Any]) -> None:
    assert CreateEOPatchTask()(**features) == EOPatch(**features)


def test_merge_eopatches() -> None:
    dummy_timestamps = [datetime(2017, 1, 14, 10, 13, 46), datetime(2017, 2, 10, 10, 1, 32)]

    patch1 = EOPatch(
        data={"bands": np.zeros((2, 4, 4, 3), dtype=np.float32)},
        mask_timeless={"LULC": np.arange(0, 16).reshape(4, 4, 1)},
        bbox=DUMMY_BBOX,
        timestamps=dummy_timestamps,
    )
    patch2 = EOPatch(
        data={"bands": np.ones((2, 4, 4, 3), dtype=np.float32)},
        mask_timeless={"LULC": np.arange(16, 32).reshape(4, 4, 1)},
        bbox=DUMMY_BBOX,
        timestamps=dummy_timestamps,
    )

    merged_patch = MergeEOPatchesTask(time_dependent_op="max", timeless_op="concatenate")(patch1, patch2)

    expected_patch = EOPatch(
        data={"bands": np.ones((2, 4, 4, 3), dtype=np.float32)},
        mask_timeless={"LULC": np.arange(0, 32).reshape((16, 2), order="F").reshape(4, 4, 2)},
        bbox=DUMMY_BBOX,
        timestamps=dummy_timestamps,
    )

    assert merged_patch == expected_patch


def test_merge_eopatches_fails() -> None:
    with pytest.raises(ValueError):
        MergeEOPatchesTask(time_dependent_op="max", timeless_op="concatenate")()
