"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import copy
import math
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import Point, Polygon

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.utils.testing import PatchGeneratorConfig, generate_eopatch
from eolearn.ml_tools import BlockSamplingTask, FractionSamplingTask, GridSamplingTask, sample_by_values
from eolearn.ml_tools.sampling import expand_to_grids, get_mask_of_samples, random_point_in_triangle


@pytest.mark.parametrize(
    ("triangle", "expected_points"),
    [
        (
            Polygon([[-10, -12], [5, 10], [15, 4]]),
            [Point(7.057238842063997, 5.037835974428286), Point(10.360934995972169, 4.508212300648985)],
        ),
        (
            Polygon([[110, 112], [117, 102], [115, 104]]),
            [Point(115.38602941857646, 103.97472742532676), Point(115.19386890346114, 104.02631432574218)],
        ),
        (
            Polygon([[0, 0], [5, 12], [15, 4]]),
            [Point(8.259761655074744, 7.46815417512301), Point(11.094879093316592, 5.949786169595648)],
        ),
    ],
)
def test_random_point_in_triangle_generator(triangle: Polygon, expected_points: List[Point]) -> None:
    generator = np.random.default_rng(seed=42)
    points = [random_point_in_triangle(triangle, generator) for _ in range(2)]
    assert all(point == expected for point, expected in zip(points, expected_points))


@pytest.mark.parametrize(
    "triangle",
    [
        (Polygon([[-10, -12], [5, 10], [15, 4]])),
        (Polygon([[110, 112], [117, 102], [115, 104]])),
        (Polygon([[0, 0], [5, 12], [15, 4]])),
    ],
)
def test_random_point_in_triangle_interior(triangle: Polygon) -> None:
    points = [random_point_in_triangle(triangle) for _ in range(1000)]
    assert all(triangle.contains(point) for point in points)


@pytest.fixture(name="small_image")
def small_image_fixture() -> np.ndarray:
    image_size = 100, 75
    image = np.zeros(image_size, dtype=np.uint8)
    image[40:60, 40:60] = 1
    image[50:80, 55:70] = 2
    return image


@pytest.mark.parametrize(
    ("image", "n_samples"),
    [
        (np.ones((100,)), {1: 100}),
        (np.ones((100, 100, 3)), {1: 100}),
        (np.ones((100, 100)), {2: 100}),
        (np.ones((100, 100)), {1: 10001}),
    ],
)
def test_sample_by_values_errors(image: np.ndarray, n_samples: Dict[int, int]) -> None:
    rng = np.random.default_rng()
    with pytest.raises(ValueError):
        sample_by_values(image, n_samples, rng=rng)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ("n_samples", "replace"),
    [
        ({0: 100, 1: 200, 2: 30}, False),
        ({1: 200}, False),
        ({0: 100, 2: 30000}, True),
    ],
)
def test_sample_by_values(small_image: np.ndarray, seed: int, n_samples: Dict[int, int], replace: bool) -> None:
    rng = np.random.default_rng(seed)
    rows, cols = sample_by_values(small_image, n_samples, rng=rng, replace=replace)
    labels = small_image[rows, cols]

    # test number of samples per value is correct
    for value, amount in n_samples.items():
        assert np.sum(labels == value) == amount, f"Incorrect amount of samples for value {value}"


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (np.array([1, 1, 2, 3, 4]), np.array([2, 3, 1, 1, 4])),
    ],
)
@pytest.mark.parametrize("sample_size", [(1, 1), (2, 3), (10, 11)])
def test_expand_to_grids(rows: np.ndarray, columns: np.ndarray, sample_size: Tuple[int, int]) -> None:
    row_grid, column_grid = expand_to_grids(rows, columns, sample_size=sample_size)
    expected_shape = sample_size[0] * rows.size, sample_size[1]

    assert row_grid.shape == expected_shape
    assert column_grid.shape == expected_shape


@pytest.mark.parametrize(
    "n_samples",
    [
        ({0: 10000, 1: 200, 2: 30}),
        ({1: 200}),
        ({0: 100, 2: 30000}),
    ],
)
def test_get_mask_of_samples(small_image: np.ndarray, n_samples: Dict[int, int]) -> None:
    row_grid, column_grid = sample_by_values(small_image, n_samples, replace=True)
    image_shape = small_image.shape
    result = get_mask_of_samples(image_shape, row_grid, column_grid)

    for key, val in n_samples.items():
        assert np.sum(result[np.nonzero(small_image == key)]) == val


@pytest.fixture(name="eopatch")
def eopatch_fixture(small_image: np.ndarray) -> EOPatch:
    config = PatchGeneratorConfig(raster_shape=small_image.shape, depth_range=(5, 6), num_timestamps=10)
    patch = generate_eopatch([(FeatureType.DATA, "bands")], config=config)
    patch.mask_timeless["raster"] = small_image.reshape((*small_image.shape, 1))
    return patch


SAMPLING_MASK = FeatureType.MASK_TIMELESS, "sampling_mask"


@pytest.fixture(name="block_task")
def block_task_fixture(request) -> EOTask:
    """Constructed for indirect=True testing."""
    return BlockSamplingTask(
        [(FeatureType.DATA, "bands", "SAMPLED_DATA"), (FeatureType.MASK_TIMELESS, "raster", "SAMPLED_LABELS")],
        amount=request.param,
        mask_of_samples=SAMPLING_MASK,
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("block_task", [100, 5231, 0.4, 0], indirect=True)
def test_object_sampling_task_mask(
    eopatch: EOPatch, small_image: np.ndarray, seed: int, block_task: BlockSamplingTask
) -> None:
    t, h, w, d = eopatch.data["bands"].shape
    dr = eopatch.mask_timeless["raster"].shape[2]
    amount = block_task.amount

    block_task.execute(eopatch, seed=seed)
    expected_amount = amount if isinstance(amount, int) else round(np.prod(small_image.shape) * amount)

    assert eopatch.data["SAMPLED_DATA"].shape == (t, expected_amount, 1, d)
    assert eopatch.mask_timeless["SAMPLED_LABELS"].shape == (expected_amount, 1, dr)
    assert eopatch.mask_timeless["sampling_mask"].shape == (h, w, 1)

    sampled_uniques, sampled_counts = np.unique(eopatch.data["SAMPLED_DATA"], return_counts=True)
    masked = eopatch.mask_timeless["sampling_mask"].squeeze(axis=2) == 1
    masked_uniques, masked_counts = np.unique(eopatch.data["bands"][:, masked, :], return_counts=True)

    assert_array_equal(sampled_uniques, masked_uniques)
    assert_array_equal(sampled_counts, masked_counts)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("block_task", [100, 0.4], indirect=True)
def test_object_sampling_reproducibility(eopatch: EOPatch, seed: int, block_task: BlockSamplingTask) -> None:
    eopatch1 = block_task.execute(copy.copy(eopatch), seed=seed)
    eopatch2 = block_task.execute(copy.copy(eopatch), seed=seed)
    eopatch3 = block_task.execute(copy.copy(eopatch), seed=(seed + 1))

    # assert features, labels and sampled rows and cols are added to eopatch
    assert eopatch1 == eopatch2
    assert (eopatch1.mask_timeless["sampling_mask"] != eopatch3.mask_timeless["sampling_mask"]).any()


@pytest.mark.parametrize(
    ("fraction", "replace"),
    [(2, False), (-0.5, True), ({1: 0.5, 3: 0.4, 5: 1.2}, False), ({1: 0.5, 3: -0.4, 5: 1.2}, True), ((1, 0.4), True)],
)
def test_fraction_sampling_errors(fraction: Union[float, Dict[int, float]], replace: bool) -> None:
    with pytest.raises(ValueError):
        FractionSamplingTask(
            [(FeatureType.DATA, "NDVI", "NDVI_SAMPLED")],
            (FeatureType.MASK_TIMELESS, "LULC"),
            fraction=fraction,
            replace=replace,
        )


@pytest.fixture(name="fraction_task")
def fraction_task_fixture(request) -> EOTask:
    """Constructed for indirect=True testing."""
    return FractionSamplingTask(
        [(FeatureType.DATA, "NDVI", "NDVI_SAMPLED"), (FeatureType.MASK_TIMELESS, "LULC", "LULC_SAMPLED")],
        (FeatureType.MASK_TIMELESS, "LULC"),
        mask_of_samples=(FeatureType.MASK_TIMELESS, "sampling_mask"),
        fraction=request.param[0],
        exclude_values=request.param[1],
    )


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "fraction_task",
    [  # fraction, exclude_values
        [0.2, None],
        [0.4, [0, 1]],
        [{1: 0.1, 3: 0.4}, None],
        [{2: 0.1, 3: 1, 5: 1}, []],
        [{2: 0.1, 3: 1, 5: 1}, [0, 2]],
    ],
    indirect=True,
)
def test_fraction_sampling_mask(fraction_task: FractionSamplingTask, seed: int, test_eopatch: EOPatch) -> None:
    task = fraction_task
    eopatch = task(test_eopatch, seed=seed)

    # Test amount
    assert eopatch.mask_timeless["LULC_SAMPLED"].shape == eopatch.data["NDVI_SAMPLED"].shape[1:]

    # Test mask
    sampled_uniques, sampled_counts = np.unique(eopatch.data["NDVI_SAMPLED"], return_counts=True)
    masked = eopatch.mask_timeless["sampling_mask"].squeeze(axis=2) == 1
    masked_uniques, masked_counts = np.unique(eopatch.data["NDVI"][:, masked], return_counts=True)
    assert_array_equal(sampled_uniques, masked_uniques)
    assert_array_equal(sampled_counts, masked_counts)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("fraction_task", [[0.2, None], [0.4, [0, 1]]], indirect=True)
def test_fraction_sampling_input_fraction(
    fraction_task: FractionSamplingTask, seed: int, test_eopatch: EOPatch
) -> None:
    eopatch = fraction_task(test_eopatch, seed=seed)

    # Test balance and inclusion/exclusion
    full_values, full_counts = np.unique(eopatch[(FeatureType.MASK_TIMELESS, "LULC")], return_counts=True)
    sample_values, sample_counts = np.unique(eopatch[(FeatureType.MASK_TIMELESS, "LULC_SAMPLED")], return_counts=True)
    full = dict(zip(full_values, full_counts))
    samples = dict(zip(sample_values, sample_counts))

    exclude = fraction_task.exclude_values or []  # get rid of pesky None
    assert set(exclude).isdisjoint(set(sample_values))

    for val, count in full.items():
        if val not in exclude:
            assert samples[val] == pytest.approx(count * fraction_task.fraction, abs=1)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "fraction_task",
    [
        [{1: 0.1, 3: 0.4}, None],
        [{2: 0.1, 3: 1, 5: 1}, []],
        [{2: 0.1, 3: 1, 5: 1}, [0, 2]],
    ],
    indirect=True,
)
def test_fraction_sampling_input_dict(fraction_task: FractionSamplingTask, seed: int, test_eopatch: EOPatch) -> None:
    eopatch = fraction_task(test_eopatch, seed=seed)

    full_values, full_counts = np.unique(eopatch[(FeatureType.MASK_TIMELESS, "LULC")], return_counts=True)
    sample_values, sample_counts = np.unique(eopatch[(FeatureType.MASK_TIMELESS, "LULC_SAMPLED")], return_counts=True)
    full = dict(zip(full_values, full_counts))
    samples = dict(zip(sample_values, sample_counts))

    exclude = fraction_task.exclude_values or []  # get rid of pesky None
    assert set(exclude).isdisjoint(set(sample_values))
    assert set(sample_values).issubset(set(fraction_task.fraction))
    assert all(count == pytest.approx(full[val] * fraction_task.fraction[val], abs=1) for val, count in samples.items())


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("fraction_task", [[0.3, None], [{1: 0.1, 4: 0.3}, []]], indirect=True)
def test_fraction_sampling_reproducibility(
    fraction_task: FractionSamplingTask, seed: int, test_eopatch: EOPatch
) -> None:
    eopatch1 = fraction_task.execute(copy.copy(test_eopatch), seed=seed)
    eopatch2 = fraction_task.execute(copy.copy(test_eopatch), seed=seed)
    eopatch3 = fraction_task.execute(copy.copy(test_eopatch), seed=(seed + 1))

    assert eopatch1 == eopatch2
    assert (eopatch1.mask_timeless["sampling_mask"] != eopatch3.mask_timeless["sampling_mask"]).any()


SAMPLE_MASK = FeatureType.MASK_TIMELESS, "SAMPLE_MASK"


@pytest.fixture(name="grid_task")
def grid_task_fixture(request) -> EOTask:
    """Constructed for indirect=True testing."""
    return GridSamplingTask(
        features_to_sample=[
            (FeatureType.DATA, "BANDS-S2-L1C", "SAMPLED_BANDS"),
            (FeatureType.MASK_TIMELESS, "LULC", "LULC"),
        ],
        sample_size=request.param[0],
        stride=request.param[1],
        mask_of_samples=SAMPLE_MASK,
    )


@pytest.mark.parametrize("grid_task", [[(1, 1), (1, 1)], [(2, 3), (5, 3)], [(6, 5), (3, 3)]], indirect=True)
def test_grid_sampling_task(test_eopatch: EOPatch, grid_task: GridSamplingTask) -> None:
    # expected_shape calculated
    sample_size = grid_task.sample_size
    expected_shape = list(test_eopatch.data["BANDS-S2-L1C"].shape)
    expected_shape[1] = (
        math.ceil((expected_shape[1] - sample_size[0] + 1) / grid_task.stride[0])
        * math.ceil((expected_shape[2] - sample_size[1] + 1) / grid_task.stride[1])
        * sample_size[0]
    )
    expected_shape[2] = sample_size[1]
    expected_shape = tuple(expected_shape)

    eopatch = grid_task.execute(test_eopatch)

    assert eopatch.data["SAMPLED_BANDS"].shape == expected_shape
    height, width = expected_shape[1:3]
    assert np.sum(eopatch[SAMPLE_MASK]) == height * width


@pytest.mark.parametrize("grid_task", [[(1, 1), (1, 1)], [(2, 3), (5, 3)]], indirect=True)
def test_grid_sampling_task_reproducibility(test_eopatch: EOPatch, grid_task: GridSamplingTask) -> None:
    eopatch1 = grid_task.execute(copy.copy(test_eopatch))
    eopatch2 = grid_task.execute(copy.copy(test_eopatch))

    assert eopatch1 == eopatch2
    assert eopatch1 is not eopatch2
