"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core import EOPatch, FeatureType
from eolearn.core.utils.testing import PatchGeneratorConfig, generate_eopatch
from eolearn.geometry import ErosionTask, MorphologicalFilterTask, MorphologicalOperations, MorphologicalStructFactory

CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
MASK_FEATURE = FeatureType.MASK, "mask"
MASK_TIMELESS_FEATURE = FeatureType.MASK_TIMELESS, "timeless_mask"
# ruff: noqa: NPY002


@pytest.fixture(name="patch")
def patch_fixture() -> EOPatch:
    config = PatchGeneratorConfig(max_integer_value=10, raster_shape=(50, 100), depth_range=(3, 4))
    patch = generate_eopatch([MASK_FEATURE, MASK_TIMELESS_FEATURE], config=config)
    patch[MASK_FEATURE] = patch[MASK_FEATURE].astype(np.uint8)
    patch[MASK_TIMELESS_FEATURE] = patch[MASK_TIMELESS_FEATURE] < 1
    patch[MASK_TIMELESS_FEATURE][10:20, 20:32] = 0
    patch[MASK_TIMELESS_FEATURE][30:, 50:] = 1

    return patch


@pytest.mark.parametrize("invalid_input", [None, 0, "a"])
def test_erosion_value_error(invalid_input):
    with pytest.raises(ValueError):
        ErosionTask((FeatureType.MASK_TIMELESS, "LULC", "TEST"), disk_radius=invalid_input)


def test_erosion_full(test_eopatch):
    erosion_task = ErosionTask((FeatureType.MASK_TIMELESS, "LULC", "LULC_ERODED"), disk_radius=3)
    eopatch = erosion_task.execute(test_eopatch)

    mask_after = eopatch.mask_timeless["LULC_ERODED"]

    assert_array_equal(np.unique(mask_after, return_counts=True)[1], [1942, 6950, 1069, 87, 52])


def test_erosion_partial(test_eopatch):
    # skip forest and artificial surface
    specific_labels = [0, 1, 3, 4]
    erosion_task = ErosionTask(
        mask_feature=(FeatureType.MASK_TIMELESS, "LULC", "LULC_ERODED"), disk_radius=3, erode_labels=specific_labels
    )
    eopatch = erosion_task.execute(test_eopatch)

    mask_after = eopatch.mask_timeless["LULC_ERODED"]

    assert_array_equal(np.unique(mask_after, return_counts=True)[1], [1145, 7601, 1069, 87, 198])


@pytest.mark.parametrize(
    ("morph_operation", "struct_element", "mask_counts", "mask_timeless_counts"),
    [
        (
            MorphologicalOperations.DILATION,
            None,
            [6, 34, 172, 768, 2491, 7405, 19212, 44912],
            [4882, 10118],
        ),
        (
            MorphologicalOperations.EROSION,
            MorphologicalStructFactory.get_disk(4),
            [54555, 15639, 3859, 770, 153, 19, 5],
            [12391, 2609],
        ),
        (
            MorphologicalOperations.OPENING,
            MorphologicalStructFactory.get_disk(3),
            [8850, 13652, 16866, 14632, 11121, 6315, 2670, 761, 133],
            [11981, 3019],
        ),
        (MorphologicalOperations.CLOSING, MorphologicalStructFactory.get_disk(11), [770, 74230], [661, 14339]),
        (
            MorphologicalOperations.OPENING,
            MorphologicalStructFactory.get_rectangle(3, 3),
            [15026, 23899, 20363, 9961, 4328, 1128, 280, 15],
            [12000, 3000],
        ),
        (
            MorphologicalOperations.DILATION,
            MorphologicalStructFactory.get_rectangle(5, 6),
            [2, 19, 198, 3929, 70852],
            [803, 14197],
        ),
    ],
)
def test_morphological_filter(patch, morph_operation, struct_element, mask_counts, mask_timeless_counts):
    task = MorphologicalFilterTask(
        [MASK_FEATURE, MASK_TIMELESS_FEATURE], morph_operation=morph_operation, struct_elem=struct_element
    )
    task.execute(patch)

    assert patch[MASK_FEATURE].shape == (5, 50, 100, 3)
    assert patch[MASK_TIMELESS_FEATURE].shape == (50, 100, 3)
    assert_array_equal(np.unique(patch[MASK_FEATURE], return_counts=True)[1], mask_counts)
    assert_array_equal(np.unique(patch[MASK_TIMELESS_FEATURE], return_counts=True)[1], mask_timeless_counts)
