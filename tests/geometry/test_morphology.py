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
    return generate_eopatch([MASK_FEATURE, MASK_TIMELESS_FEATURE], config=config)


@pytest.mark.parametrize("invalid_input", [None, 0, "a"])
def test_erosion_value_error(invalid_input):
    with pytest.raises(ValueError):
        ErosionTask((FeatureType.MASK_TIMELESS, "LULC", "TEST"), disk_radius=invalid_input)


def test_erosion_full(test_eopatch):
    erosion_task = ErosionTask((FeatureType.MASK_TIMELESS, "LULC", "LULC_ERODED"), 1)
    eopatch = erosion_task.execute(test_eopatch)

    mask_after = eopatch.mask_timeless["LULC_ERODED"]

    assert_array_equal(np.unique(mask_after, return_counts=True)[1], [1942, 6950, 1069, 87, 52])


def test_erosion_partial(test_eopatch):
    # skip forest and artificial surface
    specific_labels = [0, 1, 3, 4]
    erosion_task = ErosionTask(
        mask_feature=(FeatureType.MASK_TIMELESS, "LULC", "LULC_ERODED"), disk_radius=1, erode_labels=specific_labels
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
            [1, 30, 161, 669, 1690, 3557, 6973, 12247, 19462, 30210],
            [7, 29, 112, 304, 639, 1336, 2465, 4012, 6096],
        ),
        (MorphologicalOperations.EROSION, MorphologicalStructFactory.get_disk(5), [74925, 72, 3], [14989, 11]),
        (MorphologicalOperations.OPENING, MorphologicalStructFactory.get_disk(5), [73137, 1800, 63], [14720, 280]),
        (MorphologicalOperations.CLOSING, MorphologicalStructFactory.get_disk(5), [1157, 73843], [501, 14499]),
        (
            MorphologicalOperations.MEDIAN,
            MorphologicalStructFactory.get_rectangle(5, 6),
            [16, 562, 6907, 24516, 28864, 12690, 1403, 42],
            [71, 1280, 4733, 5924, 2592, 382, 18],
        ),
        (
            MorphologicalOperations.OPENING,
            MorphologicalStructFactory.get_rectangle(5, 6),
            [47486, 24132, 2565, 497, 169, 96, 35, 20],
            [9929, 4446, 494, 53, 54, 16, 8],
        ),
        (
            MorphologicalOperations.DILATION,
            MorphologicalStructFactory.get_rectangle(5, 6),
            [2, 20, 184, 3888, 70906],
            [3, 32, 748, 14217],
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
