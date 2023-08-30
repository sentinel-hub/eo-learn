"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import ErosionTask, MorphologicalFilterTask, MorphologicalOperations, MorphologicalStructFactory

CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
MASK_FEATURE = FeatureType.MASK, "mask"
MASK_TIMELESS_FEATURE = FeatureType.MASK_TIMELESS, "timeless_mask"
# ruff: noqa: NPY002


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


@pytest.mark.parametrize("morph_operation", MorphologicalOperations)
@pytest.mark.parametrize(
    "struct_element", [None, MorphologicalStructFactory.get_disk(5), MorphologicalStructFactory.get_rectangle(5, 6)]
)
def test_morphological_filter(morph_operation, struct_element):
    eopatch = EOPatch(bbox=BBox((0, 0, 1, 1), CRS(3857)), timestamps=["2015-7-7"] * 10)
    eopatch[MASK_FEATURE] = np.random.randint(20, size=(10, 100, 100, 3), dtype=np.uint8)
    eopatch[MASK_TIMELESS_FEATURE] = np.random.randint(20, 50, size=(100, 100, 5), dtype=np.uint8)

    task = MorphologicalFilterTask(
        [MASK_FEATURE, MASK_TIMELESS_FEATURE], morph_operation=morph_operation, struct_elem=struct_element
    )
    task.execute(eopatch)

    assert eopatch[MASK_FEATURE].shape == (10, 100, 100, 3)
    assert eopatch[MASK_TIMELESS_FEATURE].shape == (100, 100, 5)
