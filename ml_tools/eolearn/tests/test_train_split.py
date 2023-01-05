"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from typing import Any, Optional

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core import EOPatch, FeatureType
from eolearn.ml_tools.train_test_split import TrainTestSplitTask, TrainTestSplitType

INPUT_FEATURE = (FeatureType.MASK_TIMELESS, "TEST")
OUTPUT_FEATURE = (FeatureType.MASK_TIMELESS, "TEST_TRAIN_MASK")


@pytest.mark.parametrize(
    "bad_arg, bad_kwargs",
    [
        (None, {}),
        (1.5, {}),
        ([0.5, 0.3], {}),
        ([0.5], {"split_type": None}),
        ([0.5, 0.7], {"split_type": "nonsense"}),
    ],
)
def test_bad_args(bad_arg: Any, bad_kwargs: Any) -> None:
    with pytest.raises(ValueError):
        TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bad_arg, **bad_kwargs)


@pytest.fixture(name="eopatch1")
def eopatch1_fixture(seed: Optional[int] = None) -> EOPatch:
    eopatch = EOPatch()
    rng = np.random.default_rng(seed)
    eopatch[INPUT_FEATURE] = rng.integers(0, 10, size=(1000, 1000, 3))

    return eopatch


@pytest.fixture(name="eopatch2")
def eopatch2_fixture(seed: Optional[int] = None) -> EOPatch:
    eopatch = EOPatch()
    rng = np.random.default_rng(seed)
    eopatch[INPUT_FEATURE] = rng.integers(0, 10, size=(1000, 1000, 3), dtype=int)

    return eopatch


def test_train_split(eopatch1: EOPatch) -> None:
    """test hardcode some values in input feature and checks if hardcode values are split together"""
    indices = [(0, 2, 0, 2), (0, 2, 2, 4), (2, 4, 0, 2), (2, 4, 2, 4), (0, 4, 4, 8), (4, 8, 0, 4), (4, 8, 4, 8)]
    for index, (i_1, i_2, j_1, j_2) in enumerate(indices, 1):
        eopatch1[INPUT_FEATURE][i_1:i_2, j_1:j_2, :] = index * 11

    bins = [0.2, 0.5, 0.8]
    expected_unique = set(range(1, len(bins) + 2))

    split_task = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type=TrainTestSplitType.PER_CLASS)
    eopatch1 = split_task(eopatch1, seed=1)
    assert set(np.unique(eopatch1[OUTPUT_FEATURE])) <= expected_unique

    result = np.copy(eopatch1[OUTPUT_FEATURE])
    unique = (np.unique(result[i_1:i_2, j_1:j_2, :], return_counts=True) for i_1, i_2, j_1, j_2 in indices)
    expected = [(i_2 - i_1) * (j_2 - j_1) * eopatch1[OUTPUT_FEATURE].shape[-1] for i_1, i_2, j_1, j_2 in indices]

    for (unique_values, unique_counts), expected_count in zip(unique, expected):
        assert len(unique_values) == 1
        assert len(unique_counts) == 1
        assert unique_counts[0] == expected_count


def test_seed(eopatch1: EOPatch) -> None:
    """seed=2 should produce different result than seed=1"""
    bins = [0.2, 0.5, 0.8]
    split_task = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type=TrainTestSplitType.PER_CLASS)
    result_seed1 = np.copy(split_task(eopatch1, seed=1)[OUTPUT_FEATURE])
    result_seed2 = np.copy(split_task(eopatch1, seed=2)[OUTPUT_FEATURE])
    result_seed1_rerun = np.copy(split_task(eopatch1, seed=1)[OUTPUT_FEATURE])

    assert not np.array_equal(result_seed1, result_seed2)
    assert_array_equal(result_seed1, result_seed1_rerun)


def test_ignore_value(eopatch1: EOPatch) -> None:
    """test ignore_values=[2]"""

    bins = [0.2, 0.5, 0.7, 0.8]
    expected_unique = set(range(0, len(bins) + 2))

    split_task = TrainTestSplitTask(
        (FeatureType.MASK_TIMELESS, "TEST"),
        (FeatureType.MASK_TIMELESS, "BINS"),
        bins,
        split_type=TrainTestSplitType.PER_CLASS,
        ignore_values=[2],
    )

    eopatch1 = split_task(eopatch1)

    assert set(np.unique(eopatch1[(FeatureType.MASK_TIMELESS, "BINS")])) <= expected_unique
    assert np.all(eopatch1[(FeatureType.MASK_TIMELESS, "BINS")][eopatch1[INPUT_FEATURE] == 2] == 0)


def test_train_split_per_pixel(eopatch1: EOPatch) -> None:
    bins = [0.2, 0.6]
    split_task = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type=TrainTestSplitType.PER_PIXEL)
    eopatch1 = split_task(eopatch1, seed=1)

    unique, counts = np.unique(eopatch1[OUTPUT_FEATURE], return_counts=True)
    class_percentages = np.round(counts / eopatch1[INPUT_FEATURE].size, 1)
    expected_unique = list(range(1, len(bins) + 2))

    assert_array_equal(unique, expected_unique)
    assert_array_equal(class_percentages, [0.2, 0.4, 0.4])


def test_train_split_per_value(eopatch1: EOPatch, eopatch2: EOPatch) -> None:
    """Test if class ids get assigned to the same subclasses in multiple eopatches"""
    bins = [0.2, 0.6]
    split_task = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type=TrainTestSplitType.PER_VALUE)

    eopatch1, eopatch2 = split_task(eopatch1), split_task(eopatch2)
    otuput1, otuput2 = eopatch1[OUTPUT_FEATURE], eopatch2[OUTPUT_FEATURE]

    unique = set(np.unique(eopatch1[INPUT_FEATURE])) | set(np.unique(eopatch2[INPUT_FEATURE]))

    for uniq in unique:
        folds1 = otuput1[eopatch1[INPUT_FEATURE] == uniq]
        folds2 = otuput2[eopatch2[INPUT_FEATURE] == uniq]
        assert_array_equal(np.unique(folds1), np.unique(folds2))
