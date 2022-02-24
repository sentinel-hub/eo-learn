"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core import EOPatch, FeatureType
from eolearn.ml_tools import TrainTestSplitTask

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
def test_bad_args(bad_arg, bad_kwargs):
    with pytest.raises(ValueError):
        TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bad_arg, **bad_kwargs)


def test_train_split():
    shape = (1000, 1000, 3)
    data = np.random.randint(10, size=shape, dtype=int)

    indices = [(0, 2, 0, 2), (0, 2, 2, 4), (2, 4, 0, 2), (2, 4, 2, 4), (0, 4, 4, 8), (4, 8, 0, 4), (4, 8, 4, 8)]
    for index, (i_1, i_2, j_1, j_2) in enumerate(indices, 1):
        data[i_1:i_2, j_1:j_2, :] = index * 11

    patch = EOPatch()
    patch[INPUT_FEATURE] = data

    bins = [0.2, 0.5, 0.8]
    expected_unique = set(range(1, len(bins) + 2))

    patch = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type="per_class")(patch, seed=1)
    assert set(np.unique(patch[OUTPUT_FEATURE])) <= expected_unique

    result_seed1 = np.copy(patch[OUTPUT_FEATURE])
    unique = (np.unique(result_seed1[i_1:i_2, j_1:j_2, :], return_counts=True) for i_1, i_2, j_1, j_2 in indices)
    expected = [(i_2 - i_1) * (j_2 - j_1) * shape[-1] for i_1, i_2, j_1, j_2 in indices]

    for (unique_values, unique_counts), expected_count in zip(unique, expected):
        assert len(unique_values) == 1
        assert len(unique_counts) == 1
        assert unique_counts[0] == expected_count

    # seed=2 should produce different result than seed=1
    patch = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type="per_class")(patch, seed=2)
    result_seed2 = np.copy(patch[OUTPUT_FEATURE])
    assert set(np.unique(result_seed2)) <= expected_unique
    assert not np.array_equal(result_seed1, result_seed2)

    # test with seed 1 should produce the same result as before
    patch = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type="per_class")(patch, seed=1)
    result_seed_equal = patch[OUTPUT_FEATURE]
    assert set(np.unique(result_seed2)) <= expected_unique
    assert_array_equal(result_seed1, result_seed_equal)

    # test ignore_values=[2]

    bins = [0.2, 0.5, 0.7, 0.8]
    expected_unique = set(range(0, len(bins) + 2))

    data = np.random.randint(10, size=shape)
    patch[INPUT_FEATURE] = data

    split_task = TrainTestSplitTask(
        (FeatureType.MASK_TIMELESS, "TEST"),
        (FeatureType.MASK_TIMELESS, "BINS"),
        bins,
        split_type="per_class",
        ignore_values=[2],
    )

    patch = split_task(patch, seed=542)

    assert set(np.unique(patch[(FeatureType.MASK_TIMELESS, "BINS")])) <= expected_unique
    assert np.all(patch[(FeatureType.MASK_TIMELESS, "BINS")][data == 2] == 0)


def test_train_split_per_pixel():
    shape = (1000, 1000, 3)

    input_data = np.random.randint(10, size=shape, dtype=int)
    patch = EOPatch()
    patch[INPUT_FEATURE] = input_data

    bins = [0.2, 0.6]
    patch = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type="per_pixel")(patch, seed=1)

    output_data = patch[OUTPUT_FEATURE]
    unique, counts = np.unique(output_data, return_counts=True)
    class_percentages = np.round(counts / input_data.size, 1)
    expected_unique = list(range(1, len(bins) + 2))

    assert_array_equal(unique, expected_unique)
    assert_array_equal(class_percentages, [0.2, 0.4, 0.4])


def test_train_split_per_value():
    """Test if class ids get assigned to the same subclasses in multiple eopatches"""
    shape = (1000, 1000, 3)

    input1 = np.random.randint(10, size=shape, dtype=int)
    input2 = np.random.randint(10, size=shape, dtype=int)

    patch1 = EOPatch()
    patch1[INPUT_FEATURE] = input1

    patch2 = EOPatch()
    patch2[INPUT_FEATURE] = input2

    bins = [0.2, 0.6]

    split_task = TrainTestSplitTask(INPUT_FEATURE, OUTPUT_FEATURE, bins, split_type="per_value")

    # seeds should get ignored when splitting 'per_value'
    patch1 = split_task(patch1, seed=1)
    patch2 = split_task(patch2, seed=1)

    otuput1 = patch1[OUTPUT_FEATURE]
    otuput2 = patch2[OUTPUT_FEATURE]

    unique = set(np.unique(input1)) | set(np.unique(input2))

    for uniq in unique:
        folds1 = otuput1[input1 == uniq]
        folds2 = otuput2[input2 == uniq]
        assert_array_equal(np.unique(folds1), np.unique(folds2))
