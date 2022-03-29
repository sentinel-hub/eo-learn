"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from eolearn.core import EOPatch, FeatureType
from eolearn.features import FilterTimeSeriesTask, LinearFunctionTask, ValueFilloutTask


def test_content_after_timefilter():
    timestamps = [
        datetime.datetime(2017, 1, 1, 10, 4, 7),
        datetime.datetime(2017, 1, 4, 10, 14, 5),
        datetime.datetime(2017, 1, 11, 10, 3, 51),
        datetime.datetime(2017, 1, 14, 10, 13, 46),
        datetime.datetime(2017, 1, 24, 10, 14, 7),
        datetime.datetime(2017, 2, 10, 10, 1, 32),
        datetime.datetime(2017, 2, 20, 10, 6, 35),
        datetime.datetime(2017, 3, 2, 10, 0, 20),
        datetime.datetime(2017, 3, 12, 10, 7, 6),
        datetime.datetime(2017, 3, 15, 10, 12, 14),
    ]
    data = np.random.rand(10, 100, 100, 3)

    new_start, new_end = 4, -3

    new_interval = (timestamps[new_start], timestamps[new_end])

    new_timestamps = timestamps[new_start : new_end + 1]

    eop = EOPatch(timestamp=timestamps, data={"data": data})

    filter_task = FilterTimeSeriesTask(start_date=new_interval[0], end_date=new_interval[1])
    filter_task.execute(eop)

    assert new_timestamps == eop.timestamp


def test_fill():
    array = np.array(
        [
            [np.NaN] * 4 + [1.0, 2.0, 3.0, 4.0] + [np.NaN] * 3,
            [1.0, np.NaN, np.NaN, 2.0, 3.0, np.NaN, np.NaN, np.NaN, 4.0, np.NaN, 5.0],
        ]
    )

    array_ffill = ValueFilloutTask.fill(array, operation="f")
    array_bfill = ValueFilloutTask.fill(array, operation="b")
    array_fbfill = ValueFilloutTask.fill(array_ffill, operation="b")

    assert_allclose(
        array_ffill,
        np.array(
            [
                [np.NaN] * 4 + [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            ]
        ),
        equal_nan=True,
    )

    assert_allclose(
        array_bfill,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0] + [np.NaN] * 3,
                [1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0],
            ]
        ),
        equal_nan=True,
    )

    assert_allclose(
        array_fbfill,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "input_feature, operation",
    [((FeatureType.DATA, "TEST"), "x"), ((FeatureType.DATA, "TEST"), 4), (None, "f"), (np.zeros((4, 5)), "x")],
)
def test_bad_input(input_feature, operation):
    with pytest.raises(ValueError):
        ValueFilloutTask(input_feature, operations=operation)


def test_value_fillout():
    feature = (FeatureType.DATA, "TEST")
    shape = (8, 10, 10, 5)
    data = np.random.randint(0, 100, size=shape).astype(float)
    eopatch = EOPatch(data={"TEST": data})

    # nothing to be filled, return the same eopatch object immediately
    eopatch_new = ValueFilloutTask(feature, operations="fb", axis=0)(eopatch)
    assert eopatch == eopatch_new

    eopatch[feature][0, 0, 0, :] = np.nan

    def execute_fillout(eopatch, feature, **kwargs):
        input_array = eopatch[feature]
        eopatch = ValueFilloutTask(feature, **kwargs)(eopatch)
        output_array = eopatch[feature]
        return eopatch, input_array, output_array

    # filling forward temporally should not fill nans
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="f", axis=0)
    compare_mask = ~np.isnan(input_array)
    assert np.isnan(output_array[0, 0, 0, :]).all()
    assert_array_equal(input_array[compare_mask], output_array[compare_mask])
    assert id(input_array) != id(output_array)

    # filling in any direction along axis=-1 should also not fill nans since all neighbors are nans
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="fb", axis=-1)
    assert np.isnan(output_array[0, 0, 0, :]).all()
    assert id(input_array) != id(output_array)

    # filling nans backwards temporally should fill nans
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="b", axis=0)
    assert not np.isnan(output_array).any()
    assert id(input_array) != id(output_array)

    # try filling something else than nan (e.g.: -1)
    eopatch[feature][0, :, 0, 0] = -1
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="b", value=-1, axis=-1)
    assert not (output_array == -1).any()
    assert id(input_array) != id(output_array)

    # [nan, 1, nan, 2, ... ]  ---('fb')---> [1, 1, 1, 2, ... ]
    eopatch[feature][0, 0, 0, 0:4] = [np.nan, 1, np.nan, 2]
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="fb", axis=-1)
    assert_array_equal(output_array[0, 0, 0, 0:4], [1, 1, 1, 2])
    assert id(input_array) != id(output_array)

    # [nan, 1, nan, 2, ... ]  ---('bf')---> [1, 1, 2, 2, ... ]
    eopatch[feature][0, 0, 0, 0:4] = [np.nan, 1, np.nan, 2]
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="bf", axis=-1)
    assert_array_equal(output_array[0, 0, 0, 0:4], [1, 1, 2, 2])
    assert id(input_array) != id(output_array)


def test_linear_function_task():
    eopatch = EOPatch()

    data_feature = (FeatureType.DATA, "DATA_TEST")
    data_result_feature = (FeatureType.DATA, "DATA_TRANSFORMED")
    data_shape = (8, 10, 10, 5)
    eopatch[data_feature] = np.arange(np.prod(data_shape)).reshape(data_shape).astype(np.float32)

    mask_timeless_feature = (FeatureType.MASK_TIMELESS, "MASK_TIMELESS_TEST")
    mask_timeless_result_feature = (FeatureType.MASK_TIMELESS, "MASK_TIMELESS_TRANSFORMED")
    mask_shape = (10, 10, 1)
    eopatch[mask_timeless_feature] = np.ones(mask_shape, dtype=np.uint32)

    task_default = LinearFunctionTask(data_feature, data_result_feature)
    task_default(eopatch)
    assert np.array_equal(eopatch[data_feature], eopatch[data_result_feature])

    task_double_minus_five = LinearFunctionTask(
        [data_feature, mask_timeless_feature],
        [data_result_feature, mask_timeless_result_feature],
        slope=2,
        intercept=-5,
    )
    task_double_minus_five(eopatch)
    expected_result = np.arange(np.prod(data_shape)).reshape(data_shape).astype(float) * 2 - 5
    assert np.array_equal(eopatch[data_result_feature], expected_result)
    assert np.array_equal(eopatch[mask_timeless_result_feature], np.ones(mask_shape) * 2 - 5)
    assert eopatch[data_result_feature].dtype == np.float32
    # The value of the mask timeless changes here

    task_change_dtype = LinearFunctionTask(
        mask_timeless_feature, mask_timeless_result_feature, slope=256, dtype=np.uint8
    )
    task_change_dtype(eopatch)
    assert np.array_equal(eopatch[mask_timeless_result_feature], np.zeros(mask_shape))
    assert eopatch[mask_timeless_result_feature].dtype == np.uint8

    task_override = LinearFunctionTask(
        [data_feature, mask_timeless_feature],
        slope=5,
    )
    task_override(eopatch)
    assert np.array_equal(eopatch[mask_timeless_feature], np.ones(mask_shape) * 5)
