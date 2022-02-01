"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from datetime import date, timedelta

import numpy as np
from numpy.testing import assert_array_equal

from eolearn.core import EOPatch, FeatureType
from eolearn.features import AddMaxMinNDVISlopeIndicesTask, AddMaxMinTemporalIndicesTask, AddSpatioTemporalFeaturesTask


def test_temporal_indices():
    eopatch = EOPatch()
    t, h, w, c = 5, 3, 3, 2

    ndvi_shape = (t, h, w, 1)
    eopatch[FeatureType.DATA, "NDVI"] = np.arange(np.prod(ndvi_shape)).reshape(ndvi_shape)

    valid_data = np.ones(ndvi_shape, bool)
    valid_data[0] = 0
    valid_data[-1] = 0
    eopatch[FeatureType.MASK, "IS_DATA"] = np.ones(ndvi_shape, dtype=np.int16)
    eopatch[FeatureType.MASK, "VALID_DATA"] = valid_data

    new_eopatch = AddMaxMinTemporalIndicesTask(mask_data=False)(eopatch)

    assert_array_equal(new_eopatch.data_timeless["ARGMIN_NDVI"], np.zeros((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_NDVI"], (t - 1) * np.ones((h, w, 1)))

    new_eopatch = AddMaxMinTemporalIndicesTask(mask_data=True)(eopatch)

    assert_array_equal(new_eopatch.data_timeless["ARGMIN_NDVI"], np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_NDVI"], (t - 2) * np.ones((h, w, 1)))

    bands_shape = (t, h, w, c)
    eopatch[FeatureType.DATA, "BANDS"] = np.arange(np.prod(bands_shape)).reshape(bands_shape)
    add_bands = AddMaxMinTemporalIndicesTask(
        data_feature="BANDS",
        data_index=1,
        amax_data_feature="ARGMAX_B1",
        amin_data_feature="ARGMIN_B1",
        mask_data=False,
    )
    new_eopatch = add_bands(eopatch)

    assert_array_equal(new_eopatch.data_timeless["ARGMIN_B1"], np.zeros((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_B1"], (t - 1) * np.ones((h, w, 1)))


def test_ndvi_slope_indices():
    timestamp = [date(2018, 3, 1) + timedelta(days=x) for x in range(11)]
    eopatch = EOPatch(timestamp=list(timestamp))

    t, h, w, = (
        10,
        3,
        3,
    )
    ndvi_shape = (t, h, w, 1)
    xx = np.zeros(ndvi_shape, np.float32)
    x = np.linspace(0, np.pi, t)
    xx[:, :, :, :] = x[:, None, None, None]

    eopatch[FeatureType.DATA, "NDVI"] = np.sin(xx)

    valid_data = np.ones(ndvi_shape, np.uint8)
    valid_data[1] = 0
    valid_data[-1] = 0
    valid_data[4] = 0

    eopatch[FeatureType.MASK, "IS_DATA"] = np.ones(ndvi_shape, bool)
    eopatch[FeatureType.MASK, "VALID_DATA"] = valid_data

    add_ndvi_task = AddMaxMinTemporalIndicesTask(mask_data=False)
    add_ndvi_slope_task = AddMaxMinNDVISlopeIndicesTask(mask_data=False)

    new_eopatch = add_ndvi_slope_task(add_ndvi_task(eopatch))

    assert_array_equal(new_eopatch.data_timeless["ARGMIN_NDVI"], (t - 1) * np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_NDVI"], (t // 2 - 1) * np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMIN_NDVI_SLOPE"], (t - 2) * np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_NDVI_SLOPE"], np.ones((h, w, 1)))

    add_ndvi_task = AddMaxMinTemporalIndicesTask(mask_data=True)
    add_ndvi_slope_task = AddMaxMinNDVISlopeIndicesTask(mask_data=True)

    new_eopatch = add_ndvi_slope_task(add_ndvi_task(eopatch))

    assert_array_equal(new_eopatch.data_timeless["ARGMIN_NDVI"], 0 * np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_NDVI"], (t // 2) * np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMIN_NDVI_SLOPE"], (t - 3) * np.ones((h, w, 1)))
    assert_array_equal(new_eopatch.data_timeless["ARGMAX_NDVI_SLOPE"], 2 * np.ones((h, w, 1)))


def test_stf_task():
    timestamp = [date(2018, 3, 1) + timedelta(days=x) for x in range(11)]
    eopatch = EOPatch(timestamp=list(timestamp))

    t, h, w, c = 10, 3, 3, 2

    # NDVI is a sinusoid where max slope is at index 1 and min slope at index 8
    ndvi_shape = (t, h, w, 1)
    bands_shape = (t, h, w, c)
    xx = np.zeros(ndvi_shape, np.float32)
    x = np.linspace(0, np.pi, t)
    xx[:, :, :, :] = x[:, None, None, None]

    bands = np.ones(bands_shape) * np.arange(t)[:, None, None, None]

    eopatch[FeatureType.DATA, "NDVI"] = np.sin(xx)
    eopatch[FeatureType.DATA, "BANDS"] = bands
    eopatch[FeatureType.MASK, "IS_DATA"] = np.ones(ndvi_shape, bool)

    add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
    add_bands = AddMaxMinTemporalIndicesTask(
        data_feature="BANDS",
        data_index=1,
        amax_data_feature="ARGMAX_B1",
        amin_data_feature="ARGMIN_B1",
        mask_data=False,
    )
    add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=False)
    add_stf = AddSpatioTemporalFeaturesTask(argmax_red="ARGMAX_B1", data_feature="BANDS", indices=[0, 1])

    new_eopatch = add_stf(add_ndvi_slope(add_bands(add_ndvi(eopatch))))
    result = new_eopatch.data_timeless["STF"]

    assert result.shape == (h, w, c * 5)
    assert_array_equal(result[:, :, 0:c], 4 * np.ones((h, w, c)))
    assert_array_equal(result[:, :, c : 2 * c], 9 * np.ones((h, w, c)))
    assert_array_equal(result[:, :, 2 * c : 3 * c], np.ones((h, w, c)))
    assert_array_equal(result[:, :, 3 * c : 4 * c], 8 * np.ones((h, w, c)))
    assert_array_equal(result[:, :, 4 * c : 5 * c], 9 * np.ones((h, w, c)))
