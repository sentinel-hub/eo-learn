"""
A collection of bands extraction EOTasks

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType
from eolearn.features import EuclideanNormTask, NormalizedDifferenceIndexTask

INPUT_FEATURE = (FeatureType.DATA, "TEST")


def test_euclidean_norm():
    eopatch = EOPatch()

    data = np.zeros(5 * 10 * 10 * 7).reshape(5, 10, 10, 7)
    bands = [0, 1, 2, 4, 6]
    data[..., bands] = 1

    eopatch[INPUT_FEATURE] = data

    eopatch = EuclideanNormTask(INPUT_FEATURE, (FeatureType.DATA, "NORM"), bands)(eopatch)
    assert (eopatch.data["NORM"] == np.sqrt(len(bands))).all()


@pytest.mark.parametrize("bad_input", ([1, 2, 3], "test", 0.5))
def test_bad_input(bad_input):
    with pytest.raises(ValueError):
        NormalizedDifferenceIndexTask(INPUT_FEATURE, (FeatureType.DATA, "NDI"), bands=bad_input)


def test_ndi():

    eopatch = EOPatch()
    eopatch[INPUT_FEATURE] = np.zeros((4, 3, 3, 9))

    band_a, band_b = 4.123, 3.321
    eopatch[INPUT_FEATURE][..., 0] = band_a
    eopatch[INPUT_FEATURE][..., 1] = band_b
    eopatch = NormalizedDifferenceIndexTask(INPUT_FEATURE, (FeatureType.DATA, "NDI"), bands=[0, 1]).execute(eopatch)
    assert (eopatch.data["NDI"] == ((band_a - band_b) / (band_a + band_b))).all()

    eopatch[INPUT_FEATURE][..., 5] = np.nan
    eopatch[INPUT_FEATURE][..., 7] = np.inf
    eopatch = NormalizedDifferenceIndexTask(INPUT_FEATURE, (FeatureType.DATA, "NAN_INF_INPUT"), bands=[5, 7]).execute(
        eopatch
    )
    assert np.isnan(eopatch.data["NAN_INF_INPUT"]).all()

    eopatch[INPUT_FEATURE][..., 1] = 1
    eopatch[INPUT_FEATURE][..., 3] = -1
    eopatch = NormalizedDifferenceIndexTask(INPUT_FEATURE, (FeatureType.DATA, "DIV_ZERO_NAN"), bands=[1, 3]).execute(
        eopatch
    )
    assert np.isnan(eopatch.data["DIV_ZERO_NAN"]).all()

    eopatch[INPUT_FEATURE][..., 1] = 0
    eopatch[INPUT_FEATURE][..., 3] = 0
    eopatch = NormalizedDifferenceIndexTask(
        INPUT_FEATURE, (FeatureType.DATA, "DIV_INVALID"), bands=[1, 3], undefined_value=123
    ).execute(eopatch)
    assert (eopatch.data["DIV_INVALID"] == 123).all()
