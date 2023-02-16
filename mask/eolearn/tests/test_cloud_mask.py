"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np
from numpy.testing import assert_array_equal
from pytest import approx

from eolearn.core import FeatureType
from eolearn.mask import CloudMaskTask
from eolearn.mask.cloud_mask import _get_window_indices


def test_window_indices_function():
    # normal case (middle index in the middle, window size < num_all_indices)
    assert_array_equal(_get_window_indices(middle_idx=45, window_size=5, num_all_indices=50), [45 - 2, 45 + 2 + 1])
    assert_array_equal(_get_window_indices(middle_idx=45, window_size=4, num_all_indices=50), [45 - 2, 45 + 1 + 1])

    # edge indices
    assert_array_equal(_get_window_indices(middle_idx=2, window_size=10, num_all_indices=50), [0, 10])
    assert_array_equal(_get_window_indices(middle_idx=50 - 2, window_size=10, num_all_indices=50), [50 - 10 - 1, 50])

    # window size larger than num of all indices
    assert_array_equal(_get_window_indices(middle_idx=0, window_size=50, num_all_indices=10), [0, 10])
    assert_array_equal(_get_window_indices(middle_idx=5, window_size=50, num_all_indices=10), [0, 10])
    assert_array_equal(_get_window_indices(middle_idx=1, window_size=50, num_all_indices=10), [0, 10])


def test_mono_temporal_cloud_detection(test_eopatch):
    add_tcm = CloudMaskTask(
        data_feature=(FeatureType.DATA, "BANDS-S2-L1C"),
        all_bands=True,
        is_data_feature=(FeatureType.MASK, "IS_DATA"),
        mono_features=("CLP_TEST", "CLM_TEST"),
        mask_feature=None,
        average_over=4,
        dilation_size=2,
        mono_threshold=0.4,
    )
    eop_clm = add_tcm(test_eopatch)

    assert_array_equal(eop_clm.mask["CLM_TEST"], test_eopatch.mask["CLM_S2C"])
    assert_array_equal(eop_clm.data["CLP_TEST"], test_eopatch.data["CLP_S2C"])


def test_multi_temporal_cloud_detection_downscaled(test_eopatch):
    add_tcm = CloudMaskTask(
        data_feature=(FeatureType.DATA, "BANDS-S2-L1C"),
        processing_resolution=120,
        mono_features=("CLP_TEST", "CLM_TEST"),
        multi_features=("CLP_MULTI_TEST", "CLM_MULTI_TEST"),
        mask_feature=(FeatureType.MASK, "CLM_INTERSSIM_TEST"),
        average_over=8,
        dilation_size=4,
    )
    eop_clm = add_tcm(test_eopatch)

    # Check shape and type
    for feature in ((FeatureType.MASK, "CLM_TEST"), (FeatureType.DATA, "CLP_TEST")):
        assert eop_clm[feature].ndim == 4
        assert eop_clm[feature].shape[:-1] == eop_clm.data["BANDS-S2-L1C"].shape[:-1]
        assert eop_clm[feature].shape[-1] == 1
    assert eop_clm.mask["CLM_TEST"].dtype == bool
    assert eop_clm.data["CLP_TEST"].dtype == np.float32

    # Compare mean cloud coverage with provided reference
    assert np.mean(eop_clm.mask["CLM_TEST"]) == approx(np.mean(eop_clm.mask["CLM_S2C"]), abs=0.01)
    assert np.mean(eop_clm.data["CLP_TEST"]) == approx(np.mean(eop_clm.data["CLP_S2C"]), abs=0.01)

    # Check if most of the same times are flagged as cloudless
    cloudless = np.mean(eop_clm.mask["CLM_TEST"], axis=(1, 2, 3)) == 0
    assert np.mean(cloudless == eop_clm.label["IS_CLOUDLESS"][:, 0]) > 0.94

    # Check multi-temporal results and final mask
    assert_array_equal(eop_clm.data["CLP_MULTI_TEST"], test_eopatch.data["CLP_MULTI"])
    assert_array_equal(eop_clm.mask["CLM_MULTI_TEST"], test_eopatch.mask["CLM_MULTI"])
    assert_array_equal(eop_clm.mask["CLM_INTERSSIM_TEST"], test_eopatch.mask["CLM_INTERSSIM"])
