"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core import FeatureType
from eolearn.mask import CloudMaskTask
from eolearn.mask.cloud_mask import _get_window_indices


@pytest.mark.parametrize(
    ("num_of_elements", "middle_idx", "window_size", "expected_indices"),
    [
        (100, 0, 10, (0, 10)),
        (100, 1, 10, (0, 10)),
        (100, 50, 10, (45, 55)),
        (271, 270, 10, (261, 271)),
        (314, 314, 10, (304, 314)),
        (100, 0, 11, (0, 11)),
        (100, 1, 11, (0, 11)),
        (100, 50, 11, (45, 56)),
        (271, 270, 11, (260, 271)),
        (314, 314, 11, (303, 314)),
        (11, 2, 11, (0, 11)),
        (11, 2, 33, (0, 11)),
    ],
    ids=str,
)
def test_window_indices_function(num_of_elements, middle_idx, window_size, expected_indices):
    min_idx, max_idx = _get_window_indices(num_of_elements, middle_idx, window_size)
    assert (min_idx, max_idx) == expected_indices

    test_list = list(range(num_of_elements))
    assert len(test_list[min_idx:max_idx]) == min(num_of_elements, window_size)


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
    assert np.mean(eop_clm.mask["CLM_TEST"]) == pytest.approx(np.mean(eop_clm.mask["CLM_S2C"]), abs=0.01)
    assert np.mean(eop_clm.data["CLP_TEST"]) == pytest.approx(np.mean(eop_clm.data["CLP_S2C"]), abs=0.01)

    # Check if most of the same times are flagged as cloudless
    cloudless = np.mean(eop_clm.mask["CLM_TEST"], axis=(1, 2, 3)) == 0
    assert np.mean(cloudless == eop_clm.label["IS_CLOUDLESS"][:, 0]) > 0.94

    # Check multi-temporal results and final mask
    assert_array_equal(eop_clm.data["CLP_MULTI_TEST"], test_eopatch.data["CLP_MULTI"])
    assert_array_equal(eop_clm.mask["CLM_MULTI_TEST"], test_eopatch.mask["CLM_MULTI"])
    assert_array_equal(eop_clm.mask["CLM_INTERSSIM_TEST"], test_eopatch.mask["CLM_INTERSSIM"])
