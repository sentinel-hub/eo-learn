"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from numpy.testing import assert_array_equal

from eolearn.core import FeatureType
from eolearn.mask.extra.cloud_mask import CloudMaskTask


def test_cloud_detection(test_eopatch):
    add_tcm = CloudMaskTask(
        data_feature=(FeatureType.DATA, "BANDS-S2-L1C"),
        valid_data_feature=(FeatureType.MASK, "IS_DATA"),
        output_mask_feature=(FeatureType.MASK, "CLM_TEST"),
        output_proba_feature=(FeatureType.DATA, "CLP_TEST"),
        threshold=0.4,
        average_over=4,
        dilation_size=2,
    )
    eop_clm = add_tcm(test_eopatch)

    assert_array_equal(eop_clm.mask["CLM_TEST"], test_eopatch.mask["CLM_S2C"])
    assert_array_equal(eop_clm.data["CLP_TEST"], test_eopatch.data["CLP_S2C"])
