"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

import logging

import cv2
import numpy as np
import pytest

from eolearn.core import FeatureType
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.coregistration import ECCRegistrationTask

logging.basicConfig(level=logging.DEBUG)


def test_registration(example_eopatch):
    apply_to_features = [(FeatureType.DATA, "NDVI"), (FeatureType.DATA, "BANDS-S2-L1C"), (FeatureType.MASK, "CLM")]

    reg = ECCRegistrationTask(
        registration_feature=(FeatureType.DATA, "NDVI"),
        reference_feature=(FeatureType.DATA_TIMELESS, "MAX_NDVI"),
        channel=0,
        valid_mask_feature=None,
        apply_to_features=apply_to_features,
        interpolation_mode=cv2.INTER_LINEAR,
        warp_mode=cv2.MOTION_TRANSLATION,
        max_iter=100,
        gauss_kernel_size=1,
        border_mode=cv2.BORDER_REPLICATE,
        border_value=0,
        num_threads=-1,
        max_translation=5.0,
    )
    with pytest.warns(EORuntimeWarning):
        reopatch = reg(example_eopatch)

    assert example_eopatch.data["BANDS-S2-L1C"].shape == reopatch.data["BANDS-S2-L1C"].shape
    assert example_eopatch.data["NDVI"].shape == reopatch.data["NDVI"].shape
    assert example_eopatch.mask["CLM"].shape == reopatch.mask["CLM"].shape
    assert not np.allclose(
        example_eopatch.data["BANDS-S2-L1C"], reopatch.data["BANDS-S2-L1C"]
    ), "Registration did not warp .data['bands']"
    assert not np.allclose(
        example_eopatch.data["NDVI"], reopatch.data["NDVI"]
    ), "Registration did not warp .data['ndvi']"
    assert not np.allclose(example_eopatch.mask["CLM"], reopatch.mask["CLM"]), "Registration did not warp .mask['cm']"
    assert "warp_matrices" in reopatch.meta_info

    for warp_matrix in reopatch.meta_info["warp_matrices"].values():
        assert np.linalg.norm(np.array(warp_matrix)[:, 2]) <= 5.0, "Estimated translation is larger than max value"
