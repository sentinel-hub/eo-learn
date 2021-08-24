"""
Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Jernej Puc, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest
import numpy as np

from eolearn.core import FeatureType
from eolearn.mask import MaskFeatureTask


BANDS_FEATURE = FeatureType.DATA, 'BANDS-S2-L1C'
NDVI_FEATURE = FeatureType.DATA, 'NDVI'
CLOUD_MASK_FEATURE = FeatureType.MASK, 'CLM'
LULC_FEATURE = FeatureType.MASK_TIMELESS, 'LULC'


def test_bands_with_clm(test_eopatch):
    new_feature = FeatureType.DATA, 'BANDS-S2-L1C_MASKED'

    mask_task = MaskFeatureTask(BANDS_FEATURE, CLOUD_MASK_FEATURE, mask_values=[True], no_data_value=-1)
    eop = mask_task(test_eopatch)

    masked_count = np.count_nonzero(eop[new_feature] == -1)
    clm_count = np.count_nonzero(eop[CLOUD_MASK_FEATURE])
    bands_num = eop[BANDS_FEATURE].shape[-1]
    assert masked_count == clm_count * bands_num


def test_ndvi_with_clm(test_eopatch):
    new_feature = FeatureType.DATA, 'NDVI_MASKED'

    mask_task = MaskFeatureTask(NDVI_FEATURE, CLOUD_MASK_FEATURE, mask_values=[True])
    eop = mask_task(test_eopatch)

    masked_count = np.count_nonzero(np.isnan(eop[new_feature]))
    clm_count = np.count_nonzero(eop[CLOUD_MASK_FEATURE])
    assert masked_count == clm_count


def test_clm_with_lulc(test_eopatch):
    new_feature = FeatureType.MASK, 'CLM_MASKED'

    mask_task = MaskFeatureTask(CLOUD_MASK_FEATURE, LULC_FEATURE, mask_values=[2], no_data_value=255)
    eop = mask_task(test_eopatch)

    masked_count = np.count_nonzero(eop[new_feature] == 255)
    lulc_count = np.count_nonzero(eop[LULC_FEATURE] == 2)
    bands_num = eop[CLOUD_MASK_FEATURE].shape[-1]
    time_num = eop[CLOUD_MASK_FEATURE].shape[0]
    assert masked_count == lulc_count * time_num * bands_num


def test_lulc_with_lulc(test_eopatch):
    new_feature = FeatureType.MASK_TIMELESS, 'LULC_MASKED'

    mask_task = MaskFeatureTask(LULC_FEATURE, LULC_FEATURE, mask_values=[1], no_data_value=100)
    eop = mask_task(test_eopatch)

    masked_count = np.count_nonzero(eop[new_feature] == 100)
    lulc_count = np.count_nonzero(eop[LULC_FEATURE] == 1)
    assert masked_count == lulc_count


def test_wrong_arguments():
    with pytest.raises(ValueError):
        MaskFeatureTask(BANDS_FEATURE, CLOUD_MASK_FEATURE, mask_values=10)
