"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging

import pytest
import numpy as np
from eolearn.core import EOPatch, FeatureType
from eolearn.coregistration import ECCRegistrationTask, InterpolationType

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(name='eopatch')
def eopatch_fixture():
    bands = np.zeros((2, 20, 20, 1))
    bands[1] = np.arange(400).reshape(1, 20, 20, 1) / 400
    bands[0] = bands[1]
    bands[1, 5:15, 5:15, :] = .5
    bands[0, 7:17, 5:15, :] = .5
    mask = np.ones((2, 20, 20, 1), dtype=np.int16)
    ndvi = np.ones((2, 20, 20, 1))
    dem = np.ones((20, 20, 1))

    eop = EOPatch()
    eop.add_feature(FeatureType.DATA, 'bands', value=bands)
    eop.add_feature(FeatureType.DATA, 'ndvi', value=ndvi)
    eop.add_feature(FeatureType.MASK, 'cm', value=mask)
    eop.add_feature(FeatureType.DATA_TIMELESS, 'dem', value=dem)
    return eop


def test_registration(eopatch):
    reg = ECCRegistrationTask(
        (FeatureType.DATA, 'bands'),
        valid_mask_feature='cm',
        interpolation_type=InterpolationType.NEAREST,
        apply_to_features={FeatureType.DATA: {'bands', 'ndvi'}, FeatureType.MASK: {'cm'}}
    )
    reopatch = reg(eopatch)

    assert eopatch.data['bands'].shape == reopatch.data['bands'].shape, \
        "Shapes of .data['bands'] do not match"
    assert eopatch.data['ndvi'].shape == reopatch.data['ndvi'].shape, \
        "Shapes of .data['ndvi'] do not match"
    assert eopatch.mask['cm'].shape == reopatch.mask['cm'].shape, \
        "Shapes of .mask['cm'] do not match"
    assert eopatch.data_timeless['dem'].shape == reopatch.data_timeless['dem'].shape, \
        "Shapes of .data['bands'] do not match"
    assert not np.allclose(eopatch.data['bands'], reopatch.data['bands']), \
        "Registration did not warp .data['bands']"
    assert not np.allclose(eopatch.data['ndvi'], reopatch.data['ndvi']), \
        "Registration did not warp .data['ndvi']"
    assert not np.allclose(eopatch.mask['cm'], reopatch.mask['cm']), \
        "Registration did not warp .mask['cm']"
    assert np.allclose(eopatch.data_timeless['dem'], reopatch.data_timeless['dem']), \
        'Registration did warp data_timeless'
