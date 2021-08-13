"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pytest

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import PointSampler, PointRasterSampler, PointSamplingTask

import numpy as np


N_SAMPLES = 100


@pytest.fixture(name='raster')
def raster_fixture():
    raster_size = (100, 100)
    raster = np.zeros(raster_size, dtype=np.uint8)
    raster[40:60, 40:60] = 1
    return raster


def test_point_sampler(raster):
    ps = PointSampler(raster)
    assert ps.area() == np.prod(raster.shape), 'Incorrect total area'
    assert ps.area(cc_index=0) == pytest.approx(400), 'Incorrect area for label 1'
    assert ps.area(cc_index=1) == pytest.approx(9600), 'Incorrect area for label 0'
    assert ps.geometries[0]['polygon'].envelope.bounds == (40, 40, 60, 60), 'Incorrect polygon bounds'
    assert ps.geometries[1]['polygon'].envelope.bounds == (0, 0, 100, 100), 'Incorrect polygon bounds'
    del ps

    ps = PointSampler(raster, no_data_value=0)
    assert ps.area() == 400, 'Incorrect total area'
    assert len(ps) == 1, 'Incorrect handling of no data values'
    del ps

    ps = PointSampler(raster, ignore_labels=[1])
    assert ps.area() == 9600, 'Incorrect total area'
    assert len(ps) == 1, 'Incorrect handling of no data values'
    del ps


def test_point_raster_sampler(raster):
    ps = PointRasterSampler([0, 1])
    # test if it raises ndim error
    with pytest.raises(ValueError):
        ps.sample(np.ones((1000,)))
        ps.sample(np.ones((1000, 1000, 3)))

    rows, cols = ps.sample(raster, n_samples=N_SAMPLES)
    labels = raster[rows, cols]

    assert len(labels) == N_SAMPLES, 'Incorrect number of samples'
    assert len(rows) == N_SAMPLES, 'Incorrect number of samples'
    assert len(cols) == N_SAMPLES, 'Incorrect number of samples'

    # test number of sample is proportional to class frequency
    assert np.sum(labels == 1) == int(N_SAMPLES * (400 / np.prod(raster.shape))), 'Incorrect sampling distribution'
    assert np.sum(labels == 0) == int(N_SAMPLES * (9600 / np.prod(raster.shape))), 'Incorrect sampling distribution'

    # test sampling is correct
    assert (labels == raster[rows, cols]).all(), 'Incorrect sampling'
    del ps

    # test even sampling of classes
    ps = PointRasterSampler([0, 1], even_sampling=True)
    rows, cols = ps.sample(raster, n_samples=N_SAMPLES)
    labels = raster[rows, cols]

    assert np.sum(labels == 1) == N_SAMPLES // 2, 'Incorrect sampling distribution'
    assert np.sum(labels == 0) == N_SAMPLES // 2, 'Incorrect sampling distribution'
    assert (labels == raster[rows, cols]).all(), 'Incorrect sampling'


def test_point_sampling_task(raster):
    # test PointSamplingTask
    t, h, w, d = 10, *raster.shape, 5
    eop = EOPatch()
    eop.data['bands'] = np.arange(t * h * w * d).reshape(t, h, w, d)
    eop.mask_timeless['raster'] = raster.reshape(raster.shape + (1,))

    task = PointSamplingTask(
        n_samples=N_SAMPLES,
        ref_mask_feature='raster',
        ref_labels=[0, 1],
        sample_features=[
            (FeatureType.DATA, 'bands', 'SAMPLED_DATA'),
            (FeatureType.MASK_TIMELESS, 'raster', 'SAMPLED_LABELS')
        ],
        even_sampling=True
    )

    task.execute(eop)
    # assert features, labels and sampled rows and cols are added to eopatch
    assert 'SAMPLED_LABELS' in eop.mask_timeless, 'Labels not added to eopatch'
    assert 'SAMPLED_DATA' in eop.data, 'Features not added to eopatch'
    # check validity of sampling
    assert eop.data['SAMPLED_DATA'].shape == (t, N_SAMPLES, 1, d), 'Incorrect features size'
    assert eop.mask_timeless['SAMPLED_LABELS'].shape == (N_SAMPLES, 1, 1), 'Incorrect number of samples'
