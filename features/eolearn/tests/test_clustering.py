import logging

import numpy as np
from pytest import approx

from eolearn.core import FeatureType
from eolearn.features import ClusteringTask

logging.basicConfig(level=logging.DEBUG)


def test_clustering(test_eopatch):
    test_features = {FeatureType.DATA_TIMELESS: ['feature1', 'feature2']}

    ClusteringTask(
        features=test_features,
        new_feature_name='clusters_small',
        n_clusters=100,
        affinity='cosine',
        linkage='single',
        remove_small=3
    ).execute(test_eopatch)

    ClusteringTask(
        features=test_features,
        new_feature_name='clusters_mask',
        distance_threshold=0.1,
        affinity='cosine',
        linkage='average',
        mask_name='mask'
    ).execute(test_eopatch)

    clusters = test_eopatch.data_timeless['clusters_small'].squeeze()
    delta = 1e-3

    assert len(np.unique(clusters)) == 26, "Wrong number of clusters."
    assert np.median(clusters) == 92
    assert np.mean(clusters) == approx(68.665, abs=delta)

    clusters = test_eopatch.data_timeless['clusters_mask'].squeeze()
    delta = 1e-4

    assert len(np.unique(clusters)) == 45, "Wrong number of clusters."
    assert np.median(clusters) == -0.5
    assert np.mean(clusters) == approx(3.7075, abs=delta)
    assert np.all(clusters[0:5, 0:20] == -1), "Wrong area"
