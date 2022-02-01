"""
Credits:
Copyright (c) 2020 Beno Šircelj (Josef Stefan Institute)
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging

import numpy as np
from pytest import approx

from eolearn.core import FeatureType
from eolearn.features import ClusteringTask

logging.basicConfig(level=logging.DEBUG)


def test_clustering(example_eopatch):
    test_features = {FeatureType.DATA_TIMELESS: ["DEM", "MAX_NDVI"]}
    mask = np.zeros_like(example_eopatch.mask_timeless["LULC"], dtype=np.uint8)
    mask[:90, :90] = 1
    example_eopatch.mask_timeless["mask"] = mask

    ClusteringTask(
        features=test_features,
        new_feature_name="clusters_small",
        n_clusters=100,
        affinity="cosine",
        linkage="single",
        remove_small=3,
    ).execute(example_eopatch)

    ClusteringTask(
        features=test_features,
        new_feature_name="clusters_mask",
        distance_threshold=0.00000001,
        affinity="cosine",
        linkage="average",
        mask_name="mask",
    ).execute(example_eopatch)

    clusters = example_eopatch.data_timeless["clusters_small"].squeeze()

    assert len(np.unique(clusters)) == 22, "Wrong number of clusters."
    assert np.median(clusters) == 2
    assert np.mean(clusters) == approx(2.19109)

    clusters = example_eopatch.data_timeless["clusters_mask"].squeeze()

    assert len(np.unique(clusters)) == 20, "Wrong number of clusters."
    assert np.median(clusters) == 0
    assert np.mean(clusters) == approx(-0.0948515)
    assert np.all(clusters[90:, 90:] == -1), "Wrong area"
