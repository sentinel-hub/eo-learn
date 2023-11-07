"""
Module for computing clusters in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import Feature


class ClusteringTask(EOTask):
    """
    Tasks computes clusters on selected features using `sklearn.cluster.AgglomerativeClustering`.

    The algorithm produces a timeless data feature where each cell has a natural number which corresponds to specific
    group. The cells marked with -1 are not marking clusters. They are either being excluded by a mask or later removed
    by depending on the 'remove_small' threshold.
    """

    def __init__(
        self,
        features: Feature,
        new_feature_name: str,
        distance_threshold: float | None = None,
        n_clusters: int | None = None,
        affinity: Literal["euclidean", "l1", "l2", "manhattan", "cosine"] = "cosine",
        linkage: Literal["ward", "complete", "average", "single"] = "single",
        remove_small: int = 0,
        connectivity: None | np.ndarray | Callable = None,
        mask_name: str | None = None,
    ):
        """Class constructor

        :param features: A collection of features used for clustering. The features need to be of type DATA_TIMELESS
        :param new_feature_name: Name of feature that is the result of clustering
        :param distance_threshold: The linkage distance threshold above which, clusters will not be merged. If non None,
            n_clusters must be None nd compute_full_tree must be True
        :param n_clusters: The number of clusters found by the algorithm. If distance_threshold=None, it will be equal
            to the given n_clusters
        :param affinity: Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”.
        :param linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between
            sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
            - ward minimizes the variance of the clusters being merged.
            - average uses the average of the distances of each observation of the two sets.
            - complete or maximum linkage uses the maximum distances between all observations of the two sets.
            - single uses the minimum of the distances between all observations of the two sets.
        :param remove_small: If greater than 0, removes all clusters that have fewer points as "remove_small"
        :param connectivity: Connectivity matrix. Defines for each sample the neighboring samples following a given
            structure of the data. This can be a connectivity matrix itself or a callable that transforms the data into
            a connectivity matrix, such as derived from neighbors_graph. If set to None it uses the graph that has
            adjacent pixels connected.
        :param mask_name: An optional mask feature used for exclusion of the area from clustering
        """
        self.features_parser = self.get_feature_parser(features, allowed_feature_types=[FeatureType.DATA_TIMELESS])
        self.distance_threshold = distance_threshold
        self.affinity = affinity
        self.linkage = linkage
        self.new_feature_name = new_feature_name
        self.n_clusters = n_clusters
        self.compute_full_tree: Literal["auto"] | bool = "auto" if distance_threshold is None else True
        self.remove_small = remove_small
        self.connectivity = connectivity
        self.mask_name = mask_name

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: Input EOPatch
        :return: Transformed EOPatch
        """
        relevant_features = self.features_parser.get_features(eopatch)
        data = np.concatenate([eopatch[feature] for feature in relevant_features], axis=2)

        # Reshapes the data, because AgglomerativeClustering method only takes one dimensional arrays of vectors
        height, width, num_channels = data.shape
        data = np.reshape(data, (-1, num_channels))

        graph_args = {"n_x": height, "n_y": width}

        # All connections to masked pixels are removed
        if self.mask_name is not None:
            mask = eopatch.mask_timeless[self.mask_name].squeeze(axis=-1)
            graph_args["mask"] = mask
            data = data[np.ravel(mask) != 0]

        # If connectivity is not set, it uses pixel-to-pixel connections
        if not self.connectivity:
            self.connectivity = grid_to_graph(**graph_args)

        model = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            metric=self.affinity,
            linkage=self.linkage,
            connectivity=self.connectivity,
            n_clusters=self.n_clusters,
            compute_full_tree=self.compute_full_tree,
        )

        model.fit(data)
        result = model.labels_
        if self.remove_small > 0:
            for label, count in zip(*np.unique(result, return_counts=True)):
                if count < self.remove_small:
                    result[result == label] = -1

        # Transforms data back to original shape and setting all masked regions to -1
        if self.mask_name is not None:
            unmasked_result = np.full(height * width, -1)
            unmasked_result[np.ravel(mask) != 0] = result
            result = unmasked_result

        eopatch[FeatureType.DATA_TIMELESS, self.new_feature_name] = np.reshape(result, (height, width, 1))

        return eopatch
