"""
Module for computing clusters in EOPatch
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from eolearn.core import EOTask, FeatureType


class ClusteringTask(EOTask):
    """
    Tasks computes clusters on selected features using sklearn.cluster.AgglomerativeClustering.

    The algorithm produces a timeless data feature where each cell has a natural number which corresponds to specific
    group. The cells marked with -1 are not marking clusters. They are either being excluded by a mask or later removed
    by depending on the 'remove_small' threshold.
    """

    def __init__(self, features, new_feature_name, distance_threshold=None, n_clusters=None, affinity="cosine",
                 linkage="single", remove_small=0, connectivity=None, mask_name=None):
        """ Class constructor

        :param features: A collection of features used for clustering. The features need to be of type DATA_TIMELESS
        :type features: dict(FeatureType.DATA_TIMELESS: set(str))
        :param new_feature_name: Name of feature that is the result of clustering
        :type new_feature_name: str
        :param distance_threshold: The linkage distance threshold above which, clusters will not be merged. If non None,
            n_clusters must be None nd compute_full_tree must be True
        :type distance_threshold: float or None
        :param n_clusters: The number of clusters found by the algorithm. If distance_threshold=None, it will be equal
            to the given n_clusters
        :type n_clusters: int or None
        :param affinity: Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”.
        :type affinity: str
        :param linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between
            sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
            - ward minimizes the variance of the clusters being merged.
            - average uses the average of the distances of each observation of the two sets.
            - complete or maximum linkage uses the maximum distances between all observations of the two sets.
            - single uses the minimum of the distances between all observations of the two sets.
        :type linkage: {“ward”, “complete”, “average”, “single”}
        :param remove_small: If greater than 0, removes all clusters that have less points as "remove_small"
        :type remove_small: int
        :param connectivity: Connectivity matrix. Defines for each sample the neighboring samples following a given
            structure of the data. This can be a connectivity matrix itself or a callable that transforms the data into
            a connectivity matrix, such as derived from kneighbors_graph. If set to None it uses the graph that has
            adjacent pixels connected.
        :type connectivity: array-like, callable or None
        :param mask_name: An optional mask feature used for exclusion of the area from clustering
        :type mask_name: str
        """
        self.features = features
        self.distance_threshold = distance_threshold
        self.affinity = affinity
        self.linkage = linkage
        self.new_feature_name = new_feature_name
        self.n_clusters = n_clusters
        self.compute_full_tree = 'auto'
        if distance_threshold is not None:
            self.compute_full_tree = True
        if remove_small < 0:
            raise ValueError('remove_small argument should be non-negative')
        self.remove_small = remove_small
        self.connectivity = connectivity
        self.mask_name = mask_name

    @staticmethod
    def construct_data(eopatch, features):
        """
        :param eopatch: EOPatch with all features that will be used
        :param features: Features for clustering
        :return: array of vectors constructed from the features listed
        """

        feature = EOTask._parse_features(features, default_feature_type=FeatureType.DATA_TIMELESS)
        data = None
        for i in feature:
            if data is None:
                data = eopatch.data_timeless[i[1]]
            else:
                data = np.concatenate((data, eopatch.data_timeless[i[1]]), axis=2)

        return data

    def execute(self, eopatch):
        """
        :param eopatch: Input EOPatch
        :type eopatch: EOPatch
        :return: Transformed EOPatch
        :rtype: EOPatch
        """
        data = self.construct_data(eopatch, self.features)

        # Reshapes the data, because AgglomerativeClustering method only takes one dimensional arrays of vectors
        org_shape = data.shape
        data = np.reshape(data, (-1, org_shape[-1]))
        org_length = len(data)

        graph_args = {'n_x': org_shape[0], 'n_y': org_shape[1]}
        locations = None

        # All connections to masked pixels are removed
        if self.mask_name is not None:
            mask = eopatch.mask_timeless[self.mask_name].squeeze()
            graph_args['mask'] = mask
            locations = [i for i, elem in enumerate(np.ravel(mask)) if elem == 0]
            data = np.delete(data, locations, axis=0)

        # If connectivity is not set, it uses pixel-to-pixel connections
        if not self.connectivity:
            self.connectivity = grid_to_graph(**graph_args)

        model = AgglomerativeClustering(distance_threshold=self.distance_threshold, affinity=self.affinity,
                                        linkage=self.linkage,
                                        connectivity=self.connectivity,
                                        n_clusters=self.n_clusters,
                                        compute_full_tree=self.compute_full_tree)

        model.fit(data)
        trimmed_labels = model.labels_
        if self.remove_small > 0:

            # Counts how many pixels covers each cluster
            labels = np.zeros(model.n_clusters_)
            for i in trimmed_labels:
                labels[i] += 1

            # Sets to -1 all pixels corresponding to too small clusters
            for i, no_lab in enumerate(labels):
                if no_lab < self.remove_small:
                    trimmed_labels[trimmed_labels == i] = -1

        # Transforms data back to original shape and setting all masked regions to -1
        if self.mask_name is not None:
            new_data = [-1] * org_length
            for i, val in zip(np.delete(range(org_length), locations), trimmed_labels):
                new_data[i] = val
            trimmed_labels = new_data

        trimmed_labels = np.reshape(trimmed_labels, org_shape[:-1])

        eopatch.add_feature(FeatureType.DATA_TIMELESS, self.new_feature_name, trimmed_labels[..., np.newaxis])

        return eopatch
