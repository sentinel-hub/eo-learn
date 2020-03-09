"""
Module containing tasks for morphological operations

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import skimage.morphology
import numpy as np

from eolearn.core import EOTask


class ErosionTask(EOTask):
    """
    The task performs an erosion to the provided mask

    :param mask_feature: The mask which is to be eroded
    :type mask_feature: (FeatureType, str)
    :param disk_radius: Radius of the erosion disk (in pixels). Default is set to `1`
    :type disk_radius: int
    :param erode_labels: List of labels to erode. If `None`, all unique labels are eroded. Default is `None`
    :type erode_labels: list(int)
    :param no_data_label: Value used to replace eroded pixels. Default is set to `0`
    :type no_data_label: int
    """

    def __init__(self, mask_feature, disk_radius=1, erode_labels=None, no_data_label=0):
        if not isinstance(disk_radius, int) or disk_radius is None or disk_radius < 1:
            raise ValueError('Disk radius should be an integer larger than 0!')

        self.mask_type, self.mask_name, self.new_mask_name = next(self._parse_features(mask_feature, new_names=True)())
        self.disk = skimage.morphology.disk(disk_radius)
        self.erode_labels = erode_labels
        self.no_data_label = no_data_label

    def execute(self, eopatch):
        feature_array = eopatch[(self.mask_type, self.mask_name)].squeeze().copy()

        all_labels = np.unique(feature_array)
        erode_labels = self.erode_labels if self.erode_labels else all_labels

        erode_labels = set(erode_labels) - {self.no_data_label}
        other_labels = set(all_labels) - set(erode_labels) - {self.no_data_label}

        eroded_masks = [skimage.morphology.binary_erosion(feature_array == label, self.disk) for label in erode_labels]
        other_masks = [feature_array == label for label in other_labels]

        merged_mask = np.logical_or.reduce(eroded_masks + other_masks, axis=0)

        feature_array[~merged_mask] = self.no_data_label
        eopatch[(self.mask_type, self.new_mask_name)] = np.expand_dims(feature_array, axis=-1)

        return eopatch
