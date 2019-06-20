"""
Module containing tasks for morphological operations
"""

import logging

import skimage.morphology
import numpy as np

from eolearn.core import EOTask

LOGGER = logging.getLogger(__name__)


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
        self.mask_type, self.mask_name, self.new_mask_name = next(iter(self._parse_features(mask_feature,
                                                                                            new_names=True)))
        self.disk_radius = disk_radius
        self.erode_labels = erode_labels
        self.no_data_label = no_data_label

    def execute(self, eopatch):

        if self.disk_radius is None or self.disk_radius < 1 or not isinstance(self.disk_radius, int):
            LOGGER.warning('Disk radius should be an integer larger than 0! Ignoring erosion task.')
            return eopatch

        labels = eopatch[self.mask_type][self.mask_name].squeeze().copy()

        patch_labels = np.unique(labels)
        erode_labels = patch_labels if self.erode_labels is None else self.erode_labels

        mask_values = np.zeros(labels.shape, dtype=np.bool)
        for label in patch_labels:
            if label == self.no_data_label:
                continue

            label_mask = (labels == label)
            if label in erode_labels:
                label_mask = skimage.morphology.binary_erosion(label_mask, skimage.morphology.disk(self.disk_radius))
            mask_values |= label_mask

        labels[~mask_values] = self.no_data_label
        eopatch[self.mask_type][self.new_mask_name] = np.expand_dims(labels, axis=-1)
        return eopatch
