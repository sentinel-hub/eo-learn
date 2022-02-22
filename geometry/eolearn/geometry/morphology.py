"""
Module containing tasks for morphological operations

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import itertools as it
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import skimage.filters.rank
import skimage.morphology

from eolearn.core import EOTask, MapFeatureTask


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
            raise ValueError("Disk radius should be an integer larger than 0!")

        self.mask_type, self.mask_name, self.new_mask_name = self.parse_renamed_feature(mask_feature)
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


class MorphologicalOperations(Enum):
    """Enum class of morphological operations"""

    OPENING = "opening"
    CLOSING = "closing"
    DILATION = "dilation"
    EROSION = "erosion"
    MEDIAN = "median"

    @classmethod
    def get_operation(cls, morph_type):
        """Maps morphological operation type to function

        :param morph_type: Morphological operation type
        :type morph_type: MorphologicalOperations
        :return: function
        """
        return {
            cls.OPENING: skimage.morphology.opening,
            cls.CLOSING: skimage.morphology.closing,
            cls.DILATION: skimage.morphology.dilation,
            cls.EROSION: skimage.morphology.erosion,
            cls.MEDIAN: skimage.filters.rank.median,
        }[morph_type]


class MorphologicalStructFactory:
    """
    Factory methods for generating morphological structuring elements
    """

    @staticmethod
    def get_disk(radius):
        """
        :param radius: Radius of disk
        :type radius: int
        :return: The structuring element where elements of the neighborhood are 1 and 0 otherwise.
        :rtype: numpy.ndarray
        """
        return skimage.morphology.disk(radius)

    @staticmethod
    def get_diamond(radius):
        """
        :param radius: Radius of diamond
        :type radius: int
        :return: The structuring element where elements of the neighborhood are 1 and 0 otherwise.
        :rtype: numpy.ndarray
        """
        return skimage.morphology.diamond(radius)

    @staticmethod
    def get_rectangle(width, height):
        """
        :param width: Width of rectangle
        :type width: int
        :param height: Height of rectangle
        :type height: int
        :return: A structuring element consisting only of ones, i.e. every pixel belongs to the neighborhood.
        :rtype: numpy.ndarray
        """
        return skimage.morphology.rectangle(width, height)

    @staticmethod
    def get_square(width):
        """
        :param width: Size of square
        :type width: int
        :return: A structuring element consisting only of ones, i.e. every pixel belongs to the neighborhood.
        :rtype: numpy.ndarray
        """
        return skimage.morphology.square(width)


class MorphologicalFilterTask(MapFeatureTask):
    """Performs morphological operations on masks."""

    def __init__(
        self,
        input_features,
        output_features=None,
        *,
        morph_operation: Union[MorphologicalOperations, Callable],
        struct_elem: Optional[np.ndarray] = None,
    ):
        """
        :param input_features: Input features to be processed.
        :param output_features: Outputs of input features. If not provided the `input_features` are overwritten.
        :param morph_operation: A morphological operation.
        :param struct_elem: A structuring element to be used with the morphological operation. Usually it is generated
            with a factory method from MorphologicalStructElements
        """
        if output_features is None:
            output_features = input_features
        super().__init__(input_features, output_features)

        if isinstance(morph_operation, MorphologicalOperations):
            self.morph_operation = MorphologicalOperations.get_operation(morph_operation)
        else:
            self.morph_operation = morph_operation
        self.struct_elem = struct_elem

    def map_method(self, feature):
        """Applies the morphological operation to a raster feature."""
        feature = feature.copy()
        if feature.ndim == 3:
            for channel in range(feature.shape[2]):
                feature[..., channel] = self.morph_operation(feature[..., channel], self.struct_elem)
        elif feature.ndim == 4:
            for time, channel in it.product(range(feature.shape[0]), range(feature.shape[3])):
                feature[time, ..., channel] = self.morph_operation(feature[time, ..., channel], self.struct_elem)
        else:
            raise ValueError(f"Invalid number of dimensions: {feature.ndim}")

        return feature
