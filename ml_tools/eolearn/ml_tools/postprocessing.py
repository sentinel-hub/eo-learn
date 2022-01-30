"""
Module contains tools and EOTasks for morphological and postprocessing operations

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import itertools as it
from enum import Enum
from abc import abstractmethod

import skimage.morphology
import skimage.filters.rank

from eolearn.core import EOTask


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


class PostprocessingTask(EOTask):
    """Base class for all post-processing tasks

    :param feature: A feature to be processed
    :type feature: (FeatureType, str)
    """

    def __init__(self, feature):
        self.feature = self.parse_feature(feature)

    @abstractmethod
    def process(self, raster):
        """Abstract method for processing the raster"""
        raise NotImplementedError

    def execute(self, eopatch):
        """Execute method takes EOPatch and changes the specified feature"""
        eopatch[self.feature] = self.process(eopatch[self.feature])

        return eopatch


class MorphologicalFilterTask(PostprocessingTask):
    """EOTask that performs morphological operations on masks.

    :param feature: A feature to be processed
    :type feature: (FeatureType, str)
    :param morph_operation: Morphological operation
    :type morph_operation: MorphologicalOperations or function that operates on image
    :param struct_elem: The structuring element to be used with the morphological operation; usually generated with a
                        factory method from MorphologicalStructElements
    :type struct_elem: numpy.ndarray
    """

    def __init__(self, feature, morph_operation, struct_elem=None):
        super().__init__(feature)

        if isinstance(morph_operation, MorphologicalOperations):
            self.morph_operation = MorphologicalOperations.get_operation(morph_operation)
        else:
            self.morph_operation = morph_operation
        self.struct_elem = struct_elem

    def process(self, raster):
        """Applies the morphological operation to the mask object"""
        dim = len(raster.shape)
        if dim == 3:
            for dim in range(raster.shape[2]):
                raster[:, :, dim] = self.morph_operation(raster[:, :, dim], self.struct_elem)
        elif dim == 4:
            for time, dim in it.product(range(raster.shape[0]), range(raster.shape[3])):
                raster[time, :, :, dim] = self.morph_operation(raster[time, :, :, dim], self.struct_elem)
        else:
            raise ValueError(f"Invalid number of dimensions: {dim}")

        return raster
