"""
Module contains tools and EOTasks for morphological and postprocessing operations
"""

import itertools as it
from enum import Enum
from abc import abstractmethod

from skimage.morphology import opening, closing, erosion, dilation
from skimage.morphology import disk, square, rectangle, diamond
from skimage.filters.rank import median

from eolearn.core import EOTask


class MorphologicalOperations(Enum):
    """ Enum class of morphological operations
    """
    OPENING = 'opening'
    CLOSING = 'closing'
    DILATION = 'dilation'
    EROSION = 'erosion'
    MEDIAN = 'median'

    @classmethod
    def get_operation(cls, morph_type):
        """ Maps morphological operation type to function

        :param morph_type: Morphological operation type
        :type morph_type: MorphologicalOperations
        :return: function
        """
        return {
            cls.OPENING: opening,
            cls.CLOSING: closing,
            cls.DILATION: dilation,
            cls.EROSION: erosion,
            cls.MEDIAN: median
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
        return disk(radius)

    @staticmethod
    def get_diamond(radius):
        """
        :param radius: Radius of diamond
        :type radius: int
        :return: The structuring element where elements of the neighborhood are 1 and 0 otherwise.
        :rtype: numpy.ndarray
        """
        return diamond(radius)

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
        return rectangle(width, height)

    @staticmethod
    def get_square(width):
        """
        :param width: Size of square
        :type width: int
        :return: A structuring element consisting only of ones, i.e. every pixel belongs to the neighborhood.
        :rtype: numpy.ndarray
        """
        return square(width)


class PostprocessingTask(EOTask):
    """ Base class for all post-processing tasks

    :param feature_type: Type of the vector feature which will be added to EOPatch
    :type feature_type: eolearn.core.FeatureType
    :param feature_name: Name of the vector feature which will be added to EOPatch
    :type feature_name: str
    """
    def __init__(self, feature_type, feature_name):
        self.feature_type = feature_type
        self.feature_name = feature_name

    @abstractmethod
    def process(self, raster):
        raise NotImplementedError

    def execute(self, eopatch):
        """ Execute method takes EOPatch and changes the specified feature
        """

        if not eopatch.feature_exists(self.feature_type, self.feature_name):
            raise ValueError('Unknown feature {}, {}'.format(self.feature_type, self.feature_name))

        new_raster = self.process(eopatch.get_feature(self.feature_type, self.feature_name))

        eopatch.add_feature(self.feature_type, self.feature_name, new_raster)
        return eopatch


class MorphologicalFilterTask(PostprocessingTask):
    """ EOTask that performs morphological operations on masks.

    :param morph_operation: Morphological operation
    :type morph_operation: MorphologicalOperations or function that operates on image
    :param struct_elem: The structuring element to be used with the morphological operation; usually generated with a
                        factory method from MorphologicalStructElements
    :type struct_elem: numpy.ndarray
    """
    def __init__(self, feature_type, feature_name, morph_operation, struct_elem=None):
        super(MorphologicalFilterTask, self).__init__(feature_type, feature_name)

        if isinstance(morph_operation, MorphologicalOperations):
            self.morph_operation = MorphologicalOperations.get_operation(morph_operation)
        else:
            self.morph_operation = morph_operation
        self.struct_elem = struct_elem

    def process(self, raster):
        """ Applies the morphological operation to the mask object
        """
        dim = len(raster.shape)
        if dim == 3:
            for dim in range(raster.shape[2]):
                raster[:, :, dim] = self.morph_operation(raster[:, :, dim], self.struct_elem)
        elif dim == 4:
            for time, dim in it.product(range(raster.shape[0]), range(raster.shape[3])):
                raster[time, :, :, dim] = self.morph_operation(raster[time, :, :, dim], self.struct_elem)
        else:
            raise ValueError('Invalid number of dimensions: {}'.format(dim))

        return raster
