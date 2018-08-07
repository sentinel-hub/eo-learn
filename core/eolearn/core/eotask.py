"""
This module implements the core class hierarchy for implementing EO tasks. An EO task is any class the inherits
from the abstract EOTask class. Each EO task has to implement the execute method; invoking __call__ on a EO task
instance invokes the execute method. EO tasks are meant primarily to operate on EO patches (i.e. instances of EOPatch).

EO task classes are generally lightweight (i.e. not too complicated), short, and do one thing well. For example, an
EO task might take as input an EOPatch containing cloud mask and return as a result the cloud coverage for that mask.

Beside the base EOTask the hierarchy provides several additional general-purpose tasks that deserve a mention:

 - ChainedTask class represents a compositum of EO tasks. It takes as input to the constructor a list of EOTask
 instances, say [t1,t2,...,tn]. The result of invoking the execute (or __call__) on the chained task is, by definition,
 the same as invoking tn(...(t2(t1(*args, **kwargs)))...). The chained task does lazy evaluation: it performs no
 work until execute has been invoked.

 - EOSimpleFilterTask takes a predicate---a callable t that maps EOPatch to bool---and additional EOPatch-specific
 parameters, and filters all slices of the input EOPatch that don't conform to the predicate.

 - EOFeatureExtractionTask adds a feature to EOPatch by executing a function over a field of an EOPatch.

 - EORegistration task is specific to image registration algorithms implemented in a separate package
 eolearn.registration
"""

import logging

from abc import ABC, abstractmethod
from inspect import getfullargspec
from copy import deepcopy
from collections import OrderedDict

from .feature_types import FeatureType
from .utilities import FeatureParser

LOGGER = logging.getLogger(__name__)


class EOTask(ABC):
    """
    Base class for EOTask.
    """
    def __new__(cls, *args, **kwargs):
        """Here an instance of EOTask is created. All initialization parameters are deep copied and stored to a special
        instance attribute `init_args`. Order of arguments is also preserved.
        """
        self = super().__new__(cls)

        self.init_args = OrderedDict()
        for arg, value in zip(getfullargspec(self.__init__).args[1: len(args) + 1], args):
            self.init_args[arg] = deepcopy(value)
        for arg in getfullargspec(self.__init__).args[len(args) + 1:]:
            if arg in kwargs:
                self.init_args[arg] = deepcopy(kwargs[arg])

        self.uuid = None

        return self

    def __mul__(self, other):
        """ Composite of EOTasks
        """
        return CompositeTask(other, self)

    def __call__(self, *eopatches, **kwargs):
        """ EOTask is callable like a function
        """
        return self.execute(*eopatches, **kwargs)

    @staticmethod
    def _parse_features(features, new_names=False):
        """

        :param features: A collection of features in one of the supported formats
        :type features: object
        :param new_names: `True` if a collection
        :type new_names: bool
        :return: An iterator
        :rtype: FeatureParser
        :raises: ValueError
        """
        return FeatureParser(features, new_names=new_names)

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        raise NotImplementedError


class CompositeTask(EOTask):
    """ Creates a task that is composite of two tasks

    :param eotask1: Task which will be executed first
    :type eotask1: EOTask
    :param eotask2: Task which will be executed on results of first task
    :type eotask2: EOTask
    """
    def __init__(self, eotask1, eotask2):
        self.eotask1 = eotask1
        self.eotask2 = eotask2

        self.init_args = OrderedDict(list(eotask1.init_args.items()) + list(eotask2.init_args.items()))

    def execute(self, *eopatches, **kwargs):
        return self.eotask2.execute(self.eotask1.execute(*eopatches, **kwargs))


class FeatureTask(EOTask):
    """ An abstract class of EOTask that manipulates a single feature

    :param feature_type: Type of the feature
    :type feature_type: FeatureType
    :param feature_name: Name of the feature
    :type feature_name: str
    """
    def __init__(self, feature_type, feature_name=None):
        self.feature_type = FeatureType(feature_type)
        self.feature_name = feature_name

        if feature_name is not None and not feature_type.has_dict():
            raise ValueError('{} does not store a dictionary of features'.format(feature_type))

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        raise NotImplementedError
