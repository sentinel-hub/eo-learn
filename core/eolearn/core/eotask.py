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
import datetime

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

    def __call__(self, *eopatches, monitor=False, **kwargs):
        """ EOTask is callable like a function
        """
        if monitor:
            return self.execute_and_monitor(*eopatches, **kwargs)

        return self.execute(*eopatches, **kwargs)

    @staticmethod
    def _parse_features(features, new_names=False, default_feature_type=None, rename_function=None):
        """Used for parsing input features

        :param features: A collection of features in one of the supported formats
        :type features: object
        :param new_names: `True` if a collection
        :type new_names: bool
        :param default_feature_type: If feature type of any of the given features is not set this will be used
        :type default_feature_type: FeatureType or None
        :param rename_function: Default renaming function
        :type rename_function: function or None
        :return: A generator over feature types and feature names
        :rtype: FeatureParser
        :raises: ValueError
        """
        return FeatureParser(features, new_names=new_names, default_feature_type=default_feature_type,
                             rename_function=rename_function)

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        raise NotImplementedError

    def execute_and_monitor(self, *eopatches, **kwargs):
        setattr(self, 'start_time', datetime.datetime.now())
        retval = self.execute(*eopatches, **kwargs)
        setattr(self, 'end_time', datetime.datetime.now())

        return retval


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
