"""
This module implements the core class hierarchy for implementing EO tasks. An EO task is any class the inherits
from the abstract EOTask class. Each EO task has to implement the execute method; invoking __call__ on a EO task
instance invokes the execute method. EO tasks are meant primarily to operate on EO patches (i.e. instances of EOPatch).

EO task classes are generally lightweight (i.e. not too complicated), short, and do one thing well. For example, an
EO task might take as input an EOPatch containing cloud mask and return as a result the cloud coverage for that mask.
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
    """Base class for EOTask."""
    def __new__(cls, *args, **kwargs):
        """Stores initialization parameters and the order to the instance attribute `init_args`."""
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
        """Creates a composite task of this and passed task."""
        return CompositeTask(other, self)

    def __call__(self, *eopatches, monitor=False, **kwargs):
        """Executes the task."""
        if monitor:
            return self.execute_and_monitor(*eopatches, **kwargs)

        return self.execute(*eopatches, **kwargs)

    @staticmethod
    def _parse_features(features, new_names=False, default_feature_type=None, rename_function=None):
        """See FeatureParser class."""
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
    """Creates a task that is composite of two tasks.

    It takes as input to the constructor a list of EOTask instances, say [t1,t2,...,tn]. The result of invoking the
    execute (or __call__) on the chained task is, by definition, the same as invoking
    tn(...(t2(t1(*args, **kwargs)))...). The chained task does lazy evaluation: it performs no work until execute
    has been invoked.

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
