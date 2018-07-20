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

import functools
import logging
from abc import ABC, abstractmethod

from .eodata import FeatureType

LOGGER = logging.getLogger(__name__)


class EOTask(ABC):
    """
    Base class for task.
    """
    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        raise NotImplementedError

    def __call__(self, *eopatches, **kwargs):
        return self.execute(*eopatches, **kwargs)


class ChainedTask(EOTask):
    """
    A sequence of tasks  is a task: Tn(...T2(T1(eopatch))...) = T(eopatch) for T = Tn o ... o T2 o T1.

    This class represents a composite of a sequence of tasks.
    """
    def __init__(self, tasks):
        """
        :param tasks: A list of tasks to be chained together.
        :type tasks: List[EOTask]
        """
        # pylint: disable=undefined-variable
        self.tasks = list(reversed(tasks))
        self.compositum = functools.reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)), self.tasks)

    def get_tasks(self):
        """
        Returns the list of tasks whose compositum we're applying.
        :return: The list of tasks the chain is composed of
        :rtype: List[EOTask]
        """
        return self.tasks

    def execute(self, *eopatch, **kwargs):
        """
        Applies the composition of tasks to ``eopatch``.
        :param eopatch: Input EOpatch.
        :type eopatch: positional EOPatch arguments
        :param kwargs: keyword arguments (used i.e. with workflow)
        :return: Transformed tuple of patches. To each EOPatch we apply: ``Tn(...T2(T1(eopatch))...)``
        :rtype: EOPatch
        """
        return self.compositum(*eopatch, **kwargs)


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
