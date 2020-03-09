"""
This module implements the core class hierarchy for implementing EO tasks. An EO task is any class the inherits
from the abstract EOTask class. Each EO task has to implement the execute method; invoking __call__ on a EO task
instance invokes the execute method. EO tasks are meant primarily to operate on EO patches (i.e. instances of EOPatch).

EO task classes are generally lightweight (i.e. not too complicated), short, and do one thing well. For example, an
EO task might take as input an EOPatch containing cloud mask and return as a result the cloud coverage for that mask.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import sys
import logging
import datetime
import inspect
from collections import OrderedDict
from abc import ABC, abstractmethod

import attr

from .utilities import FeatureParser

LOGGER = logging.getLogger(__name__)


class EOTask(ABC):

    """Base class for EOTask."""
    def __new__(cls, *args, **kwargs):
        """Stores initialization parameters and the order to the instance attribute `init_args`."""
        self = super().__new__(cls)

        init_args = OrderedDict()
        for arg, value in zip(inspect.getfullargspec(self.__init__).args[1: len(args) + 1], args):
            init_args[arg] = repr(value)
        for arg in inspect.getfullargspec(self.__init__).args[len(args) + 1:]:
            if arg in kwargs:
                init_args[arg] = repr(kwargs[arg])

        self.private_task_config = _PrivateTaskConfig(init_args=init_args)

        return self

    def __mul__(self, other):
        """Creates a composite task of this and passed task."""
        return CompositeTask(other, self)

    def __call__(self, *eopatches, monitor=False, **kwargs):
        """Executes the task."""
        # if monitor:
        #     return self.execute_and_monitor(*eopatches, **kwargs)

        return self._execute_handling(*eopatches, **kwargs)

    def execute_and_monitor(self, *eopatches, **kwargs):
        """ In the current version nothing additional happens in this method
        """
        return self._execute_handling(*eopatches, **kwargs)

    def _execute_handling(self, *eopatches, **kwargs):
        """ Handles measuring execution time and error propagation
        """
        self.private_task_config.start_time = datetime.datetime.now()

        try:
            return_value = self.execute(*eopatches, **kwargs)
            self.private_task_config.end_time = datetime.datetime.now()
            return return_value
        except BaseException as exception:
            traceback = sys.exc_info()[2]

            # Some special exceptions don't accept an error message as a parameter and raise a TypeError in such case.
            try:
                errmsg = 'During execution of task {}: {}'.format(self.__class__.__name__, exception)
                extended_exception = type(exception)(errmsg)
            except TypeError:
                extended_exception = exception

            raise extended_exception.with_traceback(traceback)

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        """ Implement execute function
        """
        raise NotImplementedError

    @staticmethod
    def _parse_features(features, new_names=False, rename_function=None, default_feature_type=None,
                        allowed_feature_types=None):
        """ See eolearn.core.utilities.FeatureParser class.
        """
        return FeatureParser(features, new_names=new_names, rename_function=rename_function,
                             default_feature_type=default_feature_type, allowed_feature_types=allowed_feature_types)


@attr.s(cmp=False)
class _PrivateTaskConfig:
    """ A container for general EOTask parameters required during EOWorkflow and EOExecution

    :param init_args: A dictionary of parameters and values used for EOTask initialization
    :type init_args: OrderedDict
    :param uuid: An unique hexadecimal identifier string a task gets in EOWorkflow
    :type uuid: str or None
    :param start_time: Time when task execution started
    :type start_time: datetime.datetime or None
    :param end_time: Time when task execution ended
    :type end_time: datetime.datetime or None
    """
    init_args = attr.ib()
    uuid = attr.ib(default=None)
    start_time = attr.ib(default=None)
    end_time = attr.ib(default=None)

    def __add__(self, other):
        return _PrivateTaskConfig(init_args=OrderedDict(list(self.init_args.items()) + list(other.init_args.items())))


class CompositeTask(EOTask):
    """Creates a task that is composite of two tasks.

    Note: Instead of directly using this task it might be more convenient to use `'*'` operation between tasks.
    Example: `composite_task = task1 * task2`

    :param eotask1: Task which will be executed first
    :type eotask1: EOTask
    :param eotask2: Task which will be executed on results of first task
    :type eotask2: EOTask
    """
    def __init__(self, eotask1, eotask2):
        self.eotask1 = eotask1
        self.eotask2 = eotask2

        self.private_task_config = eotask1.private_task_config + eotask2.private_task_config

    def execute(self, *eopatches, **kwargs):
        return self.eotask2.execute(self.eotask1.execute(*eopatches, **kwargs))
