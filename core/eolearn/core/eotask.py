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
import copy
import inspect
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from .utilities import FeatureParser

LOGGER = logging.getLogger(__name__)


class EOTask(ABC):

    """Base class for EOTask."""
    def __new__(cls, *args, **kwargs):
        """Stores initialization parameters and the order to the instance attribute `init_args`."""
        self = super().__new__(cls)

        init_args = {}
        for arg, value in zip(inspect.getfullargspec(self.__init__).args[1: len(args) + 1], args):
            init_args[arg] = repr(value)
        for arg in inspect.getfullargspec(self.__init__).args[len(args) + 1:]:
            if arg in kwargs:
                init_args[arg] = repr(kwargs[arg])

        self._private_task_config = _PrivateTaskConfig(init_args=init_args, uid=_generate_uid(self))

        return self

    @property
    def private_task_config(self):
        """ Keeps track of the arguments for which the task was initialized as well as the original object id.

        The object id is kept to help with serialization issues. Tasks created in different sessions have a small
        chance of having an id clash. For this reason all tasks of a workflow should be created in the same session.

        :return: The initial configuration arguments of the task
        :rtype: _PrivateTaskConfig
        """
        return self._private_task_config

    def __call__(self, *eopatches, **kwargs):
        """ Executes the task and handles proper error propagation
        """
        try:
            return_value = self.execute(*eopatches, **kwargs)
        except BaseException as exception:
            traceback = sys.exc_info()[2]

            # Some special exceptions don't accept an error message as a parameter and raise a TypeError in such case.
            try:
                errmsg = f'During execution of task {self.__class__.__name__}: {exception}'
                extended_exception = type(exception)(errmsg)
            except TypeError:
                extended_exception = exception

            raise extended_exception.with_traceback(traceback)
        return return_value

    def __copy__(self):
        cls = self.__class__
        copied_task = cls.__new__(cls)
        copied_task.__dict__.update(self.__dict__)
        copied_task._private_task_config = _PrivateTaskConfig(
            init_args=self.private_task_config.init_args, uid=_generate_uid(copied_task))
        return copied_task

    def __deepcopy__(self, memo):
        cls = self.__class__
        copied_task = cls.__new__(cls)
        memo[id(self)] = copied_task

        for key, val in self.__dict__.items():
            setattr(copied_task, key, copy.deepcopy(val, memo))

        copied_task._private_task_config = _PrivateTaskConfig(
            init_args=self.private_task_config.init_args, uid=_generate_uid(copied_task))
        return copied_task

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


@dataclass(frozen=True)
class _PrivateTaskConfig:
    """ A container for configuration parameters about an EOTask itself

    :param uid: An ID of a task when it is created
    :param init_args: A dictionary of parameters and values used for EOTask initialization
    """
    uid: str
    init_args: Dict[str, object]


def _generate_uid(task):
    """ Generates a unique ID of a task

    The ID is composed from task name, a hexadecimal string obtained from the current time and a random hexadecimal
    string. This ensures
    """
    time_uid = uuid.uuid1(node=0).hex[:-12]
    random_uid = uuid.uuid4().hex[:12]
    return f'{task.__class__.__name__}-{time_uid}-{random_uid}'
