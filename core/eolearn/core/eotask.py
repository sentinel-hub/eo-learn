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
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional
from dataclasses import dataclass

from eolearn.core.constants import FeatureType

from .utilities import FeatureParser, parse_feature, parse_renamed_feature, parse_features, parse_renamed_features

LOGGER = logging.getLogger(__name__)


class EOTask(ABC):
    """ Base class for EOTask
    """
    parse_feature = staticmethod(parse_feature)
    parse_renamed_feature = staticmethod(parse_renamed_feature)
    parse_features = staticmethod(parse_features)
    parse_renamed_features = staticmethod(parse_renamed_features)

    def __new__(cls, *args, **kwargs):
        """ Stores initialization parameters and the order to the instance attribute `init_args`.
        """
        self = super().__new__(cls)

        init_args = {}
        for arg, value in zip(inspect.getfullargspec(self.__init__).args[1: len(args) + 1], args):
            init_args[arg] = repr(value)
        for arg in inspect.getfullargspec(self.__init__).args[len(args) + 1:]:
            if arg in kwargs:
                init_args[arg] = repr(kwargs[arg])

        self._private_task_config = _PrivateTaskConfig(init_args=init_args)

        return self

    @property
    def private_task_config(self):
        """ Keeps track of the arguments for which the task was initialized for better logging.

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

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        """ Implement execute function
        """
        raise NotImplementedError

    @staticmethod
    def get_feature_parser(features, allowed_feature_types: Optional[Iterable[FeatureType]] = None) -> FeatureParser:
        """ See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        return FeatureParser(features, allowed_feature_types=allowed_feature_types)


@dataclass(frozen=True)
class _PrivateTaskConfig:
    """ A container for configuration parameters about an EOTask itself

    :param init_args: A dictionary of parameters and values used for EOTask initialization
    """
    init_args: Dict[str, object]
