"""
This module implements the core class hierarchy for implementing EO tasks. An EO task is any class the inherits
from the abstract EOTask class. Each EO task has to implement the execute method; invoking __call__ on a EO task
instance invokes the execute method. EO tasks are meant primarily to operate on EO patches (i.e. instances of EOPatch).

EO task classes are generally lightweight (i.e. not too complicated), short, and do one thing well. For example, an
EO task might take as input an EOPatch containing cloud mask and return as a result the cloud coverage for that mask.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import inspect
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Type, TypeVar, Union

from .constants import FeatureType
from .types import EllipsisType, FeaturesSpecification
from .utils.parsing import FeatureParser, parse_feature, parse_features, parse_renamed_feature, parse_renamed_features

LOGGER = logging.getLogger(__name__)

Self = TypeVar("Self")


class EOTask(metaclass=ABCMeta):
    """Base class for EOTask."""

    parse_feature = staticmethod(parse_feature)
    parse_renamed_feature = staticmethod(parse_renamed_feature)
    parse_features = staticmethod(parse_features)
    parse_renamed_features = staticmethod(parse_renamed_features)

    def __new__(cls: Type[Self], *args: Any, **kwargs: Any) -> Self:
        """Stores initialization parameters and the order to the instance attribute `init_args`."""
        self = super().__new__(cls)  # type: ignore[misc]

        init_args: Dict[str, object] = {}
        for arg, value in zip(inspect.getfullargspec(self.__init__).args[1 : len(args) + 1], args):
            init_args[arg] = repr(value)
        for arg in inspect.getfullargspec(self.__init__).args[len(args) + 1 :]:
            if arg in kwargs:
                init_args[arg] = repr(kwargs[arg])

        self._private_task_config = _PrivateTaskConfig(init_args=init_args)

        return self

    @property
    def private_task_config(self) -> "_PrivateTaskConfig":
        """Keeps track of the arguments for which the task was initialized for better logging.

        :return: The initial configuration arguments of the task
        """
        return self._private_task_config  # type: ignore[attr-defined]

    def __call__(self, *eopatches, **kwargs):
        """Syntactic sugar for task execution"""
        return self.execute(*eopatches, **kwargs)

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        """Override to specify action performed by task."""

    @staticmethod
    def get_feature_parser(
        features: FeaturesSpecification, allowed_feature_types: Union[Iterable[FeatureType], EllipsisType] = ...
    ) -> FeatureParser:
        """See :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`."""
        return FeatureParser(features, allowed_feature_types=allowed_feature_types)


@dataclass(frozen=True)
class _PrivateTaskConfig:
    """A container for configuration parameters about an EOTask itself.

    :param init_args: A dictionary of parameters and values used for EOTask initialization
    """

    init_args: Dict[str, object]
