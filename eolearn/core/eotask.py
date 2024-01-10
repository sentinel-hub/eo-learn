"""
This module implements the core class hierarchy for implementing EO tasks. An EO task is any class the inherits
from the abstract EOTask class. Each EO task has to implement the execute method; invoking __call__ on a EO task
instance invokes the execute method. EO tasks are meant primarily to operate on EO patches (i.e. instances of EOPatch).

EO task classes are generally lightweight (i.e. not too complicated), short, and do one thing well. For example, an
EO task might take as input an EOPatch containing cloud mask and return as a result the cloud coverage for that mask.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from typing_extensions import deprecated

from .constants import FeatureType
from .eodata import EOPatch
from .exceptions import EODeprecationWarning
from .types import EllipsisType, Feature, FeaturesSpecification, SingleFeatureSpec
from .utils.parsing import FeatureParser, parse_feature, parse_features, parse_renamed_feature, parse_renamed_features

LOGGER = logging.getLogger(__name__)

Self = TypeVar("Self")

PARSE_RENAMED_DEPRECATE_MSG = (
    "The method will no longer be a method of `EOTask`, but can be imported as a function from"
    " `eolearn.core.utils.parsing`."
)


class EOTask(metaclass=ABCMeta):
    """Base class for EOTask."""

    parse_renamed_feature = staticmethod(
        deprecated(PARSE_RENAMED_DEPRECATE_MSG, category=EODeprecationWarning)(parse_renamed_feature)
    )
    parse_renamed_features = staticmethod(
        deprecated(PARSE_RENAMED_DEPRECATE_MSG, category=EODeprecationWarning)(parse_renamed_features)
    )

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """Stores initialization parameters and the order to the instance attribute `init_args`."""
        self = super().__new__(cls)  # type: ignore[misc]

        init_args: dict[str, object] = {}
        for arg, value in zip(inspect.getfullargspec(self.__init__).args[1 : len(args) + 1], args):
            init_args[arg] = repr(value)
        for arg in inspect.getfullargspec(self.__init__).args[len(args) + 1 :]:
            if arg in kwargs:
                init_args[arg] = repr(kwargs[arg])

        self._private_task_config = _PrivateTaskConfig(init_args=init_args)

        return self

    @property
    def private_task_config(self) -> _PrivateTaskConfig:
        """Keeps track of the arguments for which the task was initialized for better logging.

        :return: The initial configuration arguments of the task
        """
        return self._private_task_config  # type: ignore[attr-defined]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Syntactic sugar for task execution"""
        # The type cannot be more precise unless we know the type of `execute`. Possible improvement with generics +
        # the use of ParamSpec.
        return self.execute(*args, **kwargs)

    @abstractmethod
    def execute(self, *eopatches, **kwargs):  # type: ignore[no-untyped-def] # must be ignored so subclasses can change
        """Override to specify action performed by task."""

    @staticmethod
    def parse_feature(
        feature: SingleFeatureSpec,
        eopatch: EOPatch | None = None,
        allowed_feature_types: EllipsisType | Iterable[FeatureType] | Callable[[FeatureType], bool] = ...,
    ) -> Feature:
        """See `eolearn.core.utils.parse_feature`."""
        return parse_feature(feature, eopatch, allowed_feature_types)

    @staticmethod
    def parse_features(
        features: FeaturesSpecification,
        eopatch: EOPatch | None = None,
        allowed_feature_types: EllipsisType | Iterable[FeatureType] | Callable[[FeatureType], bool] = ...,
    ) -> list[Feature]:
        """See `eolearn.core.utils.parse_features`."""
        return parse_features(features, eopatch, allowed_feature_types)

    @staticmethod
    def get_feature_parser(
        features: FeaturesSpecification,
        allowed_feature_types: EllipsisType | Iterable[FeatureType] | Callable[[FeatureType], bool] = ...,
    ) -> FeatureParser:
        """See :class:`FeatureParser<eolearn.core.utils.FeatureParser>`."""
        return FeatureParser(features, allowed_feature_types=allowed_feature_types)


@dataclass(frozen=True)
class _PrivateTaskConfig:
    """A container for configuration parameters about an EOTask itself.

    :param init_args: A dictionary of parameters and values used for EOTask initialization
    """

    init_args: dict[str, object]
