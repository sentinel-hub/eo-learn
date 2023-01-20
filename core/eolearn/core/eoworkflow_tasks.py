"""
Module implementing tasks that have a special effect in `EOWorkflow`

Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Matej Batič, Miha Kadunc, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from typing import Optional

from .eodata import EOPatch
from .eotask import EOTask
from .types import FeaturesSpecification
from .utils.common import generate_uid


class InputTask(EOTask):
    """Introduces data into an EOWorkflow, where the data can be specified at initialization or at execution."""

    def __init__(self, value: Optional[object] = None):
        """
        :param value: Default value that the task should provide as a result. Can be overridden in execution arguments
        """
        self.value = value

    def execute(self, *, value: Optional[object] = None) -> object:
        """
        :param value: A value that the task should provide as its result. If not set uses the value from initialization
        :return: Directly returns `value`
        """
        return value or self.value


class OutputTask(EOTask):
    """Stores data as an output of `EOWorkflow` results."""

    def __init__(self, name: Optional[str] = None, features: FeaturesSpecification = ...):
        """
        :param name: A name under which the data will be saved in `WorkflowResults`, auto-generated if `None`
        :param features: A collection of features to be kept if the data is an `EOPatch`
        """
        self._name = name or generate_uid("output")
        self.features = features

    @property
    def name(self) -> str:
        """Provides a name under which data will be saved in `WorkflowResults`.

        :return: A name
        """
        return self._name

    def execute(self, data: object) -> object:
        """
        :param data: input data
        :return: Same data, to be stored in results (for `EOPatch` returns shallow copy containing only `features`)
        """
        if isinstance(data, EOPatch):
            return data.copy(features=self.features)
        return data
