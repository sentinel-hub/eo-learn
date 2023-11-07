"""
Module implementing tasks that have a special effect in `EOWorkflow`

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from .eodata import EOPatch
from .eotask import EOTask
from .types import FeaturesSpecification
from .utils.common import generate_uid


class InputTask(EOTask):
    """Introduces data into an EOWorkflow, where the data can be specified at initialization or at execution."""

    def __init__(self, value: object | None = None):
        """
        :param value: Default value that the task should provide as a result. Can be overridden in execution arguments
        """
        self.value = value

    def execute(self, *, value: object | None = None) -> object:
        """
        :param value: A value that the task should provide as its result. If not set uses the value from initialization
        :return: Directly returns `value`
        """
        return value or self.value


class OutputTask(EOTask):
    """Stores data as an output of `EOWorkflow` results."""

    def __init__(self, name: str | None = None, features: FeaturesSpecification = ...):
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
        :return: Same data, to be stored in results. For `EOPatch` returns shallow copy containing `features` and
            possibly BBox and timestamps (see `copy` method of `EOPatch`).
        """
        if isinstance(data, EOPatch):
            return data.copy(features=self.features, copy_timestamps=True)
        return data
