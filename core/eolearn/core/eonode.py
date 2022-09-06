"""
This module implements the EONode class, which specifies the local dependencies of an EOWorkflow

Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Matej Batič, Miha Kadunc, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union, cast

from .eotask import EOTask
from .utils.common import generate_uid


@dataclass(frozen=True)
class EONode:
    """Class representing a node in EOWorkflow graph.

    The object id is kept to help with serialization issues. Tasks created in different sessions have a small chance
    of having an id clash. For this reason all tasks of a workflow should be created in the same session.

    :param task: An instance of `EOTask` that is carried out at the node when executed
    :param inputs: A sequence of `EONode` instances whose results this node takes as input
    :param name: Custom name of the node
    """

    task: EOTask
    inputs: Sequence["EONode"] = field(default_factory=tuple)
    name: Optional[str] = field(default=None)
    uid: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Additionally verifies the parameters and adds a unique id to the node."""
        if not isinstance(self.task, EOTask):
            raise ValueError(f"Value of `task` should be an instance of {EOTask.__name__}, got {self.task}.")

        if not isinstance(self.inputs, Sequence):
            raise ValueError(f"Value of `inputs` should be a sequence (`list`, `tuple`, ...), got {self.inputs}.")
        for input_node in self.inputs:
            if not isinstance(input_node, EONode):
                raise ValueError(f"Values in `inputs` should be instances of {EONode.__name__}, got {input_node}.")
        super().__setattr__("inputs", tuple(self.inputs))

        if self.name is None:
            super().__setattr__("name", self.task.__class__.__name__)

        super().__setattr__("uid", generate_uid(self.task.__class__.__name__))

    def __hash__(self) -> int:
        return self.uid.__hash__()

    def get_name(self, suffix_number: int = 0) -> str:
        """Provides node name according to the class of the contained task and a given number."""
        if suffix_number:
            return f"{self.name}_{suffix_number}"
        return cast(str, self.name)

    def get_dependencies(self, *, _memo: Optional[Dict["EONode", Set["EONode"]]] = None) -> Set["EONode"]:
        """Returns a set of nodes that this node depends on. Set includes the node itself."""
        _memo = _memo if _memo is not None else {}
        if self not in _memo:
            result = {self}.union(*(input_node.get_dependencies(_memo=_memo) for input_node in self.inputs))
            _memo[self] = result

        return _memo[self]


def linearly_connect_tasks(*tasks: Union[EOTask, Tuple[EOTask, str]]) -> List[EONode]:
    """Creates a list of linearly linked nodes, suitable to construct an EOWorkflow.

    Nodes depend on each other in such a way, that the node containing the task at index `i` is the input node for the
    node at index `i+1`. Nodes are returned in the order of execution, so the task at index `j` is contained in the node
    at index `j`, making it easier to construct execution arguments.

    :param tasks: A sequence containing tasks and/or (task, name) pairs
    """
    nodes = []
    endpoint: Sequence[EONode] = tuple()
    for task_spec in tasks:
        if isinstance(task_spec, EOTask):
            node = EONode(task_spec, inputs=endpoint)
        else:
            task, name = task_spec
            node = EONode(task, inputs=endpoint, name=name)
        nodes.append(node)
        endpoint = [node]

    return nodes


@dataclass(frozen=True)
class NodeStats:
    """An object containing statistical info about a node execution."""

    node_uid: str
    node_name: str
    start_time: dt.datetime
    end_time: dt.datetime
    exception: Optional[BaseException] = None
    exception_traceback: Optional[str] = None
