"""
The eoworkflow module, together with eotask and eodata, provides core building blocks for specifying and executing
workflows.

A workflow is a directed (acyclic) graph composed of instances of EONode objects. Each node may take as input the
results of tasks in other nodes as well as external arguments. The external arguments are passed anew each time the
workflow is executed. The workflow builds the computational graph, performs dependency resolution, and executes the
tasks inside each node. If the input graph is cyclic, the workflow raises a `CyclicDependencyError`.

The result of a workflow execution is an immutable mapping from nodes to results. The result also contain data that
was marked as output through the use of `OutputTask` objects.

The workflow can be exported to a DOT description language and visualized.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime as dt
import logging
import traceback
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

from .eodata import EOPatch
from .eonode import EONode, NodeStats
from .eotask import EOTask
from .eoworkflow_tasks import OutputTask
from .graph import DirectedGraph

LOGGER = logging.getLogger(__name__)


class EOWorkflow:
    """An object for verifying and executing workflows defined by inter-dependent `EONodes`.

    Example:

    .. code-block:: python

        node1 = EONode(task1, name='first task')  # custom names can be provided for better logging
        node2 = EONode(task2, inputs=[node1])  # depends on previous task
        node3 = EONode(task3, inputs=[node1])
        node4 = EONode(task4, inputs=[node2, node3])  # depends on two tasks

        workflow = EOWorkflow([node1, node2, node3, node4])

        # One can pass keyword arguments to task execution in the form of a dictionary
        results = workflow.execute(
            {node2: {'k': 2, 'ascending': True}}
        )
    """

    def __init__(self, workflow_nodes: Sequence[EONode]):
        """
        :param workflow_nodes: A sequence of EONodes, specifying the computational graph.
        """
        workflow_nodes = self._parse_and_validate_nodes(workflow_nodes)
        self._uid_dict = self._make_uid_dict(workflow_nodes)
        self.uid_dag = self._create_dag(workflow_nodes)

        topologically_ordered_nodes = self.uid_dag.topologically_ordered_vertices()

        self._nodes = [self._uid_dict[uid] for uid in topologically_ordered_nodes]

    @staticmethod
    def _parse_and_validate_nodes(nodes: Sequence[EONode]) -> Sequence[EONode]:
        """Parses and verifies workflow nodes.

        :param nodes: Sequence of nodes forming a workflow
        :return: Sequence of verified nodes
        """
        if not isinstance(nodes, Sequence):
            raise ValueError(f"{EOWorkflow.__name__} must be initialized with a sequence of {EONode.__name__} objects.")

        for node in nodes:
            if not isinstance(node, EONode):
                raise ValueError(f"Expected a {EONode.__name__} object but got {type(node)}")

        return nodes

    @staticmethod
    def _make_uid_dict(nodes: Sequence[EONode]) -> Dict[str, EONode]:
        """Creates a dictionary mapping node IDs to nodes while checking uniqueness of tasks.

        :param nodes: The sequence of workflow nodes defining the computational graph
        :return: A dictionary mapping IDs to nodes
        """
        uid_dict = {}
        for node in nodes:
            if node.uid in uid_dict:
                raise ValueError(
                    f"EOWorkflow should not contain the same node twice. Found multiple instances of {node}."
                )
            uid_dict[node.uid] = node

        return uid_dict

    def _create_dag(self, nodes: Sequence[EONode]) -> DirectedGraph[str]:
        """Creates a directed graph from workflow nodes that is used for scheduling purposes.

        :param nodes: A sequence of `EONode` objects
        :return: A directed graph of the workflow, with graph nodes containing `EONode` uids
        """
        dag = DirectedGraph[str]()
        for node in nodes:
            for input_node in node.inputs:
                if input_node.uid not in self._uid_dict:
                    raise ValueError(
                        f"Node {input_node}, which is an input of a task {node.get_name()}, is not part of the workflow"
                    )
                dag.add_edge(input_node.uid, node.uid)
            if not node.inputs:
                dag.add_vertex(node.uid)
        return dag

    @classmethod
    def from_endnodes(cls, *endnodes: EONode) -> "EOWorkflow":
        """Constructs the EOWorkflow from the end-nodes by recursively extracting nodes in the workflow structure."""
        all_nodes: Set[EONode] = set()
        memo: Dict[EONode, Set[EONode]] = {}
        for endnode in endnodes:
            all_nodes = all_nodes.union(endnode.get_dependencies(_memo=memo))
        return cls(list(all_nodes))

    def execute(
        self, input_kwargs: Optional[Dict[EONode, Dict[str, object]]] = None, raise_errors: bool = True
    ) -> "WorkflowResults":
        """Executes the workflow.

        :param input_kwargs: External input arguments to the workflow. They have to be in a form of a dictionary where
            each key is an `EONode` used in the workflow and each value is a dictionary or a tuple of arguments.
        :param raise_errors: In case a task in the workflow raises an error this parameter determines how the error
            will be handled. If `True` it will propagate the error and if `False` it will catch the error, write its
            stack trace in logs and in the `WorkflowResults`. In either case workflow execute will stop if an error is
            raised. This rule is not followed only in case of `KeyboardInterrupt` exception where the exception is
            always raised.
        :return: An immutable mapping containing results of terminal tasks
        """
        start_time = dt.datetime.now()

        out_degrees: Dict[str, int] = self.uid_dag.get_outdegrees()

        input_kwargs = input_kwargs or {}
        self.validate_input_kwargs(input_kwargs)
        uid_input_kwargs = {node.uid: args for node, args in input_kwargs.items()}

        output_results, stats_dict = self._execute_nodes(
            uid_input_kwargs=uid_input_kwargs, out_degrees=out_degrees, raise_errors=raise_errors
        )

        results = WorkflowResults(
            outputs=output_results, start_time=start_time, end_time=dt.datetime.now(), stats=stats_dict
        )

        LOGGER.debug("EOWorkflow ended with results %s", repr(results))
        status = "failed" if results.workflow_failed() else "finished"
        LOGGER.debug("EOWorkflow execution %s!", status)
        return results

    @staticmethod
    def validate_input_kwargs(input_kwargs: Dict[EONode, Dict[str, object]]) -> None:
        """Validates EOWorkflow input arguments provided by user and raises an error if something is wrong.

        :param input_kwargs: A dictionary mapping tasks to task execution arguments
        """
        for node, kwargs in input_kwargs.items():
            if not isinstance(node, EONode):
                raise ValueError(
                    f"Keys of the execution argument dictionary should be instances of {EONode.__name__}, got"
                    f" {type(node)} instead."
                )

            if not isinstance(kwargs, dict):
                raise ValueError(
                    "Execution arguments of each node should be a dictionary, for node "
                    f"{node.get_name()} got arguments of type {type(kwargs)}."
                )

            if not all(isinstance(key, str) for key in kwargs):
                raise ValueError(
                    "Keys of input argument dictionaries should names of variables, in arguments for node "
                    f"{node.get_name()} one of the keys is not a string."
                )

    def _execute_nodes(
        self, *, uid_input_kwargs: Dict[str, Dict[str, object]], out_degrees: Dict[str, int], raise_errors: bool
    ) -> Tuple[dict, dict]:
        """Executes workflow nodes in the predetermined order.

        :param uid_input_kwargs: External input arguments to the workflow.
        :param out_degrees: Dictionary mapping node IDs to their out-degrees. (The out-degree equals the number
            of tasks that depend on this task.)
        :return: Results of a workflow
        """
        intermediate_results: Dict[str, object] = {}
        output_results = {}
        stats_dict = {}

        for node in self._nodes:
            result, stats = self._execute_node(
                node=node,
                node_input_values=[intermediate_results[input_node.uid] for input_node in node.inputs],
                node_input_kwargs=uid_input_kwargs.get(node.uid, {}),
                raise_errors=raise_errors,
            )

            stats_dict[node.uid] = stats
            if stats.exception is not None:
                break

            intermediate_results[node.uid] = result
            if isinstance(node.task, OutputTask):
                output_results[node.task.name] = result

            self._relax_dependencies(node=node, out_degrees=out_degrees, intermediate_results=intermediate_results)

        return output_results, stats_dict

    def _execute_node(
        self, *, node: EONode, node_input_values: List[object], node_input_kwargs: Dict[str, object], raise_errors: bool
    ) -> Tuple[object, NodeStats]:
        """Executes a node in the workflow by running its task and returning the results.

        :param node: A node of the workflow.
        :param node_input_values: Values obtained from input nodes in the workflow.
        :param node_input_kwargs: Dictionary containing execution arguments specified by the user.
        :return: The result and statistics of the task in the node.
        """
        # EOPatches are copied beforehand
        task_args = [(arg.copy() if isinstance(arg, EOPatch) else arg) for arg in node_input_values]

        LOGGER.debug("Computing %s(*%s, **%s)", node.task.__class__.__name__, str(task_args), str(node_input_kwargs))
        start_time = dt.datetime.now()
        result, is_success = self._execute_task(node.task, task_args, node_input_kwargs, raise_errors=raise_errors)
        end_time = dt.datetime.now()

        if is_success:
            exception, exception_traceback = None, None
        else:
            exception, exception_traceback = cast(Tuple[BaseException, str], result)  # temporary fix until 3.8
            result = None
            LOGGER.error(
                "Task '%s' with id %s failed with stack trace:\n%s", node.get_name(), node.uid, exception_traceback
            )

        node_stats = NodeStats(
            node_uid=node.uid,
            node_name=node.get_name(),
            start_time=start_time,
            end_time=end_time,
            exception=exception,
            exception_traceback=exception_traceback,
        )
        return result, node_stats

    @staticmethod
    def _execute_task(
        task: EOTask, task_args: List[object], task_kwargs: Dict[str, object], raise_errors: bool
    ) -> Tuple[object, bool]:
        """Executes an EOTask and handles any potential exceptions."""
        if raise_errors:
            return task.execute(*task_args, **task_kwargs), True

        try:
            return task.execute(*task_args, **task_kwargs), True
        except KeyboardInterrupt as exception:
            raise KeyboardInterrupt from exception
        except BaseException as exception:
            exception_traceback = traceback.format_exc()
            return (exception, exception_traceback), False

    @staticmethod
    def _relax_dependencies(
        *, node: EONode, out_degrees: Dict[str, int], intermediate_results: Dict[str, object]
    ) -> None:
        """Relaxes dependencies incurred by `node` after it has been successfully executed. All the nodes it
        depended on are updated. If `node` was the last remaining node depending on a node `n` then `n`'s result
        are removed from memory.

        :param node: A workflow node
        :param out_degrees: Out-degrees of tasks
        :param intermediate_results: The dictionary containing the intermediate results (needed by nodes that have yet
        to be executed) of the already-executed nodes
        """
        for input_node in node.inputs:
            out_degrees[input_node.uid] -= 1

        for relevant_node in {node} | set(node.inputs):
            # use sets in order not to attempt to delete the same node twice
            if out_degrees[relevant_node.uid] == 0:
                LOGGER.debug(
                    "Removing intermediate result of %s (node uid: %s)", relevant_node.get_name(), relevant_node.uid
                )
                del intermediate_results[relevant_node.uid]

    def get_nodes(self) -> List[EONode]:
        """Returns an ordered list of all nodes within this workflow, ordered in the execution order.

        :return: List of all nodes withing workflow. The order of nodes is the same as the order of execution.
        """
        return self._nodes[:]

    def get_node_with_uid(self, uid: Optional[str], fail_if_missing: bool = False) -> Optional[EONode]:
        """Returns node with give uid, if it exists in the workflow."""
        if uid in self._uid_dict:
            return self._uid_dict[uid]
        if fail_if_missing:
            raise KeyError(f"No {EONode.__name__} with uid {uid} found in workflow.")
        return None

    def get_dot(self):
        """Generates the DOT description of the underlying computational graph.

        :return: The DOT representation of the computational graph
        :rtype: Digraph
        """
        visualization = self._get_visualization()
        return visualization.get_dot()

    def dependency_graph(self, filename: Optional[str] = None):
        """Visualize the computational graph.

        :param filename: Filename of the output image together with file extension. Supported formats: `png`, `jpg`,
            `pdf`, ... . Check `graphviz` Python package for more options
        :return: The DOT representation of the computational graph, with some more formatting
        :rtype: Digraph
        """
        visualization = self._get_visualization()
        return visualization.dependency_graph(filename=filename)

    def _get_visualization(self):
        """Helper method which provides EOWorkflowVisualization object."""
        # pylint: disable=import-outside-toplevel,raise-missing-from
        try:
            from eolearn.visualization.eoworkflow import EOWorkflowVisualization
        except ImportError:
            raise RuntimeError(
                "Subpackage eo-learn-visualization has to be installed in order to use EOWorkflow visualization methods"
            )
        return EOWorkflowVisualization(self._nodes)


@dataclass(frozen=True)
class WorkflowResults:
    """An object containing results of an EOWorkflow execution."""

    outputs: Dict[str, object]
    start_time: dt.datetime
    end_time: dt.datetime
    stats: Dict[str, NodeStats]
    error_node_uid: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Checks if there is any node that failed during the workflow execution."""
        for node_uid, node_stats in self.stats.items():
            if node_stats.exception is not None:
                super().__setattr__("error_node_uid", node_uid)
                break

    def workflow_failed(self) -> bool:
        """Informs if the EOWorkflow execution failed."""
        return self.error_node_uid is not None

    def drop_outputs(self) -> "WorkflowResults":
        """Creates a new WorkflowResults object without outputs which can take a lot of memory."""
        new_params = {
            param.name: {} if param.name == "outputs" else getattr(self, param.name)
            for param in fields(self)
            if param.init
        }
        return WorkflowResults(**new_params)
