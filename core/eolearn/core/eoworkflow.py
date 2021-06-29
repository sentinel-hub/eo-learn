"""
The eoworkflow module, together with eotask and eodata, provides core building blocks for specifying and executing
workflows.

A workflow is a directed (acyclic) graph composed of instances of EOTask objects. Each task may take as input the
results of other tasks and external arguments. The external arguments are passed anew each time the workflow is
executed. The workflow builds the computational graph, performs dependency resolution, and executes the tasks.
If the input graph is cyclic, the workflow raises a `CyclicDependencyError`.

The result of a workflow execution is an immutable mapping from tasks to results. The result contains tasks with
zero out-degree (i.e. terminal tasks).

The workflow can be exported to a DOT description language and visualized.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import collections
import logging
import copy
import datetime as dt
from typing import Dict
from dataclasses import dataclass

import attr

from .eotask import EOTask
from .eoworkflow_tasks import OutputTask
from .graph import DirectedGraph


LOGGER = logging.getLogger(__name__)


class CyclicDependencyError(ValueError):
    """ This error is raised when trying to initialize `EOWorkflow` with a cyclic dependency graph
    """


class EOWorkflow:
    """ A basic eo-learn object for building workflows from a list of task dependencies

    Example:

        .. code-block:: python

            workflow = EOWorkflow([  # task1, task2, task3 are initialized EOTasks
                (task1, [], 'My first task'),
                (task2, []),
                (task3, [task1, task2], 'Task that depends on previous 2 tasks')
            ])
    """
    def __init__(self, dependencies):
        """
        :param dependencies: A list of dependencies between tasks, specifying the computational graph.
        :type dependencies: list(tuple or Dependency)
        """
        dependencies = self._parse_dependencies(dependencies)
        self._uid_dict = self._make_uid_dict(dependencies)
        self.dag = self._create_dag(dependencies)
        self._dependencies = self._schedule_dependencies(self.dag)

    @staticmethod
    def _parse_dependencies(dependencies):
        """ Parses dependencies into correct form

        :param dependencies: List of inputs that define of dependencies
        :type dependencies: list(tuple or Dependency)
        :return: List of dependencies
        :rtype: list(Dependency)
        """
        return [copy.copy(dep) if isinstance(dep, Dependency) else Dependency(*dep) for dep in dependencies]

    @staticmethod
    def _make_uid_dict(dependencies):
        """ Creates a dictionary mapping task IDs to task dependency while checking uniqueness of tasks

        :param dependencies: The list of dependencies between tasks defining the computational graph
        :type dependencies: list(Dependency)
        :return: A dictionary mapping task IDs to dependencies
        :rtype: dict(int: Dependency)
        """
        uid_dict = {}
        for dep in dependencies:
            uid = dep.task.private_task_config.uid
            if uid in uid_dict:
                raise ValueError(f'EOWorkflow cannot execute the same instance of EOTask {dep.task.__class__.__name__}'
                                 ' multiple times')
            uid_dict[uid] = dep

        return uid_dict

    def _create_dag(self, dependencies):
        """ Creates a directed graph from dependencies

        :param dependencies: A list of Dependency objects
        :type dependencies: list(Dependency)
        :return: A directed graph of the workflow
        :rtype: DirectedGraph
        """
        dag = DirectedGraph()
        for dep in dependencies:
            for vertex in dep.inputs:
                task_uid = vertex.private_task_config.uid
                if task_uid not in self._uid_dict:
                    raise ValueError(f'Task {vertex.__class__.__name__}, which is an input of a task {dep.name}, is not'
                                     ' part of the defined workflow')
                dag.add_edge(self._uid_dict[task_uid], dep)
            if not dep.inputs:
                dag.add_vertex(dep)
        return dag

    @staticmethod
    def _schedule_dependencies(dag):
        """ Computes an ordering < of tasks so that for any two tasks t and t' we have that if t depends on t' then
        t' < t. In words, all dependencies of a task precede the task in this ordering.

        :param dag: A directed acyclic graph representing dependencies between tasks.
        :type dag: DirectedGraph
        :return: A list of topologically ordered dependencies
        :rtype: list(Dependency)
        """
        in_degrees = dag.get_indegrees()

        independent_vertices = collections.deque([vertex for vertex in dag if dag.get_indegree(vertex) == 0])
        topological_order = []
        while independent_vertices:
            v_vertex = independent_vertices.popleft()
            topological_order.append(v_vertex)

            for u_vertex in dag[v_vertex]:
                in_degrees[u_vertex] -= 1
                if in_degrees[u_vertex] == 0:
                    independent_vertices.append(u_vertex)

        if len(topological_order) != len(dag):
            raise CyclicDependencyError('Tasks form a cyclic graph')

        return topological_order

    def execute(self, input_args=None):
        """ Executes the workflow

        :param input_args: External input arguments to the workflow. They have to be in a form of a dictionary where
            each key is an EOTask used in the workflow and each value is a dictionary or a tuple of arguments.
        :type input_args: dict(EOTask: dict(str: object) or tuple(object))
        :return: An immutable mapping containing results of terminal tasks
        :rtype: WorkflowResults
        """
        out_degs = self.dag.get_outdegrees()

        self.validate_input_args(input_args)
        uid_input_args = self._make_uid_input_args(input_args)

        results = self._execute_tasks(uid_input_args=uid_input_args, out_degs=out_degs)

        LOGGER.debug('Workflow finished with %s', repr(results))
        return results

    @staticmethod
    def validate_input_args(input_args):
        """ Validates EOWorkflow input arguments provided by user and raises an error if something is wrong.

        :param input_args: A dictionary mapping tasks to task execution arguments
        :type input_args: dict
        """
        input_args = input_args or {}

        for task, args in input_args.items():
            if not isinstance(task, EOTask):
                raise ValueError(f'Invalid input argument {task}, should be an instance of EOTask')

            if not isinstance(args, (tuple, dict)):
                raise ValueError('Execution input arguments of each task should be a dictionary or a tuple, for task '
                                 f'{task.__class__.__name__} got arguments of type {type(args)}')

    @staticmethod
    def _make_uid_input_args(input_args):
        """ Parses EOWorkflow input arguments, switching keys from tasks to task uids to avoid serialization issues.

        :param input_args: A dictionary mapping tasks to task execution arguments
        :type input_args: dict
        :return: A dictionary mapping task uids to task execution arguments
        :rtype: dict
        """
        input_args = input_args or {}
        return {task.private_task_config.uid: args for task, args in input_args.items()}

    def _execute_tasks(self, *, uid_input_args, out_degs):
        """ Executes tasks comprising the workflow in the predetermined order

        :param uid_input_args: External input arguments to the workflow.
        :type uid_input_args: Dict
        :param out_degs: Dictionary mapping vertices (task IDs) to their out-degrees. (The out-degree equals the number
        of tasks that depend on this task.)
        :type out_degs: Dict
        :return: An object with results of a workflow
        :rtype: WorkflowResults
        """
        intermediate_results = {}
        output_results = {}
        stats_dict = {}

        for dep in self._dependencies:
            result, stats = self._execute_task(dependency=dep,
                                               uid_input_args=uid_input_args,
                                               intermediate_results=intermediate_results)

            intermediate_results[dep] = result
            if isinstance(dep.task, OutputTask):
                output_results[dep.task.name] = result

            stats_dict[dep.task.private_task_config.uid] = stats

            self._relax_dependencies(dependency=dep,
                                     out_degrees=out_degs,
                                     intermediate_results=intermediate_results)

        return WorkflowResults(outputs=output_results, stats=stats_dict)

    def _execute_task(self, *, dependency, uid_input_args, intermediate_results):
        """ Executes a task of the workflow

        :param dependency: A workflow dependency
        :type dependency: Dependency
        :param uid_input_args: External task parameters.
        :type uid_input_args: dict
        :param intermediate_results: The dictionary containing intermediate results, including the results of all
        tasks that the current task depends on.
        :type intermediate_results: dict
        :return: The result of the task in dependency
        :rtype: (object, TaskStats)
        """
        task = dependency.task
        inputs = tuple(intermediate_results[self._uid_dict[input_task.private_task_config.uid]]
                       for input_task in dependency.inputs)

        kw_inputs = uid_input_args.get(task.private_task_config.uid, {})
        if isinstance(kw_inputs, tuple):
            inputs += kw_inputs
            kw_inputs = {}

        LOGGER.debug("Computing %s(*%s, **%s)", task.__class__.__name__, str(inputs), str(kw_inputs))
        start_time = dt.datetime.now()
        result = task(*inputs, **kw_inputs)
        end_time = dt.datetime.now()

        return result, TaskStats(start_time=start_time, end_time=end_time)

    def _relax_dependencies(self, *, dependency, out_degrees, intermediate_results):
        """ Relaxes dependencies incurred by ``task_id``. After the task with ID ``task_id`` has been successfully
        executed, all the tasks it depended on are updated. If ``task_id`` was the last remaining dependency of a task
        ``t`` then ``t``'s result is removed from memory and, depending on ``remove_intermediate``, from disk.

        :param dependency: A workflow dependency
        :type dependency: Dependency
        :param out_degrees: Out-degrees of tasks
        :type out_degrees: dict
        :param intermediate_results: The dictionary containing the intermediate results (needed by tasks that have yet
        to be executed) of the already-executed tasks
        :type intermediate_results: dict
        """
        for input_task in [dependency.task] + dependency.inputs:
            dep = self._uid_dict[input_task.private_task_config.uid]

            if input_task is not dependency.task:
                out_degrees[dep] -= 1

            if out_degrees[dep] == 0:
                LOGGER.debug('Removing intermediate result of %s', input_task.__class__.__name__)
                del intermediate_results[dep]

    def get_tasks(self):
        """ Returns an ordered dictionary {task_name: task} of all tasks within this workflow

        :return: Ordered dictionary with key being task_name (str) and an instance of a corresponding task from this
            workflow. The order of tasks is the same as in which they will be executed.
        :rtype: OrderedDict
        """
        task_dict = collections.OrderedDict()
        for dep in self._dependencies:
            task_name = dep.name

            if task_name in task_dict:
                count = 0
                while dep.get_custom_name(count) in task_dict:
                    count += 1

                task_name = dep.get_custom_name(count)

            task_dict[task_name] = dep.task

        return task_dict

    def get_dot(self):
        """ Generates the DOT description of the underlying computational graph

        :return: The DOT representation of the computational graph
        :rtype: Digraph
        """
        visualization = self._get_visualization()
        return visualization.get_dot()

    def dependency_graph(self, filename=None):
        """ Visualize the computational graph

        :param filename: Filename of the output image together with file extension. Supported formats: `png`, `jpg`,
            `pdf`, ... . Check `graphviz` Python package for more options
        :type filename: str
        :return: The DOT representation of the computational graph, with some more formatting
        :rtype: Digraph
        """
        visualization = self._get_visualization()
        return visualization.dependency_graph(filename=filename)

    def _get_visualization(self):
        """ Helper method which provides EOWorkflowVisualization object
        """
        # pylint: disable=import-outside-toplevel,raise-missing-from
        try:
            from eolearn.visualization import EOWorkflowVisualization
        except ImportError:
            raise RuntimeError('Subpackage eo-learn-visualization has to be installed in order to use EOWorkflow '
                               'visualization methods')
        return EOWorkflowVisualization(self._dependencies, self._uid_dict)


class LinearWorkflow(EOWorkflow):
    """ A linear version of EOWorkflow where each tasks only gets results of the previous task

    Example:

        .. code-block:: python

            workflow = LinearWorkflow(task1, task2, task3)
    """
    def __init__(self, *tasks, **kwargs):
        """
        :param tasks: Tasks in the order of execution. Each entry can either be an instance of EOTask or a tuple of
            an EOTask instance and a custom task name.
        :type tasks: EOTask or (EOTask, str)
        """
        tasks = [self._parse_task(task) for task in tasks]

        dependencies = [(task, [tasks[idx - 1][0]] if idx > 0 else [], name) for idx, (task, name) in enumerate(tasks)]
        super().__init__(dependencies, **kwargs)

    @staticmethod
    def _parse_task(task):
        """ Parses input task
        """
        if isinstance(task, EOTask):
            return task, None
        if isinstance(task, (tuple, list)) and len(task) == 2:
            return task

        raise ValueError(f'Cannot parse {task}, expected an instance of EOTask or a tuple (EOTask, name)')


@attr.s(eq=False)  # eq=False preserves the original hash
class Dependency:
    """ Class representing a node in EOWorkflow graph

    :param task: An instance of EOTask
    :type task: EOTask
    :param inputs: A list of EOTask instances which are dependencies of the given `task`
    :type inputs: list(EOTask) or EOTask
    :param name: Name of the Dependency node
    :type name: str or None
    """
    task = attr.ib(default=None)  # validator parameter could be used, but its error msg is ugly
    inputs = attr.ib(factory=list)
    name = attr.ib(default=None)

    def __attrs_post_init__(self):
        """ This is executed right after init method
        """
        if not isinstance(self.task, EOTask):
            raise ValueError(f'Value {self.task} should be an instance of {EOTask.__name__}')
        self.task = self.task

        if isinstance(self.inputs, EOTask):
            self.inputs = [self.inputs]
        if not isinstance(self.inputs, (list, tuple)):
            raise ValueError(f'Value {self.inputs} should be a list')
        for input_task in self.inputs:
            if not isinstance(input_task, EOTask):
                raise ValueError(f'Value {input_task} should be an instance of {EOTask.__name__}')

        if self.name is None:
            self.name = self.task.__class__.__name__

    def get_custom_name(self, number=0):
        """ Provides custom task name according to given number. E.g. FooTask -> FooTask
        """
        if number:
            return f'{self.name}_{number}'
        return self.name


@dataclass(frozen=True)
class TaskStats:
    """ An object containing statistical info about a task execution
    """
    start_time: dt.datetime
    end_time: dt.datetime


@dataclass(frozen=True)
class WorkflowResults:
    """ An object containing results of an EOWorkflow execution
    """
    outputs: Dict[str, object]
    stats: Dict[str, TaskStats]

