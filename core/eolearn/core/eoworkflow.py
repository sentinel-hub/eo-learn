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
import warnings
import uuid
import copy

import attr

from .eotask import EOTask
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
    def __init__(self, dependencies, task_names=None):
        """
        :param dependencies: A list of dependencies between tasks, specifying the computational graph.
        :type dependencies: list(tuple or Dependency)
        """
        self.id_gen = _UniqueIdGenerator()

        if task_names:
            warnings.warn("Parameter 'task_names' will soon be removed. Everything can be specified with "
                          "'dependencies' parameter, including task names", DeprecationWarning, stacklevel=2)

        self.dependencies = self._parse_dependencies(dependencies, task_names)
        self.uuid_dict = self._set_task_uuid(self.dependencies)
        self.dag = self.create_dag(self.dependencies)
        self.ordered_dependencies = self._schedule_dependencies(self.dag)

    @staticmethod
    def _parse_dependencies(dependencies, task_names):
        """ Parses dependencies and adds names of task_names

        :param dependencies: Input of dependency parameter
        :type dependencies: list(tuple or Dependency)
        :param task_names: Human-readable names of tasks
        :type task_names: dict(EOTask: str) or None
        :return: List of dependencies
        :rtype: list(Dependency)
        """
        parsed_dependencies = [dep if isinstance(dep, Dependency) else Dependency(*dep) for dep in dependencies]
        for dep in parsed_dependencies:
            if task_names and dep.task in task_names:
                dep.set_name(task_names[dep.task])
        return parsed_dependencies

    def _set_task_uuid(self, dependencies):
        """ Adds universally unique user ids (UUID) to each task of the workflow

        :param dependencies: The list of dependencies between tasks defining the computational graph
        :type dependencies: list(Dependency)
        :return: A dictionary mapping UUID to dependencies
        :rtype: dict(str: Dependency)
        """
        uuid_dict = {}
        for dep in dependencies:
            task = dep.task
            if task.private_task_config.uuid in uuid_dict:
                raise ValueError('EOWorkflow cannot execute the same instance of EOTask multiple times')

            task.private_task_config.uuid = self.id_gen.next()
            uuid_dict[task.private_task_config.uuid] = dep

        return uuid_dict

    def create_dag(self, dependencies):
        """ Creates a directed graph from dependencies

        :param dependencies: A list of Dependency objects
        :type dependencies: list(Dependency)
        :return: A directed graph of the workflow
        :rtype: DirectedGraph
        """
        dag = DirectedGraph()
        for dep in dependencies:
            for vertex in dep.inputs:
                task_uuid = vertex.private_task_config.uuid
                if task_uuid not in self.uuid_dict:
                    raise ValueError('Task {}, which is an input of a task {}, is not part of the defined '
                                     'workflow'.format(vertex.__class__.__name__, dep.name))
                dag.add_edge(self.uuid_dict[task_uuid], dep)
            if not dep.inputs:
                dag.add_vertex(dep)
        return dag

    @staticmethod
    def _schedule_dependencies(dag):
        """ Computes an ordering < of tasks so that for any two tasks t and t' we have that if t depends on t' then
        t' < t. In words, all dependencies of a task precede the task in this ordering.

        :param dag: A directed acyclic graph representing dependencies between tasks.
        :type dag: DirectedGraph
        :return: A list of topologically ordered dependecies
        :rtype: list(Dependency)
        """
        in_degrees = dict(dag.get_indegrees())

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
            raise CyclicDependencyError('Tasks do not form an acyclic graph')

        return topological_order

    def execute(self, input_args=None, monitor=False):
        """ Executes the workflow

        :param input_args: External input arguments to the workflow. They have to be in a form of a dictionary where
            each key is an EOTask used in the workflow and each value is a dictionary or a tuple of arguments.
        :type input_args: dict(EOTask: dict(str: object) or tuple(object))
        :param monitor: If True workflow execution will be monitored
        :type monitor: bool
        :return: An immutable mapping containing results of terminal tasks
        :rtype: WorkflowResults
        """
        out_degs = dict(self.dag.get_outdegrees())

        input_args = self.parse_input_args(input_args)

        results = WorkflowResults(self._execute_tasks(input_args=input_args, out_degs=out_degs, monitor=monitor))

        LOGGER.debug('Workflow finished with %s', repr(results))
        return results

    @staticmethod
    def parse_input_args(input_args):
        """ Parses EOWorkflow input arguments provided by user and raises an error if something is wrong. This is
        done automatically in the process of workflow execution
        """
        input_args = input_args if input_args else {}
        for task, args in input_args.items():
            if not isinstance(task, EOTask):
                raise ValueError('Invalid input argument {}, should be an instance of EOTask'.format(task))

            if not isinstance(args, (tuple, dict)):
                raise ValueError('Execution input arguments of each task should be a dictionary or a tuple, for task '
                                 '{} got arguments of type {}'.format(task.__class__.__name__, type(args)))

        return input_args

    def _execute_tasks(self, *, input_args, out_degs, monitor):
        """ Executes tasks comprising the workflow in the predetermined order

        :param input_args: External input arguments to the workflow.
        :type input_args: Dict
        :param out_degs: Dictionary mapping vertices (task IDs) to their out-degrees. (The out-degree equals the number
        of tasks that depend on this task.)
        :type out_degs: Dict
        :return: A dictionary mapping dependencies to task results
        :rtype: dict
        """
        intermediate_results = {}

        for dep in self.ordered_dependencies:
            result = self._execute_task(dependency=dep,
                                        input_args=input_args,
                                        intermediate_results=intermediate_results,
                                        monitor=monitor)

            intermediate_results[dep] = result

            self._relax_dependencies(dependency=dep,
                                     out_degrees=out_degs,
                                     intermediate_results=intermediate_results)

        return intermediate_results

    def _execute_task(self, *, dependency, input_args, intermediate_results, monitor):
        """ Executes a task of the workflow

        :param dependency: A workflow dependency
        :type dependency: Dependency
        :param input_args: External task parameters.
        :type input_args: dict
        :param intermediate_results: The dictionary containing intermediate results, including the results of all
        tasks that the current task depends on.
        :type intermediate_results: dict
        :return: The result of the task in dependency
        :rtype: object
        """
        task = dependency.task
        inputs = tuple(intermediate_results[self.uuid_dict[input_task.private_task_config.uuid]]
                       for input_task in dependency.inputs)

        kw_inputs = input_args.get(task, {})
        if isinstance(kw_inputs, tuple):
            inputs += kw_inputs
            kw_inputs = {}

        LOGGER.debug("Computing %s(*%s, **%s)", task.__class__.__name__, str(inputs), str(kw_inputs))
        return task(*inputs, **kw_inputs, monitor=monitor)

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
        current_task = dependency.task
        for input_task in dependency.inputs:
            dep = self.uuid_dict[input_task.private_task_config.uuid]
            out_degrees[dep] -= 1

            if out_degrees[dep] == 0:
                LOGGER.debug("Removing intermediate result for %s", current_task.__class__.__name__)
                del intermediate_results[dep]

    def get_tasks(self):
        """ Returns an ordered dictionary {task_name: task} of all tasks within this workflow

        :return: Ordered dictionary with key being task_name (str) and an instance of a corresponding task from this
        workflow
        :rtype: OrderedDict
        """
        tasks = collections.OrderedDict()
        for dep in self.ordered_dependencies:
            tasks[dep.name] = dep.task

        return tasks

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
        try:
            # pylint: disable=C0415
            from eolearn.visualization import EOWorkflowVisualization
        except ImportError:
            raise RuntimeError('Subpackage eo-learn-visualization has to be installed in order to use EOWorkflow '
                               'visualization methods')
        return EOWorkflowVisualization(self)


class LinearWorkflow(EOWorkflow):
    """ A linear version of EOWorkflow where each tasks only gets results of the previous task

    Example:

        .. code-block:: python

            workflow = LinearWorkflow(task1, task2, task3)
    """
    def __init__(self, *tasks, **kwargs):
        """
        :param tasks: Tasks in the order of execution
        :type tasks: *EOTask
        """
        tasks = [self._parse_task(task) for task in tasks]
        tasks = self._make_tasks_unique(tasks)

        dependencies = [(task, [tasks[idx - 1][0]] if idx > 0 else []) for idx, (task, name) in enumerate(tasks)]
        super().__init__(dependencies, **kwargs)

    @staticmethod
    def _parse_task(task):
        """ Parses input task
        """
        if isinstance(task, EOTask):
            return task, None
        if isinstance(task, (tuple, list)) and len(task) == 2:
            return task

        raise ValueError('Cannot parse {}, expected an instance of EOTask or a tuple (EOTask, name)'.format(task))

    @staticmethod
    def _make_tasks_unique(tasks):
        """ If the same instances of tasks are given to LinearWorkflow, this will deep copy them
        """
        unique_tasks = []
        prev_tasks = set()

        for task, name in tasks:
            if task in prev_tasks:
                task = copy.deepcopy(task)

            prev_tasks.add(task)
            unique_tasks.append((task, name))

        return unique_tasks


@attr.s(cmp=False)  # cmp=False preserves the original hash
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
            raise ValueError('Value {} should be an instance of {}'.format(self.task, EOTask.__name__))
        self.task = self.task

        if isinstance(self.inputs, EOTask):
            self.inputs = [self.inputs]
        if not isinstance(self.inputs, (list, tuple)):
            raise ValueError('Value {} should be a list'.format(self.inputs))
        for input_task in self.inputs:
            if not isinstance(input_task, EOTask):
                raise ValueError('Value {} should be an instance of {}'.format(input_task, EOTask.__name__))

        if self.name is None:
            self.name = self.task.__class__.__name__

    def set_name(self, name):
        """ Sets a new name
        """
        self.name = name


class WorkflowResults(collections.Mapping):
    """ The result of a workflow is an (immutable) dictionary mapping [1] from EOTasks to results of the workflow.

    When an EOTask is passed as an index, its UUID, assigned during workflow execution, is used as the key (as opposed
    to the result of invoking __repr__ on the EO task). This ensures that indexing by task works even after pickling,
    and makes dealing with checkpoints more convenient.

    [1] https://docs.python.org/3.6/library/collections.abc.html#collections-abstract-base-classes
    """
    def __init__(self, results):
        self._result = results
        self._uuid_dict = {dep.task.private_task_config.uuid: dep for dep in results}

    def __getitem__(self, item):
        if isinstance(item, EOTask):
            item = self._uuid_dict[item.private_task_config.uuid]
        return self._result[item]

    def __len__(self):
        return len(self._result)

    def __iter__(self):
        return iter(self._result)

    def __contains__(self, item):
        if isinstance(item, EOTask):
            item = self._uuid_dict.get(item.private_task_config.uuid, item)
        return item in self._result

    def __eq__(self, other):
        return self._result == other

    def __ne__(self, other):
        return self._result != other

    def keys(self):
        """ Returns dictionary keys """
        return {dep.task: None for dep in self._result}.keys()

    def values(self):
        """ Returns dictionary values """
        return self._result.values()

    def items(self):
        """ Returns dictionary items """
        return {dep.task: value for dep, value in self._result.items()}.items()

    def get(self, key, default=None):
        """ Dictionary get method """
        if isinstance(key, EOTask):
            key = self._uuid_dict[key.private_task_config.uuid]
        return self._result.get(key, default)

    def eopatch(self):
        """ Return the EOPatch from the workflow result """
        return list(self.values())[-1]

    def __repr__(self):
        repr_list = ['{}('.format(self.__class__.__name__)]

        for _, dep in self._uuid_dict.items():
            result_repr = repr(self._result[dep]).replace('\n', '\n    ')
            dependency_repr = '{}({}):\n    {}'.format(Dependency.__name__, dep.name, result_repr)

            repr_list.append(dependency_repr)

        return '\n  '.join(repr_list) + '\n)'


class _UniqueIdGenerator:
    """ Generator of unique IDs, which is used in workflows only
    """

    MAX_UUIDS = 2 ** 20

    def __init__(self):
        self.uuids = set()

    def _next(self):
        if len(self.uuids) + 1 > _UniqueIdGenerator.MAX_UUIDS:
            raise MemoryError('Limit of max UUIDs reached')

        while True:
            uid = uuid.uuid4()
            if uid not in self.uuids:
                self.uuids.add(uid)
                return uid

    def next(self):
        """ Generates an ID
        """
        return self._next().hex
