"""
The eoworkflow module, together with eotask and eodata, provides core building blocks for specifying and executing
workflows.

-- Summary --
A workflow is a directed (acyclic) graph composed of instances of EOTask objects. Each task may take as input the
results of other tasks and external arguments. The external arguments are passed anew each time the workflow is
executed. The workflow builds the computational graph, performs dependency resolution, and executes the tasks.
If the input graph is cyclic, the workflow raises a `CyclicDependencyError`.

The result of a workflow execution is an immutable mapping from tasks to results. The result contains tasks with
zero out-degree (i.e. terminal tasks).

The workflow can be exported to a DOT description language and visualized.
"""

import collections
import logging
import uuid

# the next are needed for DAG visualization only
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt

from .eotask import EOTask
from .graph import DirectedGraph
from .utilities import deep_eq


LOGGER = logging.getLogger(__file__)

GRAPHVIZ_EXT = 'dot'


class CyclicDependencyError(ValueError):
    pass


Dependency = collections.namedtuple('Dependency', ('transform', 'inputs'))


class _UniqueIdGenerator:
    """
    Generates a sequence of unique IDs. Should be used for the purposes of
    workflow class only.
    """
    MAX_UUIDS = 2**20

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
        return self._next().hex


class WorkflowResult(collections.abc.Mapping):
    """
    The result of a workflow is an (immutable) dictionary mapping [1] from unique IDs (strings) to tuples.

    When an EOTask is passed as an index, its UUID, asigned during workflow execution, is used as the key (as opposed
    to the result of invoking __repr__ on the EO task). This ensures that indexing by task works even after pickling,
    and makes dealing with checkpoints more convenient.

    [1] https://docs.python.org/3.6/library/collections.abc.html#collections-abstract-base-classes
    """
    def __init__(self, *args, **kwargs):
        self._result = dict(*args, **kwargs)

    def __getitem__(self, item):
        return self._result[WorkflowResult.get_key(item)]

    def __len__(self):
        return len(self._result)

    def __iter__(self):
        return iter(self._result)

    def __contains__(self, item):
        return WorkflowResult.get_key(item) in self._result

    def __eq__(self, other):
        return self._result == other

    def __ne__(self, other):
        return self._result != other

    def keys(self):
        return self._result.keys()

    def values(self):
        return self._result.values()

    def items(self):
        return self._result.items()

    def get(self, key, default=None):
        key = WorkflowResult.get_key(key)
        return self._result.get(key, default)

    def __repr__(self):
        return repr(self._result)

    @staticmethod
    def get_key(item):
        return item.uuid if isinstance(item, EOTask) else item


class EOWorkflow:
    def __init__(self, dependencies, task2id=None):
        """
        The constructor to instantiate a workflow from a list of dependencies.

        The optional parameter ``task2id``, defaulting to ``None``, provides human-readable names to tasks comprising
        the workflow. When ``task2id`` is ``None`` unique ID is generated for each task and used in all subsequent
        computations. This ensures that the task identity is preserved across workflow instantiations. When ``task2id``
        is not ``None`` the parameter ``dependencies`` may use these names to define the graph.

        :param dependencies: A list of dependencies between tasks, specifying the computational graph.
        :type dependencies: List[Dependency]
        :param task2id: A dictionary providing human-readable names to EOTask's (optional); defaults to ``None``
        :type task2id: Dict
        """
        self.id_gen = _UniqueIdGenerator()

        self.task2id = task2id if task2id else {dep.transform: self.id_gen.next() for dep in dependencies}
        self.id2task = {v: k for k, v in self.task2id.items()}

        dependencies = EOWorkflow._rename_dependencies(dependencies, self.task2id)

        self.deps, self.dag = EOWorkflow.create_dag(dependencies)
        self.order = self._resolve_dependencies(self.dag)

        self._name_task()

    def _name_task(self):
        """
        Assigns names to tasks.
        """
        for task, uid in self.task2id.items():
            task.uuid = uid

    @staticmethod
    def _rename_dependencies(dependencies, task2id):
        """
        Renames dependencies using names provided by ``task2id``; this way the graph remains unchanged across
        instantiations of the workflow (from disk).

        :param dependencies: The list of dependencies between tasks defining the computational graph
        :type dependencies: List[Dependency]
        :param task2id: Human-readable names of tasks
        :type task2id: Dict
        :return: List of dependencies using human-readable names as given by ``task2id``
        :rtype: List[Dependency]
        """
        return [
            Dependency(
                transform=task2id.get(dep.transform, dep.transform),
                inputs=[task2id.get(task, task) for task in dep.inputs]
            )
            for dep in dependencies
        ]

    @staticmethod
    def make_linear_workflow(*tasks, **kwargs):
        """
        Factory method for creating linear workflows.

        :param tasks: EOTask's t1,t2,...,tk with dependencies t1->t2->...->tk
        :param kwargs: Optional keyword arguments (such as workflow name) forwarded to the constructor
        :return: A new EO workflow instance
        :rtype: EOWorkflow
        """
        return EOWorkflow(dependencies=[
            Dependency(
                transform=task,
                inputs=[tasks[i - 1]] if i > 0 else []
            )
            for i, task in enumerate(tasks)
        ], **kwargs)

    @staticmethod
    def create_dag(dependencies):
        """
        Creates a directed graph from dependencies.

        :param dependencies: A list of Dependency objects
        :type dependencies: List[Dependency]
        :return: A pair ``(deps, dag)``. Here ``deps`` maps vertices to its dependencies, ``dag`` is the directed
        graph
        :rtype: Tuple[Dict, DirectedGraph]
        """
        deps = collections.defaultdict(list)
        dag = DirectedGraph()
        for dep in dependencies:
            deps[dep.transform].extend(dep.inputs)
            for vertex in dep.inputs:
                dag.add_edge(vertex, dep.transform)
            if not dep.inputs:
                dag.add_vertex(dep.transform)
        return deps, dag

    @staticmethod
    def _resolve_dependencies(dag):
        """
        Computes an ordering < of tasks so that for any two tasks t and t' we have that if t depends on t' then
        t' < t. In words, all dependencies of a task precede the task in this ordering.

        :param dag: A directed (acylic) graph representing dependencies between tasks.
        :type dag: DirectedGraph
        :return: A list of topologically ordered vertices of ``dag`` if such ordering exists; otherwise raises
        ``CyclicDependencyException``
        :rtype: List[object]
        """
        in_degs = dict(dag.get_indegrees())

        independent_vertices = [vertex for vertex in dag if dag.get_indegree(vertex) == 0]
        topological_order = []
        while independent_vertices:
            v_vertex = independent_vertices.pop()
            topological_order.append(v_vertex)

            for u_vertex in dag[v_vertex]:
                in_degs[u_vertex] -= 1
                if in_degs[u_vertex] == 0:
                    independent_vertices.append(u_vertex)

        if len(topological_order) != len(dag):
            raise CyclicDependencyError

        return topological_order

    @staticmethod
    def _args_consistent(task_id, kw_inputs, old_kw_inputs):
        """
        Checks whether the external arguments for ``task_id`` remain unchanged. The arguments have changed if
        (i) the values of the arguments differ, (ii) an argument with key ``k`` is present in one dictionary but
        not in the other.

        The method assumes values are comparable with the ``__eq_``_ operator.

        :param task_id: The ID of the task whose status we're determining
        :type task_id: str
        :param kw_inputs: The first dictionary of external arguments
        :type kw_inputs: Dict
        :param old_kw_inputs: The second dictionary of external arguments
        :type old_kw_inputs: Dict
        :return: ``True`` if arguments for ``task_id`` remain unchanged, ``False`` otherwise
        :rtype: bool
        """
        pred_1 = task_id in kw_inputs
        pred_2 = task_id in old_kw_inputs

        if pred_1 and pred_2:
            return deep_eq(kw_inputs[task_id], old_kw_inputs[task_id])

        return pred_1 == pred_2

    def execute(self, input_args=None):
        """
        Execute the current workflow.

        :param input_args: External input arguments to the workflow.
        :type input_args: Dict
        :return: An immutable mapping containing results of terminal tasks
        :rtype: WorkflowResult
        """
        outdegs = dict(self.dag.get_outdegrees())

        input_args = {WorkflowResult.get_key(k): v for k, v in input_args.items()} if input_args else {}

        _, intermediate_results = self._execute_tasks(input_args=input_args, outdegs=outdegs)

        return WorkflowResult(intermediate_results)

    def _execute_tasks(self, *, input_args, outdegs):
        """
        Executes tasks comprising the workflow in the predetermined order.

        :param input_args: External input arguments to the workflow.
        :type input_args: Dict
        :param outdegs: Dictionary mapping vertices (task IDs) to their out-degrees. (The out-degree equals the number
        of tasks that depend on this task.)
        :type outdegs: Dict
        :return: An immutable mapping containing results of terminal tasks
        :rtype: WorkflowResult
        """
        done_tasks = set()

        intermediate_results = {}

        for t_id in self.order:
            result = self._execute_task(input_args=input_args,
                                        intermediate_results=intermediate_results,
                                        task_id=t_id)

            intermediate_results[t_id] = result

            self._relax_dependencies(out_degrees=outdegs,
                                     current_task_id=t_id,
                                     intermediate_results=intermediate_results)

        return done_tasks, intermediate_results

    def _execute_task(self, *, input_args, intermediate_results, task_id):
        """
        Either executes or loads from disk the result of the task with ID ``task_id``.

        :param input_args: External task parameters.
        :type input_args: Dict
        :param intermediate_results: The dictionary containing intermediate results, including the results of all
        tasks that the current task depends on.
        :type intermediate_results: Dict
        :param task_id: The ID of the current task.
        :type task_id: str
        :return: The result of the task with ID ``task_id``
        :rtype: object
        """
        task = self.id2task[task_id]
        kw_inputs = input_args[task_id] if input_args and task_id in input_args else {}
        inputs = tuple(intermediate_results[t_dep] for t_dep in self.deps[task_id])
        LOGGER.debug("Computing %s(*%s, **%s)", str(task), str(inputs), str(kw_inputs))
        return task(*inputs, **kw_inputs)

    def _relax_dependencies(self, *, intermediate_results, out_degrees, current_task_id):
        """
        Relaxes dependencies incurred by ``task_id``. After the task with ID ``task_id`` has been successfully
        executed, all the tasks it depended on are upadted. If ``task_id`` was the last remaining dependency of a task
        ``t`` then ``t``'s result is removed from memory and, depending on ``remove_intermediate``, from disk.

        :param intermediate_results: The dictionary containing the intermediate results (needed by tasks that have yet
        to be executed) of the already-executed tasks
        :type intermediate_results: Dict
        :param out_degrees: Out-degrees of tasks
        :type out_degrees: Dict
        :param current_task_id: The ID of the current task whose dependencies are being relaxed
        :type current_task_id: str
        """
        current_task = self.id2task[current_task_id]
        for dep_task_id in self.deps[current_task_id]:
            if out_degrees[dep_task_id] == 1:
                LOGGER.debug("Removing intermediate result for %s", str(current_task))
                intermediate_results.pop(dep_task_id)
            out_degrees[dep_task_id] -= 1

    def get_dot(self):
        """
        Generate the DOT description of the underlying computational graph.

        :return: The DOT representation of the computational graph
        :rtype: graphviz.Digraph
        """
        dot = Digraph()
        for u_task, u_id in self.task2id.items():
            for v_id in self.deps[u_id]:
                dot.edge(
                    '{}_{}'.format(type(self.id2task[v_id]).__name__, v_id[:4]),
                    '{}_{}'.format(type(u_task).__name__, u_id[:4])
                )
        return dot

    def dependency_graph(self, outfile, view=False):
        """
        Visualize the computational graph.

        :param outfile: The name of the output image of the graph.
        :type outfile: str
        :param view: A flag indicating whether to view the image of the graph
        :type view: bool
        """
        dot = self.get_dot()

        out_fpath = './{}'.format(outfile)
        with open(out_fpath, 'w') as fout:
            fout.write(dot.source)

        if view:
            graph = nx.drawing.nx_pydot.read_dot(out_fpath)
            nx.draw(graph, with_labels=True)
            plt.show()
