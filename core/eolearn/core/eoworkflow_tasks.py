"""
Module implementing tasks that have a special effect in `EOWorkflow`
"""
from .eotask import EOTask
from .eodata import EOPatch


class OutputTask(EOTask):
    """ Stores data as an output of `EOWorkflow` results
    """
    def __init__(self, name=None, features=...):
        """
        :param name: A name under which the data will be saved in `WorkflowResults`
        :type name: str or None
        :param features: A collection of features to be kept if the data is an `EOPatch`
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        self._name = name or f'output_{self.private_task_config.uid}'
        self.features = features

    @property
    def name(self):
        """ Provides a name under which data will be saved in `WorkflowResults`

        :return: A name
        :rtype: str
        """
        return self._name

    def execute(self, data):
        """
        :param data: input data
        :type data: object
        :return: Same data, to be stored in results (for `EOPatch` returns shallow copy containing only `features`)
        :rtype: object
        """
        if isinstance(data, EOPatch):
            return data.__copy__(features=self.features)
        return data
