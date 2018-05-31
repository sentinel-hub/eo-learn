from eolearn.core import EOTask, FeatureType


class AddValidDataMaskTask(EOTask):
    """ EOTask for adding custom mask array used to filter reflectances data

        This task allows the user to specify the criteria used to generate a valid data mask, which can be used to
        filter the data stored in the `FeatureType.DATA`
    """
    def __init__(self, predicate, valid_data_field='VALID_DATA'):
        """ Constructor of the class requires a predicate defining the function used to generate the valid data mask. A
        predicate is a function that returns the truth value of some condition.

        An example predicate could be an `and` operator between a cloud mask and a snow mask.

        :param predicate: Function used to generate a `valid_data` mask
        :type predicate: func
        :param valid_data_field: Name of `FeatureType.MASK` field storing the result of the predicate function
        :type valid_data_field: str
        """
        self.predicate = predicate
        self.valid_data_field = valid_data_field

    def execute(self, eopatch):
        """ Execute predicate on input eopatch

        :param eopatch: Input `eopatch` instance
        :return: The same `eopatch` instance with a `mask.valid_data` array computed according to the predicate
        """
        eopatch.add_feature(FeatureType.MASK, self.valid_data_field, self.predicate(eopatch))
        return eopatch
