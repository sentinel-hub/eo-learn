from eolearn.core import EOTask, FeatureType


class AddValidDataMaskTask(EOTask):
    """ EOTask for adding custom mask array used to filter reflectances data

        This task allows the user to specify the criteria used to generate a valid data mask, which can be used to
        filter the data stored in the `FeatureType.DATA`
    """
    def __init__(self, predicate, valid_data_feature='VALID_DATA'):
        """ Constructor of the class requires a predicate defining the function used to generate the valid data mask. A
        predicate is a function that returns the truth value of some condition.

        An example predicate could be an `and` operator between a cloud mask and a snow mask.

        :param predicate: Function used to generate a `valid_data` mask
        :type predicate: func
        :param valid_data_feature: Feature which will store valid data mask
        :type valid_data_feature: str
        """
        self.predicate = predicate
        self.valid_data_feature = self._parse_features(valid_data_feature, default_feature_type=FeatureType.MASK)

    def execute(self, eopatch):
        """ Execute predicate on input eopatch

        :param eopatch: Input `eopatch` instance
        :return: The same `eopatch` instance with a `mask.valid_data` array computed according to the predicate
        """
        feature_type, feature_name = next(self.valid_data_feature())
        eopatch[feature_type][feature_name] = self.predicate(eopatch)
        return eopatch
    
    
class SCLmask(EOTask):
    """
    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018
    Masks out a layer using defined values (mask_values) from the Sen2cor Scene Classification Layer (SCL)
    """
    def __init__(self, mask_values, layer):
        self.mask_values = mask_values
        self.layer = layer
        self.replacement = np.nan
        
    def execute(self, eopatch):
        data = np.copy(eopatch.data[self.layer])
        mask = eopatch.mask['SCL']
        for band in range(data.shape[3]):
            for mv in self.mask_values:
                data[:,:,:,band][mask[:,:,:,0] == mv] = self.replacement
        eopatch.add_feature(FeatureType.DATA, 'masked', data)
        return eopatch
