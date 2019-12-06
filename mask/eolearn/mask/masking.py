"""
Module for creating mask features

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)
Copyright (c) 2018-2019 Johannes Schmid (GeoVille)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np

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


class MaskFeature(EOTask):
    """ Masks out values of a feature using defined values of a given mask feature.

        As an example, it can be used to mask the data feature using values from the Sen2cor Scene Classification
        Layer (SCL).

        Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018

        :param feature: A feature to be masked with optional new feature name
        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param mask_feature: Masking feature. Values of this mask will be used to mask values of `feature`
        :type mask_feature: (FeatureType, str)
        :param mask_values: List of values of `mask_feature` to be used for masking `feature`
        :type mask_values: list of int
        :param no_data_value: Value that replaces masked values in `feature`. Default is `NaN`
        :type no_data_value: np.float32
        :return: The same `eopatch` instance with a masked array
    """
    def __init__(self, feature, mask_feature, mask_values, no_data_value=np.nan):
        self.feature = self._parse_features(feature, new_names=True,
                                            default_feature_type=FeatureType.DATA,
                                            rename_function='{}_MASKED'.format)
        self.mask_feature = self._parse_features(mask_feature, default_feature_type=FeatureType.MASK)
        self.mask_values = mask_values
        self.no_data_value = no_data_value

    def execute(self, eopatch):
        """ Mask values of `feature` according to the `mask_values` in `mask_feature`

        :param eopatch: `eopatch` to be processed
        :return: Same `eopatch` instance with masked `feature`
        """
        feature_type, feature_name, new_feature_name = next(self.feature(eopatch))
        mask_feature_type, mask_feature_name = next(self.mask_feature(eopatch))

        data = np.copy(eopatch[feature_type][feature_name])
        mask = eopatch[mask_feature_type][mask_feature_name]

        if not isinstance(self.mask_values, list):
            raise ValueError('Incorrect format or values of argument `mask_values`')

        for value in self.mask_values:
            data[mask.squeeze() == value] = self.no_data_value

        eopatch.add_feature(feature_type, new_feature_name, data)

        return eopatch
