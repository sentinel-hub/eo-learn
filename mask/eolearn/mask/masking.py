"""
Module for creating mask features

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)
Copyright (c) 2018-2019 Johannes Schmid (GeoVille)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from typing import Callable, Union

import numpy as np

from eolearn.core import EOTask, ZipFeatureTask


class JoinMasksTask(ZipFeatureTask):
    """Joins together masks with the provided logical operation."""

    def __init__(self, input_features, output_feature, join_operation: Union[str, Callable] = "and"):
        """
        :param input_features: Mask features to be joined together.
        :param output_feature: Feature to which to save the joined mask.
        :param join_operation: How to join masks. Supports `'and'`, `'or'`, `'xor'`, or a `Callable` object.
        """
        input_features = self.parse_features(input_features)
        output_feature = self.parse_feature(output_feature)

        if isinstance(join_operation, str):
            methods = {"and": np.logical_and, "or": np.logical_or, "xor": np.logical_xor}
            if join_operation not in methods:
                raise ValueError(
                    f"Join operation {join_operation} is not a viable choice. For operations other than {list(methods)}"
                    "the user must provide a `Callable` object."
                )
            self.join_method = methods[join_operation]
        else:
            self.join_method = join_operation

        super().__init__(input_features, output_feature)

    def zip_method(self, *masks: np.ndarray) -> np.ndarray:
        """Joins masks using the provided operation"""
        final_mask, *masks = masks
        for mask in masks:
            final_mask = self.join_method(final_mask, mask)
        return final_mask


class MaskFeatureTask(EOTask):
    """Masks out values of a feature using defined values of a given mask feature.

    As an example, it can be used to mask the data feature using values from the Sen2cor Scene Classification
    Layer (SCL).

    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018
    """

    def __init__(self, feature, mask_feature, mask_values, no_data_value=np.nan):
        """
        :param feature: A feature to be masked with optional new feature name
        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param mask_feature: Masking feature. Values of this mask will be used to mask values of `feature`
        :type mask_feature: (FeatureType, str)
        :param mask_values: List of values of `mask_feature` to be used for masking `feature`
        :type mask_values: list of int
        :param no_data_value: Value that replaces masked values in `feature`. Default is `NaN`
        :type no_data_value: float
        :return: The same `eopatch` instance with a masked array
        """
        self.renamed_feature = self.parse_renamed_feature(feature)
        self.mask_feature = self.parse_feature(mask_feature)
        self.mask_values = mask_values
        self.no_data_value = no_data_value

        if not isinstance(self.mask_values, list):
            raise ValueError("Incorrect format or values of argument 'mask_values'")

    def execute(self, eopatch):
        """Mask values of `feature` according to the `mask_values` in `mask_feature`

        :param eopatch: `eopatch` to be processed
        :return: Same `eopatch` instance with masked `feature`
        """
        feature_type, feature_name, new_feature_name = self.renamed_feature
        mask_feature_type, mask_feature_name = self.mask_feature

        data = np.copy(eopatch[feature_type][feature_name])
        mask = eopatch[mask_feature_type][mask_feature_name]

        for value in self.mask_values:
            data = apply_mask(data, mask, value, self.no_data_value, feature_type, mask_feature_type)

        eopatch[feature_type][new_feature_name] = data
        return eopatch


def apply_mask(data, mask, old_value, new_value, data_type, mask_type):
    """A general masking function

    :param data: A data feature
    :type data: numpy.ndarray
    :param mask: A mask feature
    :type mask: numpy.ndarray
    :param old_value: An old value in data that will be replaced
    :type old_value: float
    :param new_value: A new value that will replace the old value in data
    :type new_value: float
    :param data_type: A data feature type
    :type data_type: FeatureType
    :param mask_type: A mask feature type
    :type mask_type: FeatureType
    """
    if not (data_type.is_spatial() and mask_type.is_spatial()):
        raise ValueError("Masking with non-spatial data types is not yet supported")

    if data_type.is_timeless() and mask_type.is_temporal():
        raise ValueError("Cannot mask timeless data feature with time dependent mask feature")

    if data.shape[-3:-1] != mask.shape[-3:-1]:
        raise ValueError("Data feature and mask feature have different spatial dimensions")
    if mask_type.is_temporal() and data.shape[0] != mask.shape[0]:
        raise ValueError("Data feature and mask feature have different temporal dimensions")

    if mask.shape[-1] == data.shape[-1]:
        data[..., mask == old_value] = new_value
    elif mask.shape[-1] == 1:
        data[..., mask[..., 0] == old_value, :] = new_value
    else:
        raise ValueError(
            f"Mask feature has {mask.shape[-1]} number of bands while data feature has {data.shape[-1]} number of bands"
        )
    return data
