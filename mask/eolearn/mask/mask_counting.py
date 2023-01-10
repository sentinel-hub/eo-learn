"""
Module for generating count masks

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from typing import List

import numpy as np

from eolearn.core import MapFeatureTask
from eolearn.core.types import FeaturesSpecification


class ClassFrequencyTask(MapFeatureTask):
    """Calculates frequencies of each provided class through the temporal dimension."""

    def __init__(
        self,
        input_feature: FeaturesSpecification,
        output_feature: FeaturesSpecification,
        classes: List[int],
        no_data_value: int = 0,
    ):
        """
        :param input_feature: A source feature from which to read the values.
        :param output_feature: An output feature to which to write the frequencies.
        :param classes: Classes of which frequencies to calculate.
        """
        super().__init__(input_feature, output_feature)

        if not isinstance(classes, list) or not all(isinstance(x, int) for x in classes):
            raise ValueError("classes argument should be a list of integers.")

        if no_data_value in classes:
            raise ValueError("classes argument must not contain no_data_value")

        self.classes = classes
        self.no_data_value = no_data_value

    def map_method(self, feature):
        """Map method being applied to the feature that calculates the frequencies."""
        count_valid = np.count_nonzero(feature != self.no_data_value, axis=0)

        class_counts = (np.count_nonzero(feature == scl, axis=0) for scl in self.classes)

        with np.errstate(invalid="ignore"):
            class_counts = [np.divide(count, count_valid, dtype=np.float32) for count in class_counts]

        return np.concatenate(class_counts, axis=-1)
