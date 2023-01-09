"""
Tasks used for train set preparation

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import FeaturesSpecification


class TrainTestSplitType(Enum):
    """An enum defining TrainTestSplitTask's methods of splitting the data into subsets"""

    PER_PIXEL = "per_pixel"
    PER_CLASS = "per_class"
    PER_VALUE = "per_value"


class TrainTestSplitTask(EOTask):
    """Randomly assign each pixel or groups of pixels to multiple subsets (e.g., test/train/validate).

    When sampling PER_PIXEL the input feature only specifies the shape of the output feature. For PER_CLASS and
    PER_VALUE the input MASK_TIMELESS feature should group together pixels with similar properties, e.g. polygon ids,
    connected component ids, etc. The task then ensures that such groups are kept together (so the whole polygon is
    either in train or test).

    There are three modes of split operation:

    - :attr:`PER_PIXEL<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_PIXEL>` (default), where pixels are
      assigned to a subset randomly, regardless of their value,

    - :attr:`PER_CLASS<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_CLASS>`, where pixels of the same value
      are assigned to the same subset,

    - :attr:`PER_VALUE<eolearn.ml_tools.train_test_split.TrainTestSplitType.PER_VALUE>`, where pixels of the same value
      are assigned to the same subset consistently across EOPatches. In other words, if a group of pixels of the same
      value lies on multiple EOPatches, they are assigned to the same subset in all EOPatches. In this case, the `seed`
      argument of the `execute` method is ignored.

    Classes are defined by a list of cumulative probabilities, passed as the `bins` argument, the same way as the `bins`
    argument in `numpy.digitize <https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html>`_. Valid
    classes are enumerated from 1 onward and if ignore_values is provided, all values equal to it get assigned to class
    0.

    To get a train/test split as 80/20, bins argument should be provided as `bins=[0.8]`.

    To get a train/val/test split as 60/20/20, bins argument should be provided as `bins=[0.6, 0.8]`.

    Splits can also be made into as many subsets as desired, e.g., `bins=[0.1, 0.2, 0.3, 0.7, 0.9]`.

    After execution each pixel will have a value representing the train, test and/or validation set stored in the
    output feature.
    """

    def __init__(
        self,
        input_feature: FeaturesSpecification,
        output_feature: FeaturesSpecification,
        bins: Union[float, List[Any]],
        split_type: TrainTestSplitType = TrainTestSplitType.PER_PIXEL,
        ignore_values: Optional[List[int]] = None,
    ):
        """
        :param input_feature: The input feature to guide the split.
        :param input_feature: The output feature where to save the mask.
        :param bins: Cumulative probabilities of all value classes or a single float, representing a fraction.
        :param split_type: Value split type, either 'PER_PIXEL', 'PER_CLASS' or 'PER_VALUE'.
        :param ignore_values: A list of values in input_feature to ignore and not assign them to any subsets.
        """
        self.input_feature = self.parse_feature(input_feature, allowed_feature_types=[FeatureType.MASK_TIMELESS])
        self.output_feature = self.parse_feature(output_feature, allowed_feature_types=[FeatureType.MASK_TIMELESS])

        if np.isscalar(bins):
            bins = [bins]

        if (
            not isinstance(bins, list)
            or not all(isinstance(bi, float) for bi in bins)
            or np.any(np.diff(bins) <= 0)
            or bins[0] <= 0
            or bins[-1] >= 1
        ):
            raise ValueError("bins argument should be a list of ascending floats inside an open interval (0, 1)")

        self.ignore_values = set() if ignore_values is None else set(ignore_values)
        self.bins = bins
        self.split_type = TrainTestSplitType(split_type)

    def execute(self, eopatch: EOPatch, *, seed: Optional[int] = None) -> EOPatch:
        """
        :param eopatch: input EOPatch
        :param seed: An argument to be passed to numpy.random.seed function.
        :return: Input EOPatch with the train set mask.
        """
        if self.split_type in [TrainTestSplitType.PER_CLASS, TrainTestSplitType.PER_PIXEL]:
            np.random.seed(seed)

        data = eopatch[self.input_feature]

        if self.split_type == TrainTestSplitType.PER_PIXEL:
            rands = np.random.rand(*data.shape)
            eopatch[self.output_feature] = np.digitize(rands, self.bins) + 1
            return eopatch

        classes = set(np.unique(data)) - self.ignore_values
        output_mask = np.zeros_like(data)

        for class_id in classes:
            if self.split_type == TrainTestSplitType.PER_VALUE:
                np.random.seed(class_id)

            fold = np.digitize(np.random.rand(), self.bins) + 1
            output_mask[data == class_id] = fold

        eopatch[self.output_feature] = output_mask
        return eopatch
