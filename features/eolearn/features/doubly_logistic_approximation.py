"""
Module for calculating doubly logistic approximation.

Credits:
Copyright (c) 2020 Beno Šircelj (Josef Stefan Institute)
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import itertools as it
from typing import List, Optional

import numpy as np
from scipy.optimize import curve_fit

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import SingleFeatureSpec


def doubly_logistic(middle, initial_value, scale, a1, a2, a3, a4, a5) -> np.ndarray:
    # pylint: disable=invalid-name
    """Function passed to `scipy.optimize`"""
    funclist = [lambda y: np.exp(-(((a1 - y) / a4) ** a5)), lambda y: np.exp(-(((y - a1) / a2) ** a3))]
    return initial_value + scale * np.piecewise(middle, [middle < a1, middle >= a1], funclist)


class DoublyLogisticApproximationTask(EOTask):
    """
    EOTask class for calculation of doubly logistic approximation on each pixel for a feature. The task creates new
    feature with the function parameters for each pixel as vectors.
    :param feature: A feature on which the function will be approximated
    :param new_feature: Name of the new feature where parameters of the function are saved
    :param initial_parameters: Initial parameter guess
    :param valid_mask: A feature used as a mask for valid regions. If left as None the whole patch is used
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        new_feature: SingleFeatureSpec = (FeatureType.DATA_TIMELESS, "DOUBLY_LOGISTIC_PARAM"),
        initial_parameters: Optional[List[float]] = None,
        valid_mask: Optional[SingleFeatureSpec] = None,
    ):
        self.initial_parameters = initial_parameters
        self.feature = self.parse_feature(feature)
        self.new_feature = self.parse_feature(new_feature)
        self.valid_mask = (
            self.parse_feature(valid_mask, allowed_feature_types=[FeatureType.MASK]) if valid_mask else None
        )

    def _fit_optimize(self, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
        """
        :param x_axis: Horizontal coordinates of points
        :param y_axis: Vertical coordinates of points
        :return: List of optimized parameters [c1, c2, a1, a2, a3, a4, a5]
        """
        bounds_lower = [
            np.min(y_axis),
            -np.inf,
            x_axis[0],
            0.15,
            1,
            0.15,
            1,
        ]
        bounds_upper = [
            np.max(y_axis),
            np.inf,
            x_axis[-1],
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ]
        if self.initial_parameters is None:
            self.initial_parameters = [np.mean(y_axis), 0.2, (x_axis[-1] - x_axis[0]) / 2, 0.15, 10, 0.15, 10]
        optimal_values = curve_fit(
            doubly_logistic,
            x_axis,
            y_axis,
            self.initial_parameters,
            bounds=(bounds_lower, bounds_upper),
            maxfev=1000000,
            absolute_sigma=True,
        )
        return optimal_values[0]

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: Input eopatch with data on which the doubly logistic approximation is computed
        :return: Output patch with doubly logistic approximation parameters
        """
        data = eopatch[self.feature].squeeze(axis=3)

        times = np.asarray([time.toordinal() for time in eopatch.timestamp])
        times = (times - times[0]) / (times[-1] - times[0])

        time, height, width = data.shape

        if self.valid_mask:
            valid_data_mask = eopatch[self.valid_mask]
        else:
            valid_data_mask = np.ones((time, height, width), dtype=bool)

        all_parameters = np.zeros((height, width, 7))
        for height_ind, width_ind in it.product(range(height), range(width)):
            valid_curve = data[:, height_ind, width_ind][valid_data_mask[:, height_ind, width_ind].squeeze()]
            valid_times = times[valid_data_mask[:, height_ind, width_ind].squeeze()]

            all_parameters[height_ind, width_ind] = self._fit_optimize(valid_times, valid_curve)

        eopatch[self.new_feature] = all_parameters

        return eopatch
