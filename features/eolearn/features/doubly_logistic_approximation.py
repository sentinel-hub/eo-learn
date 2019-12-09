"""
Module for calculating doubly logistic approximation.
"""
import itertools as it
import numpy as np
from scipy.optimize import curve_fit
from eolearn.core import EOTask


def doubly_logistic(middle, initial_value, scale, a1, a2, a3, a4, a5):
    # pylint: disable=invalid-name,locally-disabled
    """
    Function that is passed to scipy.optimize
    """
    return initial_value + scale * np.piecewise(middle, [middle < a1, middle >= a1],
                                                [lambda y: np.exp(-((a1 - y) / a4) ** a5),
                                                 lambda y: np.exp(-((y - a1) / a2) ** a3)])


class DoublyLogisticApproximationTask(EOTask):
    """
    EOTask class for calculation of doubly logistic approximation on each pixel for a feature. The task creates new
    feature with the function parameters for each pixel as vectors.
    :param feature: A feature on which the function will be approximated
    :type feature: str
    :param new_feature: Name of the new feature where parameters of the function are saved
    :type new_feature: str
    :param initial_parameters: Initial parameter guess
    :type initial_parameters: List of floats length 7 corresponding to each parameter
    :param mask_feature: A feature used as a mask for valid regions. If left as None the whole patch is used
    :type mask_feature: (FeatureType, str) or None
    """

    def __init__(self, feature, new_feature='DOUBLY_LOGISTIC_PARAM', initial_parameters=None, mask_feature=None):
        self.feature = feature
        self.initial_parameters = initial_parameters
        self.new_feature = new_feature
        self.mask_feature = mask_feature

    def _fit_optimize(self, x_axis, y_axis):
        """
        :param x_axis: Horizontal coordinates of points
        :type x_axis: List of floats
        :param y_axis: Vertical coordinates of points
        :type y_axis: List of floats
        :return: List of optimized parameters [c1, c2, a1, a2, a3, a4, a5]
        """
        bounds_lower = [np.min(y_axis), -np.inf, x_axis[0], 0.15, 1, 0.15, 1, ]
        bounds_upper = [np.max(y_axis), np.inf, x_axis[-1], np.inf, np.inf, np.inf, np.inf, ]
        if self.initial_parameters is None:
            self.initial_parameters = [np.mean(y_axis), 0.2, (x_axis[-1] - x_axis[0]) / 2, 0.15, 10, 0.15, 10]
        optimal_values = curve_fit(doubly_logistic, x_axis, y_axis, self.initial_parameters,
                                   bounds=(bounds_lower, bounds_upper), maxfev=1000000,
                                   absolute_sigma=True)
        return optimal_values[0]

    def execute(self, eopatch):
        """
        :param eopatch: Input eopatch with data on which the doubly logistic approximation is computed
        :return: Output patch with doubly logistic approximation parameters
        """
        data = eopatch.data[self.feature].squeeze(axis=3)

        times = np.asarray([time.toordinal() for time in eopatch.timestamp])
        times = (times - times[0]) / (times[-1] - times[0])

        time, height, width = data.shape

        if self.mask_feature:
            valid_data_mask = eopatch[self.mask_feature[0]][self.mask_feature[1]]
        else:
            valid_data_mask = np.ones((time, height, width), dtype=bool)

        all_parameters = np.zeros((height, width, 7))

        for height_ind, width_ind in it.product(range(height), range(width)):
            valid_curve = data[:, height_ind, width_ind][valid_data_mask[:, height_ind, width_ind].squeeze()]
            valid_times = times[valid_data_mask[:, height_ind, width_ind].squeeze()]

            all_parameters[height_ind, width_ind] = self._fit_optimize(valid_times, valid_curve)

        eopatch.data_timeless[self.new_feature] = all_parameters

        return eopatch
