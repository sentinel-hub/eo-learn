import numpy as np
from eolearn.core import EOTask, FeatureType
from scipy.optimize import curve_fit


def _optimizing_function(x, c1, c2, a1, a2, a3, a4, a5):
    """Double logistic function which is needed for scipy.optimize
    :param x, c1, c2, a1, a2, a3, a4, a5: All inputs are the parameters which are optimized
    """

    def left(y):
        return np.exp(-((a1 - y) / a4) ** a5)

    def right(y):
        return np.exp(-((y - a1) / a2) ** a3)

    return c1 + c2 * np.piecewise(x, [x < a1, x >= a1], [left, right])


def _fit_optimize(x, y, p0=None):
    """
    :param x: Vertical coordinates of points
    :type x: List of floats
    :param y: Horizontal coordinates of points
    :type y: List of floats
    :param p0: Initial parameter guess
    :type p0: List of floats
    :return: List of optimized parameters [c1, c2, a1, a2, a3, a4, a5]
    """
    # Normalization
    norm = (max(x) - min(x))
    x = x / norm

    bounds_lower = [-1, -np.inf, x[0], 0.15, 1, 0.15, 1, ]
    bounds_upper = [1, np.inf, x[-1], np.inf, np.inf, np.inf, np.inf, ]
    # bounds=(bounds_lower, bounds_upper),
    if p0 is None:
        p0 = [np.mean(x / norm), 0.2, 0.5, 0.15, 10, 0.15, 10]
    optimal_values = curve_fit(_optimizing_function, x, y, bounds=(bounds_lower, bounds_upper), maxfev=1000000,
                               absolute_sigma=True)
    return optimal_values[0]


class DoublyLogisticApproximationTask(EOTask):
    """
    EOTask class for calculation of doubly logistic approximation on each pixel for a feature. The task creates new
    feature with the function parameters for each pixel as vectors.

    :param feature: A feature on which the function will be approximated
    :type feature: str
    :param new_feature: Name of the new feature where parameters of the function are saved
    :type new_feature: str
    :param p0: Initial parameter guess
    :type p0: List of floats length 7 corresponding to each parameter
    """

    def __init__(self, feature, new_feature='DOUBLY_LOGISTIC_PARAM', p0=None):
        self.feature = feature
        self.p0 = p0
        self.new_feature = new_feature

    def execute(self, eopatch):
        data = eopatch.data[self.feature].squeeze(axis=3)
        print(data)
        t, width, height = data.shape
        all_parameters = np.zeros((width, height, 7))

        for i in range(width):
            for j in range(height):
                all_parameters[i, j, :] = _fit_optimize(data[:, i, j], range(t), self.p0)

        all_parameters = np.expand_dims(all_parameters, axis=3)
        eopatch.add_feature(FeatureType.DATA, self.new_feature, all_parameters)

        return eopatch
