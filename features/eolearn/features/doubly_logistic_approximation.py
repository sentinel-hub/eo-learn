import numpy as np
from eolearn.core import EOTask
from scipy.optimize import curve_fit
import multiprocessing
import itertools as it


def doubly_logistic(x, c1, c2, a1, a2, a3, a4, a5):
    return c1 + c2 * np.piecewise(x, [x < a1, x >= a1],
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
    :param p0: Initial parameter guess
    :type p0: List of floats length 7 corresponding to each parameter
    """

    def __init__(self, feature, new_feature='DOUBLY_LOGISTIC_PARAM', p0=None, mask_data=False):
        self.feature = feature
        self.p0 = p0
        self.new_feature = new_feature
        self.mask_data = mask_data

    def _pool_wrapper(self, parameters):
        """
        Wrapper for _fit_optimize function because Pool accepts only a one dimensional array but _fit_optimize needs two
        :param parameters: Tuple with two lists of floats corresponding to horizontal and vertical axis
        :return: List of optimized parameters [c1, c2, a1, a2, a3, a4, a5]
        """
        return self._fit_optimize(*parameters)

    def _fit_optimize(self, x, y):
        """
        :param x: Horizontal coordinates of points
        :type x: List of floats
        :param y: Vertical coordinates of points
        :type y: List of floats
        :return: List of optimized parameters [c1, c2, a1, a2, a3, a4, a5]
        """
        bounds_lower = [np.min(y), -np.inf, x[0], 0.15, 1, 0.15, 1, ]
        bounds_upper = [np.max(y), np.inf, x[-1], np.inf, np.inf, np.inf, np.inf, ]
        if self.p0 is None:
            self.p0 = [np.mean(y), 0.2, (x[-1] - x[0]) / 2, 0.15, 10, 0.15, 10]
        optimal_values = curve_fit(doubly_logistic, x, y, self.p0,
                                   bounds=(bounds_lower, bounds_upper), maxfev=1000000,
                                   absolute_sigma=True)
        return optimal_values[0]

    def execute(self, eopatch):
        data = eopatch.data[self.feature].squeeze(axis=3)

        times = np.asarray([time.toordinal() for time in eopatch.timestamp])
        times = (times - times[0]) / (times[-1] - times[0])

        t, h, w = data.shape

        if self.mask_data:
            valid_data_mask = eopatch.mask['VALID_DATA']
        else:
            valid_data_mask = np.ones((t, h, w), dtype=bool)

        valid_parameters = []

        # Make an array that can be passed to pool function
        for ih, iw in it.product(range(h), range(w)):
            valid_parameters.append((times[valid_data_mask[:, ih, iw].squeeze()],
                                     data[:, ih, iw][valid_data_mask[:, ih, iw].squeeze()]))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        all_parameters = pool.map(self._pool_wrapper, valid_parameters)

        all_parameters = np.reshape(all_parameters, (h, w, 7))

        eopatch.data_timeless[self.new_feature] = all_parameters

        return eopatch
