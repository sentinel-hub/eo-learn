"""
Utility function for image co-registration

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
# pylint: disable=invalid-name

import logging

import numpy as np
import scipy

LOGGER = logging.getLogger(__name__)


def ransac(npts, model, n, k, t, d):
    """ Fit model parameters to data using the RANSAC algorithm

    This implementation is written from pseudo-code found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    :param npts: A set of observed data points
    :param model: A model that can be fitted to data points
    :param n: The minimum number of data values required to fit the model
    :param k: The maximum number of iterations allowed in the algorithm
    :param t: A threshold value for determining when a data point fits a model
    :param d: The number of close data values required to assert that a model fits well to data
    :return: Model parameters which best fit the data (or None if no good model is found)
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    # best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, npts)
        maybemodel = model.fit(maybe_idxs)
        test_err = model.score(test_idxs, maybemodel)
        also_idxs = test_idxs[test_err < t]  # select indices of rows with accepted points

        LOGGER.debug('test_err.min() %f', test_err.min() if test_err.size else None)
        LOGGER.debug('test_err.max() %f', test_err.max() if test_err.size else None)
        LOGGER.debug('numpy.mean(test_err) %f', np.mean(test_err) if test_err.size else None)
        LOGGER.debug('iteration %d, len(alsoinliers) = %d', iterations, len(also_idxs))

        if len(also_idxs) > d:
            betteridxs = np.concatenate((maybe_idxs, also_idxs))
            bettermodel = model.fit(betteridxs)
            better_errs = model.score(betteridxs, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                # best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    return bestfit


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class EstimateEulerTransformModel:
    """ Estimate Euler transform linear system solved using linear least squares

        This class estimates an Euler 2D transformation between two cloud of 2D points using SVD decomposition
    """
    def __init__(self, src_pts, trg_pts):
        """ Initialise target and source cloud points as Nx2 matrices. The transformation aligning source points to
        target points is estimated.

        :param src_pts: Array of source points
        :param trg_pts: Array of target points
        """
        self.src_pts = src_pts
        self.trg_pts = trg_pts

    def estimate_rigid_transformation(self, idx):
        """ Estimate rigid transformation given a set of indices

        :param idx: Array of indices used to estimate the transformation
        :return: Estimated transformation matrix
        """
        # Look at points of given indices only
        src_pts = self.src_pts[idx, :]
        trg_pts = self.trg_pts[idx, :]
        # Get centroid location
        src_centroid, trg_centroid = np.mean(src_pts, axis=0), np.mean(trg_pts, axis=0)
        # Centre sets of points
        src_pts_centr = src_pts - src_centroid
        trg_pts_centr = trg_pts - trg_centroid
        # SVD decomposition
        u, _, v = np.linalg.svd(np.matmul(src_pts_centr.transpose(), trg_pts_centr))
        e = np.eye(2)
        e[1, 1] = np.linalg.det(np.matmul(v.transpose(), u.transpose()))
        # Estimate rotation matrix
        r = np.matmul(v.transpose(), np.matmul(e, u.transpose()))
        # Estimate translation
        t = trg_centroid - np.matmul(r, src_centroid.transpose()).transpose()
        # Fill in transformation matrix and return
        warp_matrix = np.zeros((2, 3))
        warp_matrix[:2, :2] = r
        warp_matrix[:, 2] = t
        return warp_matrix

    def fit(self, idx):
        """ Estimate Euler transform on points listed in `idx`

        :param idx: Indices used to estimate transformation
        :return: Transformation matrix
        """
        x = self.estimate_rigid_transformation(idx)
        return x

    def score(self, idx, warp_matrix):
        """ Estimate the registration error of estimated transformation matrix

        :param idx: List of points used to estimate the transformation
        :param warp_matrix: Matrix estimating Euler trasnformation
        :return: Square root of Target Registration Error
        """
        # Transform source points with estimated transformation
        trg_fit = scipy.dot(warp_matrix, np.concatenate((self.src_pts[idx, :], np.ones((len(idx), 1))), axis=1).T).T
        # Compute error in transformation
        err_per_point = np.sqrt(np.sum((self.trg_pts[idx, :] - trg_fit[:, :2])**2, axis=1))  # sum squared error per row
        return err_per_point
