"""
Tasks for spatial sampling of points for building training/validation samples for example.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import collections
import functools
import logging
from math import sqrt

import numpy as np
import rasterio.features
import shapely.ops
from shapely.geometry import Polygon, Point, LinearRing

from eolearn.core import EOTask, EOPatch, FeatureType

LOGGER = logging.getLogger(__name__)


class PointSampler:
    """
    Samples randomly points from a raster mask, where the number of points sampled from a polygon with specific label
    is proportional to its area.

    The sampler first vectorizes the raster mask and then samples.

    :param raster_mask: A raster mask based on which the points are sampled.
    :type raster_mask: A numpy array of shape (height, width) and type int.
    :param no_data_value: A value indicating no data value -- points that are not labeled and should not be sampled
    :type no_data_value: integer
    :param ignore_labels: A list of label values that should not be sampled.
    :type ignore_labels: list of integers
    """
    # pylint: disable=invalid-name
    def __init__(self, raster_mask, no_data_value=None, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = []

        self.geometries = [{'label': int(label),
                            'polygon': Polygon(LinearRing(shp['coordinates'][0]),
                                               [LinearRing(pts) for pts in shp['coordinates'][1:]])}
                           for index, (shp, label) in enumerate(rasterio.features.shapes(raster_mask, mask=None))
                           if (int(label) is not no_data_value) and (int(label) not in ignore_labels)]

        self.areas = np.asarray([entry['polygon'].area for entry in self.geometries])
        self.decomposition = [shapely.ops.triangulate(entry['polygon']) for entry in self.geometries]

        self.label2cc = collections.defaultdict(list)
        for index, entry in enumerate(self.geometries):
            self.label2cc[entry['label']].append(index)

    def __len__(self):
        """
        Returns number of polygons from vectorization the raster mask.
        """
        return len(self.geometries)

    def labels(self):
        """
        Returns all label values found in the raster mask (except for the no_data_value and label values from
        ignore_labels).
        """
        return self.label2cc.keys()

    def area(self, cc_index=None):
        """
        Returns the area of the selected polygon if index is provided or of all polygons if it's not.
        """
        if cc_index is not None:
            return self.areas[cc_index]
        return np.sum(self.areas)

    def sample(self, nsamples=1, weighted=True):
        """
        Sample n points from the provided raster mask. The number of points belonging
        to each class is proportional to the area this class covers in the raster mask,
        if weighted is set to True.

        TODO: If polygon has holes the number of sampled points will be less than nsamples

        :param nsamples: number of sampled samples
        :type nsamples: integer
        :param weighted: flag to apply weights proportional to total area of each class/polygon when sampling
        :type weighted: bool, default is True
        """
        weights = self.areas / np.sum(self.areas) if weighted else None
        index = np.random.choice(a=len(self.geometries), size=nsamples, p=weights)

        labels = []
        rows = []
        cols = []
        for idx in index:
            polygon = self.geometries[idx]['polygon']
            label = self.geometries[idx]['label']
            point = PointSampler.random_point(polygon.envelope.bounds)
            if PointSampler.contains(polygon, point):
                labels.append(label)
                rows.append(int(point.y))
                cols.append(int(point.x))
                # samples.append({'label':label, 'row':point.y, 'col':point.x})

        return labels, rows, cols

    def sample_cc(self, nsamples=1, weighted=True):
        """
        Returns a random polygon of any class. The probability of each polygon to be sampled
        is proportional to its area if weighted is True.
        """
        weights = self.areas / np.sum(self.areas) if weighted else None
        for index in np.random.choice(a=len(self.geometries), size=nsamples, p=weights):
            yield self.geometries[index]

    def sample_within_cc(self, cc_index, nsamples=1):
        """
        Returns randomly sampled points from a polygon.

        Complexity of this procedure is (A/a * nsamples) where A=area(bbox(P))
        and a=area(P) where P is the polygon of the connected component cc_index
        """
        polygon = self.geometries[cc_index]['polygon']
        samples = []
        while len(samples) < nsamples:
            point = PointSampler.random_point(polygon.envelope.bounds)
            if PointSampler.contains(polygon, point):
                samples.append(point)
        return samples

    @staticmethod
    def contains(polygon, point):
        """
        Tests whether point lies within the polygon
        """
        in_hole = functools.reduce(
            lambda P, Q: P and Q,
            [interior.covers(point) for interior in polygon.interiors]
        ) if polygon.interiors else False
        return polygon.covers(point) and not in_hole

    @staticmethod
    def random_point(bounds):
        """ Selects a random point in interior of a rectangle

        :param bounds: Rectangle coordinates (x_min, y_min, x_max, y_max)
        :type bounds: tuple(float)
        :return: Random point from interior of rectangle
        :rtype: shapely.geometry.Point
        """
        return Point(PointSampler.random_coords(bounds))

    @staticmethod
    def random_coords(bounds):
        """ Selects a random point in interior of a rectangle

        :param bounds: Rectangle coordinates (x_min, y_min, x_max, y_max)
        :type bounds: tuple(float)
        :return: Random point from interior of rectangle
        :rtype: tuple of x and y coordinates
        """
        x_min, y_min, x_max, y_max = bounds
        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)
        return x, y

    @staticmethod
    def random_point_triangle(triangle, use_int_coords=True):
        """
        Selects a random point in interior of a triangle
        """
        xs, ys = triangle.exterior.coords.xy
        A, B, C = zip(xs[:-1], ys[:-1])
        r1, r2 = np.random.rand(), np.random.rand()
        rx, ry = (1 - sqrt(r1)) * np.asarray(A) + sqrt(r1) * (1 - r2) * np.asarray(B) + sqrt(r1) * r2 * np.asarray(C)
        if use_int_coords:
            rx, ry = round(rx), round(ry)
            return Point(int(rx), int(ry))
        return Point(rx, ry)


class PointRasterSampler:
    """ Class to perform point sampling of a label image

        Class that handles sampling of points from a label image representing classification labels. Labels are
        encoded as `uint8` and the raster is a 2D or single-channel 3D array.

        Supported operations include:
         * exclusion of some labels from sampling
         * sampling based on label frequency in raster or even sampling of labels (i.e. over-sampling)

    """
    def __init__(self, labels, even_sampling=False):
        """ Initialisation of sampler parameters

        :param labels: A list of labels that will be sampled
        :type labels: list(int)
        :param even_sampling: Whether to sample class labels evenly or not. If `True`, labels will have the same number
                                samples, with less frequent labels being over-sampled (i.e. same observation is sampled
                                multiple times). If `False`, sampling follows the label distribution in raster.
                                Default is `False`
        :type even_sampling: bool
        """
        self.labels = labels
        self.even_sampling = even_sampling

    def _get_unknown_value(self):
        """ Finds the smallest integer value >=0 that is not in `labels`

        :return: Value that is not in the labels
        :rtype: int
        """
        label_set = set(self.labels)
        value = 0
        while value in label_set:
            value += 1
        return value

    @staticmethod
    def _binary_sample(image, label, n_samples_per_label, label_count):
        """ Sample `nsamples_per_label` points from the binary mask corresponding to `label`

        Randomly sample `nsamples_per_label` point form the binary mask corresponding to `label`. Sampling with
        replacement is used if the required `nsamples_per_label` is larger than the available `label_count`

        :param image: Input 2D raster label image
        :type image: uint8 numpy array
        :param label: Scalar value of label to consider
        :type label: uint8
        :param n_samples_per_label: Number of points to sample form the binary mask
        :type n_samples_per_label: uint32
        :param label_count: Number of points available for `label`
        :type label_count: uint32
        :return: Sampled label value, row index of samples, col index of samples
        """
        h_idx, w_idx = np.where(image == label)

        rand_idx = np.random.choice(h_idx.size, size=n_samples_per_label, replace=label_count < n_samples_per_label)

        return h_idx[rand_idx], w_idx[rand_idx]

    def sample(self, raster, n_samples=1000):
        """ Sample `nsamples` points form raster

        :param raster: Input 2D or single-channel 3D label image
        :type raster: uint8 numpy array
        :param n_samples: Number of points to sample in total
        :type n_samples: uint32
        :return: List of row indices of samples, list of column indices of samples
        :rtype: numpy.array, numpy.array
        """
        # Check dimensionality and reshape to 2D
        raster = raster.copy()
        if raster.ndim == 3 and raster.shape[-1] == 1:
            raster = raster.squeeze()
        elif raster.ndim != 2:
            raise ValueError('Class operates on 2D or 3D single-channel raster images')

        # Calculate mask of all pixels which can be sampled
        mask = np.zeros(raster.shape, dtype=np.bool)
        for label in self.labels:
            label_mask = (raster == label)
            mask |= label_mask

        unique_labels, label_count = np.unique(raster[mask], return_counts=True)
        if not unique_labels.size:
            LOGGER.warning('No samples matching given parameters found in EOPatch')
            return np.empty((0,), dtype=np.uint32), np.empty((0,), dtype=np.uint32)

        if self.even_sampling:
            # Valid labels have the same (or off by one) number of samples
            n_samples_per_label = np.diff(np.round(np.linspace(0, n_samples,
                                                               num=label_count.size + 1))).astype(np.uint32)
        else:
            # Number of samples per label is proportional to label frequency
            label_ratio = label_count / np.sum(label_count)
            n_samples_per_label = (np.ceil(n_samples * label_ratio)).astype(np.uint32)

        # Apply mask to raster
        unknown_value = self._get_unknown_value()
        raster[~mask] = unknown_value
        if not np.array_equal(~mask, raster == unknown_value):
            raise ValueError('Failed to set unknown value. Too many labels for sampling reference mask of type '
                             '{}'.format(raster.dtype))

        # Sample raster
        samples = np.concatenate([self._binary_sample(raster, label, n_sample_label, label_count)
                                  for label, n_sample_label, label_count in zip(unique_labels, n_samples_per_label,
                                                                                label_count)], axis=1).T

        # Shuffle to mix labels in case they are fed directly to train a ML model
        np.random.shuffle(samples)

        # Return row index and col index. Return exactly `n_sample` values
        return samples[:n_samples, 0], samples[:n_samples, 1]


class PointSamplingTask(EOTask):
    """ Task for spatially sampling points from a time-series.

    This task performs random spatial sampling of a time-series based on a label mask. The user specifies the number of
    points to be sampled, the name of the `DATA` time-series, the name of the label raster image, and the name of the
    output sample features and sampled labels.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, n_samples, ref_mask_feature, ref_labels, sample_features, return_new_eopatch=False,
                 **sampling_params):
        """ Initialise sampling task.

        The data to be sampled is supposed to be a time-series stored in `DATA` type of the eopatch, while the raster
        image is supposed to be stored in `MASK_TIMELESS`. The output sampled features are stored in `DATA` and have
        shape T x N_SAMPLES x 1 x D, where T is the number of time-frames, N_SAMPLES the number of random samples, and D
        is the number of channels of the input time-series.

        The row and column index of sampled points can also be stored in the eopatch, to allow the same random sampling
        of other masks.

        :param n_samples: Number of random spatial points to be sampled from the time-series
        :type n_samples: int
        :param ref_mask_feature: Name of `MASK_TIMELESS` raster image to be used as a reference for sampling
        :type ref_mask_feature: str
        :param ref_labels: List of labels of `ref_mask_feature` mask which will be sampled
        :type ref_labels: list(int)
        :param sample_features: A collection of features that will be resampled. Each feature is represented by a tuple
            in a form of `(FeatureType, 'feature_name')` or
            (FeatureType, '<feature_name>', '<sampled feature name>'). If `sampled_feature_name` is not
            set the default name `'<feature_name>_SAMPLED'` will be used.

            Example: [(FeatureType.DATA, 'NDVI'), (FeatureType.MASK, 'cloud_mask', 'cloud_mask_1')]

        :type sample_features: list(tuple(FeatureType, str, str) or tuple(FeatureType, str))
        :param return_new_eopatch: If `True` the task will create new EOPatch, put sampled data and copy of timestamps
            and meta_info data in it and return it. If `False` it will just add sampled data to input EOPatch and
            return it.
        :type return_new_eopatch: bool
        :param sampling_params: Any other parameter used by `PointRasterSampler` class
        """
        self.n_samples = n_samples
        self.ref_mask_feature = self._parse_features(ref_mask_feature, default_feature_type=FeatureType.MASK_TIMELESS)
        self.ref_labels = list(ref_labels)
        self.sample_features = self._parse_features(sample_features, new_names=True,
                                                    rename_function='{}_SAMPLED'.format)
        self.return_new_eopatch = return_new_eopatch
        self.sampling_params = sampling_params

    def execute(self, eopatch, seed=None):
        """ Execute random spatial sampling of time-series stored in the input eopatch

        :param eopatch: Input eopatch to be sampled
        :type eopatch: EOPatch
        :param seed: Setting seed of random sampling. If None no seed will be used.
        :type seed: int or None
        :return: An EOPatch with spatially sampled temporal features and associated labels
        :type eopatch: EOPatch
        """
        np.random.seed(seed)

        # Retrieve data and raster label image from eopatch
        f_type, f_name = next(self.ref_mask_feature(eopatch))
        raster = eopatch[f_type][f_name]

        # Initialise sampler
        sampler = PointRasterSampler(self.ref_labels, **self.sampling_params)
        # Perform sampling
        rows, cols = sampler.sample(raster, n_samples=self.n_samples)

        if self.return_new_eopatch:
            new_eopatch = EOPatch()
            new_eopatch.timestamp = eopatch.timestamp[:]
            new_eopatch.bbox = eopatch.bbox  # Should be copied
            new_eopatch.meta_info = eopatch.meta_info.copy()  # Should be deep copied - implement this in core
        else:
            new_eopatch = eopatch

        # Add sampled features
        for feature_type, feature_name, new_feature_name in self.sample_features(eopatch):

            if feature_type.is_time_dependent():
                sampled_data = eopatch[feature_type][feature_name][:, rows, cols, :]
            else:
                sampled_data = eopatch[feature_type][feature_name][rows, cols, :]
            new_eopatch[feature_type][new_feature_name] = sampled_data[..., np.newaxis, :]

        return new_eopatch
