"""
Tasks for spatial sampling of points for building training/validation samples for example.
"""

import numpy as np

from shapely.geometry import Polygon, Point
from shapely.geometry import LinearRing
from shapely.ops import triangulate

from rasterio.features import shapes

from eolearn.core import EOTask, FeatureType

from skimage.morphology import binary_erosion, disk

import collections
import functools
import math


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
                           for index, (shp, label) in enumerate(shapes(raster_mask, mask=None))
                           if (int(label) is not no_data_value) and (int(label) not in ignore_labels)]

        self.areas = np.asarray([entry['polygon'].area for entry in self.geometries])
        self.decomposition = [triangulate(entry['polygon']) for entry in self.geometries]

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
        rx, ry = (1 - math.sqrt(r1)) * np.asarray(A) \
                 + math.sqrt(r1) * (1 - r2) * np.asarray(B) \
                 + math.sqrt(r1) * r2 * np.asarray(C)
        if use_int_coords:
            rx, ry = math.round(rx), math.round(ry)
            return Point(int(rx), int(ry))
        return Point(rx, ry)


class PointRasterSampler:
    """ Class to perform point sampling of a label image

        Class that handles sampling of points from a label image representing classification labels. Labels are
        encoded as `uint8` and the raster is a 2D or single-channel 3D array.

        Supported operations include:
         * erosion of label classes to remove border effects
         * exclusion of some labels from sampling
         * sampling based on label frequency in raster or even sampling of labels (i.e. over-sampling)

    """
    def __init__(self, disk_radius=None, even_sampling=False, no_data_value=None, ignore_labels=None):
        """ Initialisation of sampler parameters

        :param disk_radius: Radius of disk used as structure element for erosion. If `None`, no erosion is performed.
                            Default is `None`
        :type disk_radius: uint8 or None
        :param even_sampling: Whether to sample class labels evenly or not. If `True`, labels will have the same number
                                samples, with less frequent labels being over-sampled (i.e. same observation is sampled
                                multiple times). If `False`, sampling follows the label distribution in raster.
                                Default is `False`
        :type even_sampling: bool
        :param no_data_value: Label value denoting no data. This value will not be sampled. Default is `None`
        :type no_data_value: None or uint8
        :param ignore_labels: List of label values to ignore during sampling. Default is `None`
        :type ignore_labels: None or list of uint8
        """
        self.disk_radius = disk_radius
        self.ignore_labels = [] if ignore_labels is None else ignore_labels
        self.even_sampling = even_sampling
        self.no_data_value = no_data_value

    @staticmethod
    def _erosion(image, label, struct_elem):
        """ Perform erosion of the binary mask corresponding to `label`

        If `struct_elem` is defined, erosion of the binary mask corresponding to `label` is returned, otherwise
        the un-eroded binary mask is returned

        :param image: Input 2D raster label image
        :type image: uint8 numpy array
        :param label: Scalar value of label to consider
        :type label: uint8
        :param struct_elem: Structuring element used for erosion
        :type struct_elem: uint8 numpy array
        :return: Binary mask corresponding to label. If `struct_elem` is defined, the binary mask is eroded
        :rtype: uint8 numpy array
        """
        if struct_elem is not None:
            return binary_erosion(image == label, struct_elem).astype(np.uint16)
        return (image == label).astype(np.uint16)

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
        replace = True if label_count < n_samples_per_label else False
        rand_idx = np.random.choice(len(h_idx), size=n_samples_per_label, replace=replace)
        # Here we rescale the labels as per input
        return image[h_idx[rand_idx], w_idx[rand_idx]] - 1, h_idx[rand_idx], w_idx[rand_idx]

    def sample(self, raster, n_samples=1000):
        """ Sample `nsamples` points form raster

        :param raster: Input 2D or single-channel 3D label image
        :type raster: uint8 numpy array
        :param n_samples: Number of points to sample in total
        :type n_samples: uint32
        :return: Sampled label values, row index of samples, col index of samples
        """
        # check dimensionality and reshape to 2D
        if raster.ndim == 3 and raster.shape[-1] == 1:
            raster = raster.squeeze()
        elif raster.ndim == 2:
            pass
        else:
            raise ValueError('Class operates on 2D or 3D single-channel raster images')

        # trick to deal with label of value 0. Raster is casted to uint16 not to overflow incase 255 classes are used
        raster = raster.astype(np.uint16) + 1
        no_data_value = self.no_data_value + 1 if self.no_data_value is not None else self.no_data_value
        ignore_labels = [il + 1 for il in self.ignore_labels] if self.ignore_labels else self.ignore_labels

        # find unique labels
        unique_labels = np.unique(raster)

        # structured element for binary erosion
        struct_elem = disk(self.disk_radius) if self.disk_radius is not None else None

        # filter out unwanted labels and no data values and apply erosion if required
        raster_filtered = np.sum(np.asarray([lbl * self._erosion(raster, lbl, struct_elem)
                                             for lbl in unique_labels
                                             if (lbl != no_data_value) and (lbl not in ignore_labels)]),
                                 axis=0)

        # update unique labels and compute counts for each filtered label
        unique_labels, label_count = np.unique(raster_filtered, return_counts=True)

        # 0 now has all pixels to ignore due to erosion or no_data_value or ignore labels
        zero_index, = np.where(unique_labels == 0)
        zero_index = 0 if zero_index.size == 0 else zero_index[0] + 1

        # define number of samples per filtered label
        n_valid_labels = len(unique_labels[zero_index:])
        if self.even_sampling:
            # valid labels have the same number of samples
            n_samples_per_label = np.diff(np.round(np.linspace(0, n_samples, num=n_valid_labels + 1))).astype(np.uint32)
        else:
            # number of samples per label is proportional to label frequency
            label_ratio = label_count[zero_index:] / np.sum(label_count[zero_index:])
            n_samples_per_label = (np.ceil(n_samples * label_ratio)).astype(np.uint32)

        # sample raster
        samples = np.concatenate([self._binary_sample(raster, lbl, n_sample_lbl, lbl_count)
                                  for lbl, n_sample_lbl, lbl_count in
                                  zip(unique_labels[zero_index:], n_samples_per_label, label_count[zero_index:])],
                                 axis=1).T

        # shuffle to mix labels in case they are fed directly to train a ML model
        np.random.shuffle(samples)

        # return value, row index and col index. Return exactly `n_sample` values
        return samples[:n_samples, 0].astype(np.uint8), samples[:n_samples, 1], samples[:n_samples, 2]


class PointSamplingTask(EOTask):
    """
    Task for sampling points.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, nsamples,
                 out_feature_type, out_feature_name, out_truth_name,
                 sample_raster_feature_type, sample_raster_name,
                 data_feature_type, data_feature_name,
                 no_data_value=None, ignore_labels=None):
        self.nsamples = nsamples
        self.out_feature_type = out_feature_type
        self.out_feature_name = out_feature_name
        self.out_truth_name = out_truth_name
        self.sample_raster_feature_type = sample_raster_feature_type
        self.sample_raster_name = sample_raster_name
        self.data_feature_type = data_feature_type
        self.data_feature_name = data_feature_name
        self.no_data_value = no_data_value
        self.ignore_labels = ignore_labels

    def execute(self, eopatch):
        raster = eopatch.get_feature(self.sample_raster_feature_type, self.sample_raster_name)
        data = eopatch.get_feature(self.data_feature_type, self.data_feature_name)

        sampler = PointSampler(raster, no_data_value=self.no_data_value, ignore_labels=self.ignore_labels)

        labels, rows, cols = sampler.sample(self.nsamples)
        labels = np.asarray(labels)

        sampled_points = data[:, rows, cols, :]

        eopatch.add_feature(self.out_feature_type, self.out_feature_name, sampled_points[:, :, np.newaxis, :])
        eopatch.add_feature(FeatureType.LABEL_TIMELESS, self.out_truth_name, labels[:, np.newaxis])

        return eopatch
