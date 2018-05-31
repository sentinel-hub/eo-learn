"""
Tasks for spatial sampling of points for building training/validation samples for example.
"""

import numpy as np

from shapely.geometry import Polygon, Point
from shapely.geometry import LinearRing
from shapely.ops import triangulate

from rasterio.features import shapes

from eolearn.core import EOTask, FeatureType

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
        :param type: integer
        :param weighted: flag to apply weights proportional to total area of each class/polygon when sampling
        :param weighted: bool, default is True
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
