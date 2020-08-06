"""
Tasks for spatial sampling of points for building training/validation samples for example.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import collections
import functools
import logging
from math import sqrt
import random
import pandas as pd
from sklearn.utils import resample
import numpy as np
import rasterio.features
import shapely.ops
from shapely.geometry import Polygon, Point, LinearRing

from eolearn.core import EOTask, EOPatch, FeatureType, FeatureParser

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


class BalancedClassSampler:
    """
    A class that samples points from multiple patches and returns a balanced set depending on the class label.
    This is done by sampling the desired amount on each patch and then balancing the data based on the smallest class
    or amount. If the amount is provided and there are classes with less than that number of points, random
    point are duplicated to reach the necessary size.
    """

    def __init__(self, class_feature, samples_amount=0.1, valid_mask=None,
                 ignore_labels=None, features=None, weak_classes=None, search_radius=3,
                 samples_per_class=None, seed=None):
        """
        :param class_feature: Feature that contains class labels
        :type class_feature: (FeatureType, string) or string
        :param samples_amount: Number of samples taken per patch. If the number is on the interval of [0...1] then that
            percentage of all points is taken. If the value is 1 all eligible points are taken.
        :type samples_amount: float
        :param valid_mask: Feature that defines the area from where samples are
            taken, if None the whole image is used
        :type valid_mask: (FeatureType, string), string or None
        :param ignore_labels: A single item or a list of values that should not be sampled.
        :type ignore_labels: list of integers or int
        :param features: Temporal features to include in dataset for each pixel sampled
        :type features: Input that the FeatureParser class can parse
        :param samples_per_class: Number of samples per class returned after
            balancing. If the number is higher than total amount of some classes, then those classes have their
            points duplicated to reach the desired amount. If the argument is None then limit is set to the size of
            the number of samples of the smallest class
        :type samples_per_class: int or None
        :param seed: Seed for random generator
        :type seed: int
        :param weak_classes: Classes that upon finding, also the neighbouring regions
            will be checked and added if they contain one of the weak classes. Used to enrich the samples
        :type weak_classes: list of integers or int
        :param search_radius: How many points in each direction to check for additional weak classes
        :type search_radius: int
        """
        self.class_feature = next(FeatureParser(class_feature, default_feature_type=FeatureType.MASK_TIMELESS)())
        self.samples_amount = samples_amount
        self.valid_mask = next(
            FeatureParser(valid_mask, default_feature_type=FeatureType.MASK_TIMELESS)()) if valid_mask else None
        self.ignore_labels = ignore_labels if isinstance(ignore_labels, list) else [ignore_labels]
        self.features = FeatureParser(features, default_feature_type=FeatureType.DATA_TIMELESS) if features else None
        self.columns = [self.class_feature[1]] + ['patch_identifier', 'x', 'y']
        if features:
            self.columns += [x[1] for x in self.features]

        self.samples_per_class = samples_per_class
        self.seed = seed
        if seed is not None:
            random.seed(self.seed)
        self.weak_classes = weak_classes if isinstance(weak_classes, list) else [weak_classes]
        self.search_radius = search_radius
        self.sampled_data = []
        self.balanced_data = None
        self.class_distribution = None

    def sample(self, eopatch, patch_identifier):
        """
        Collects samples on eopatch. This method does not modify the patch. Once the desired patches are sampled
        the samples can be balanced and returned by calling get_balanced_data() function
        :param eopatch: eopatch on which the sampling is performed
        :param patch_identifier: Patch identifier that is saved along with the samples
        """

        height, width, _ = eopatch[self.class_feature].shape
        mask = eopatch[self.valid_mask].squeeze() if self.valid_mask else None
        total_points = height * width
        no_samples = self.samples_amount if self.samples_amount > 1 else int(total_points * self.samples_amount)
        no_samples = min(total_points, no_samples)

        # Finds all the pixels which are not masked or ignored
        subsample_id = []
        for loc_h in range(height):
            for loc_w in range(width):
                if mask is not None and not mask[loc_h][loc_w]:
                    continue
                if self.ignore_labels is not None \
                        and eopatch[self.class_feature][loc_h][loc_w][0] in self.ignore_labels:
                    continue
                subsample_id.append((loc_h, loc_w))

        subsample_id = random.sample(
            subsample_id,
            min(no_samples, len(subsample_id))
        )

        for loc_h, loc_w in subsample_id:
            class_value = eopatch[self.class_feature][loc_h][loc_w][0]

            point_data = [(self.class_feature[1], class_value)] + [('patch_identifier', patch_identifier),
                                                                   ('x', loc_h),
                                                                   ('y', loc_w)]
            if self.features:
                point_data += [(f[1], float(eopatch[f][loc_h][loc_w])) for f in self.features]

            self.sampled_data.append(dict(point_data))

            # If the point belongs to one of the weak classes, additional sampling is done in the neighbourhood
            if self.weak_classes and self.search_radius and (class_value in self.weak_classes):
                self.local_enrichment(loc_h, loc_w, eopatch, patch_identifier)

    def sample_folder(self, folder, **kwargs):
        """
        Samples all the patches in specified folder
        :param folder: location of folder which contains patches to be sampled
        :param kwargs: key word arguments that are passed to EOPatch.load()
        """
        self.sample_patch_list([f'{folder}/{name}' for name in next(os.walk(folder))[1]], **kwargs)

    def sample_patch_list(self, patch_list, **kwargs):
        """
        Samples patches on specified locations
        :param patch_list: location of patches to be sampled
        :param kwargs: key word arguments that are passed to EOPatch.load()
        """
        for patch_location in patch_list:
            eopatch = EOPatch.load(patch_location, **kwargs)
            patch_identifier = os.path.basename(patch_location)
            self.sample(eopatch, patch_identifier)

    def sample_patch(self, patch_location, **kwargs):
        """
        Samples a patch on specified location
        :param patch_location: location of patch to be sampled
        :param kwargs: key word arguments that are passed to EOPatch.load()
        """
        eopatch = EOPatch.load(patch_location, **kwargs)
        patch_identifier = os.path.basename(patch_location)
        self.sample(eopatch, patch_identifier)

    def local_enrichment(self, loc_h, loc_w, eopatch, patch_identifier):
        """
        Class that performs additional search on the patch around specified location. All new found points are saved
        in self.sampled_data
        :param loc_h: Starting vertical location of search
        :type loc_h: int
        :param loc_w: Starting horizontal location of search
        :type loc_w: int
        :param eopatch: EOPatch on which to perform the search
        :param patch_identifier: Patch identifier used to save along the found points
        :type patch_identifier: string
        """
        height, width, _ = eopatch[self.class_feature].shape
        neighbours = list(range(-self.search_radius, self.search_radius + 1))
        for vertical_shift in neighbours:
            for horizontal_shift in neighbours:
                if vertical_shift != 0 or horizontal_shift != 0:
                    search_h = loc_h + vertical_shift
                    search_w = loc_w + horizontal_shift
                    max_h, max_w = height, width
                    # Check bounds
                    if search_h >= max_h or search_w >= max_w \
                            or search_h <= 0 or search_w <= 0:
                        continue
                    # Check if the point is masked
                    if self.valid_mask:
                        mask = eopatch[self.valid_mask].squeeze()
                        if not mask[search_h, search_w]:
                            continue

                    point_feature = eopatch[self.class_feature][search_h][search_w][0]
                    if point_feature in self.weak_classes:
                        point_data = [(self.class_feature[1], point_feature)] \
                                     + [('patch_identifier', patch_identifier), ('x', search_h), ('y', search_w)]
                        if self.features:
                            point_data += [(f[1], float(eopatch[f][search_h][search_w])) for f in self.features]
                        point_data = dict(point_data)

                        # Add only if its not a duplicate point
                        if point_data not in self.sampled_data:
                            self.sampled_data.append(dict(point_data))

    def balance_data(self):
        """
        Balances the samples and stores them in self.balanced_data
        """

        all_sampled_data = pd.DataFrame(self.sampled_data, columns=self.columns)
        all_sampled_data.dropna(axis=0, inplace=True)

        # Getting the distribution of classes
        self.class_distribution = collections.Counter(all_sampled_data[self.class_feature[1]])
        class_count = self.class_distribution.most_common()

        # Setting the bound to which all classes are resampled depending on the least common class if not set previously
        duplication = False
        if self.samples_per_class is not None:
            least_common = self.samples_per_class
            # Classes will be duplicated if the limit is higher than number of classes
            duplication = True
        else:
            least_common = class_count[-1][1]

        self.balanced_data = pd.DataFrame(columns=self.columns)
        class_names = [name[0] for name in class_count]

        # Separation of all found classes into arrays used to resample
        sampled_classes_data = [all_sampled_data[all_sampled_data[self.class_feature[1]] == x] for x in class_names]
        for individual_class in sampled_classes_data:
            # Points for each class are resampled to equal number
            single_data = resample(
                individual_class,
                replace=duplication,
                n_samples=least_common,
                random_state=self.seed
            )
            self.balanced_data = self.balanced_data.append(single_data)

    def get_balanced_data(self):
        """
        Balances and returns the dataset
        :return: Balanced dataset with new index
        """
        self.balance_data()
        return self.balanced_data.reset_index(drop=True)

    def get_prior_class_distribution(self):
        """
        :return: Distribution of samples before balancing
        """
        all_sampled_data = pd.DataFrame(self.sampled_data, columns=self.columns)
        all_sampled_data.dropna(axis=0, inplace=True)
        return dict(collections.Counter(collections.Counter(all_sampled_data[self.class_feature[1]])))


class BalancedClassSamplerTask(EOTask):
    """
    Task that collects and balances samples from multiple patches
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the BalancedClassSampler class.
        :param args: arguments passed to BalancedClassSampler
        :param kwargs: key word arguments passed to BalancedClassSampler
        """
        self.balanced_sampler = BalancedClassSampler(*args, **kwargs)

    def get_balanced_data(self):
        """
        Balances and returns the dataset that was sampled previously
        :return: Balanced dataset with new index
        """
        return self.balanced_sampler.get_balanced_data()

    def get_prior_class_distribution(self):
        """
        Returns the distribution of samples before balancing. Also balances the classes if they are not already.
        :return: Distribution of samples before balancing
        """
        return self.balanced_sampler.get_prior_class_distribution()

    def execute(self, eopatch, patch_identifier='N/A'):
        """
        Collects samples on eopatch. This method does not modify the patches. Once the desired patches are sampled
        the samples can be balanced and returned by calling get_balanced_data function
        :param eopatch: Input eopatch
        :param patch_identifier: Name of patch to be stored along the samples
        :return: Unmodified input eopatch
        """
        self.balanced_sampler.sample(eopatch, patch_identifier)
        return eopatch
