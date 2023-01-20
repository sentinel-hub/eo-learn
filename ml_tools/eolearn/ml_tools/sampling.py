"""
Tasks for spatial sampling of points for building training/validation samples for example.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from abc import ABCMeta
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon

from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet
from eolearn.core.types import FeaturesSpecification, SingleFeatureSpec

_FractionType = Union[float, Dict[int, float]]


def random_point_in_triangle(triangle: Polygon, rng: Optional[np.random.Generator] = None) -> Point:
    """Selects a random point from an interior of a triangle.

    :param triangle: A triangle polygon.
    :param rng: A random numbers generator. If not provided it will be initialized without a seed.
    """
    rng = rng or np.random.default_rng()

    x_coords, y_coords = triangle.exterior.coords.xy
    vertex_a, vertex_b, vertex_c = zip(x_coords[:-1], y_coords[:-1])
    random1, random2 = rng.random(), rng.random()
    random_x, random_y = (
        (1 - sqrt(random1)) * np.asarray(vertex_a)
        + sqrt(random1) * (1 - random2) * np.asarray(vertex_b)
        + sqrt(random1) * random2 * np.asarray(vertex_c)
    )

    return Point(random_x, random_y)


def sample_by_values(
    image: np.ndarray,
    n_samples_per_value: Dict[int, int],
    rng: Optional[np.random.Generator] = None,
    replace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points from image with the amount of samples specified for each value.

    :param image: A 2-dimensional numpy array
    :param n_samples_per_value: A dictionary specifying the amount of samples per value. Values that are not in the
        dictionary will not be sampled.
    :param rng: A random numbers generator. If not provided it will be initialized without a seed.
    :param replace: Whether to sample with replacement. False means each value can only be chosen once.
    :return: A pair of numpy arrays first one containing row indices and second one containing column indices of sampled
        points.
    """
    if image.ndim != 2:
        raise ValueError(f"Given image has shape {image.shape} but sampling operates only on 2D images")

    rng = rng or np.random.default_rng()
    rows = np.empty((0,), dtype=np.int16)
    columns = np.empty((0,), dtype=np.int16)

    for value, n_samples in n_samples_per_value.items():
        sample_rows, sample_cols = rng.choice(np.nonzero(image == value), size=n_samples, replace=replace, axis=1)
        rows = np.concatenate((rows, sample_rows))
        columns = np.concatenate((columns, sample_cols))

    return rows, columns


def expand_to_grids(
    rows: np.ndarray, columns: np.ndarray, sample_size: Tuple[int, int] = (1, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """Expands sampled points into blocks and returns a pair of arrays. Each array represents a grid of indices
    of pixel locations, the first one row indices and the second one column indices. Each array is of shape
    `(N * sample_height, sample_width)`, where each element represent a row or column index in an original array
    of dimensions `(height, width)`. The grid arrays can then be used to transform the original array into a sampled
    array of shape `(N * sample_height, sample_width)`.

    :param rows: A 1-dimensional numpy array of row indices of sampled points
    :param columns: A 1-dimensional numpy array of column indices of sampled points
    :param sample_size: A size of sampled blocks to which sampled points will be extended. The given input points
        will be upper left points of created blocks.
    :return: A pair of 2-dimensional numpy arrays. The first contains a grid of row indices and the second one contains
        a grid of column indices of all points in sampled blocks
    """
    if sample_size == (1, 1):
        # A performance optimization for the trivial case:
        return rows[:, np.newaxis], columns[:, np.newaxis]

    sample_height, sample_width = sample_size
    row_grids = np.empty((0, sample_width), dtype=int)
    column_grids = np.empty((0, sample_width), dtype=int)

    for row, column in zip(rows, columns):
        row_grid, column_grid = np.meshgrid(
            np.arange(row, row + sample_height), np.arange(column, column + sample_width)
        )
        row_grids = np.concatenate((row_grids, np.transpose(row_grid)), axis=0)
        column_grids = np.concatenate((column_grids, np.transpose(column_grid)), axis=0)

    return row_grids, column_grids


def get_mask_of_samples(image_shape: Tuple[int, int], row_grid: np.ndarray, column_grid: np.ndarray) -> np.ndarray:
    """Creates a mask of counts how many times each pixel has been sampled.

    :param image_shape: Height and width of a sampled image.
    :param row_grid: A 2-dimensional numpy array of row indices of all sampled points.
    :param column_grid: A 2-dimensional numpy array of column indices of all sampled points.
    :return: An image mask where each pixel is assigned a count of how many times it was sampled.
    """
    mask = np.zeros(image_shape, dtype=np.uint16)

    sampled_pixels = np.stack([row_grid.flatten(), column_grid.flatten()])
    sampled_points, sampled_counts = np.unique(sampled_pixels, return_counts=True, axis=1)

    rows, columns = sampled_points
    mask[rows, columns] = sampled_counts

    if np.max(mask) < 256:
        mask = mask.astype(np.uint8)
    return mask


class BaseSamplingTask(EOTask, metaclass=ABCMeta):  # noqa: B024
    """A base class for sampling tasks"""

    def __init__(
        self,
        features_to_sample: FeaturesSpecification,
        *,
        mask_of_samples: Optional[Tuple[FeatureType, str]] = None,
    ):
        """
        :param features_to_sample: Features that will be spatially sampled according to given sampling parameters.
        :param mask_of_samples: An output mask timeless feature of counts how many times each pixel has been sampled.
        """
        self.features_parser = self.get_feature_parser(
            features_to_sample,
            allowed_feature_types=FeatureTypeSet.SPATIAL_TYPES & FeatureTypeSet.RASTER_TYPES,
        )

        self.mask_of_samples = mask_of_samples
        if mask_of_samples is not None:
            self.mask_of_samples = self.parse_feature(  # type: ignore[assignment]
                self.mask_of_samples, allowed_feature_types={FeatureType.MASK_TIMELESS}
            )

    def _apply_sampling(self, eopatch: EOPatch, row_grid: np.ndarray, column_grid: np.ndarray) -> EOPatch:
        """Applies masks of sampled indices to EOPatch features to create sampled features and a mask of samples"""
        image_shape = None
        for feature_type, feature_name, new_feature_name in self.features_parser.get_renamed_features(eopatch):
            if feature_name is not None:
                data_to_sample = eopatch[feature_type][feature_name]

                feature_shape = eopatch.get_spatial_dimension(feature_type, feature_name)
                image_shape = feature_shape

                eopatch[feature_type][new_feature_name] = data_to_sample[..., row_grid, column_grid, :]

        if self.mask_of_samples is not None and image_shape is not None:
            mask = get_mask_of_samples(image_shape, row_grid, column_grid)
            eopatch[self.mask_of_samples] = mask[..., np.newaxis]

        return eopatch


class FractionSamplingTask(BaseSamplingTask):
    """The main task for pixel-based sampling that samples a fraction of viable points determined by a mask feature.

    The task aims to preserve the value distribution of the mask feature in the samples. Values can be excluded and the
    process can also be fine-tuned by passing a dictionary of fractions for each value of the mask feature.
    """

    def __init__(
        self,
        features_to_sample: FeaturesSpecification,
        sampling_feature: SingleFeatureSpec,
        fraction: _FractionType,
        exclude_values: Optional[List[int]] = None,
        replace: bool = False,
        **kwargs: Any,
    ):
        """
        :param features_to_sample: Features that will be spatially sampled according to given sampling parameters.
        :param sampling_feature: A timeless mask feature according to which points will be sampled.
        :param fraction: Fraction of points to sample. Can be dictionary mapping values of mask to fractions.
        :param exclude_values: Skips points that have these values in `sampling_mask`
        :param replace: Whether to sample with replacement. False means each value can only be chosen once.
        :param kwargs: Arguments for :class:`BaseSamplingTask<eolearn.geometry.sampling.BaseSamplingTask>`
        """
        super().__init__(features_to_sample, **kwargs)

        self.sampling_feature = self.parse_feature(sampling_feature, allowed_feature_types={FeatureType.MASK_TIMELESS})

        self.fraction = fraction
        self.exclude_values = exclude_values or []
        self.replace = replace

        self._validate_fraction_input(fraction)

    def _validate_fraction_input(self, fraction: _FractionType) -> None:
        """Validates that the input for `fraction` is correct

        The input should either be a number or a dictionary linking labels to numbers. Number representing fractions
        are checked to be in [0, 1] or (if replacement is used) in [0, inf).
        """
        if isinstance(fraction, (int, float)):
            if fraction < 0:
                raise ValueError(f"Sampling fractions have to be positive, but {fraction} was given.")
            if not self.replace and not 0 <= fraction <= 1:
                raise ValueError(f"Each sampling fraction has to be between 0 and 1, but {fraction} was given.")
        elif isinstance(fraction, dict):
            for class_fraction in fraction.values():
                self._validate_fraction_input(class_fraction)
        else:
            raise ValueError(
                f"The fraction input is {fraction} but needs to be a number or a dictionary mapping labels to numbers."
            )

    def _calculate_amount_per_value(self, image: np.ndarray, fraction: _FractionType) -> Dict[int, int]:
        """Calculates the number of samples needed for each value present in mask according to the fraction parameter"""
        uniques, counts = np.unique(image, return_counts=True)
        available = {val: n for val, n in zip(uniques, counts) if val not in self.exclude_values}

        if isinstance(fraction, dict):
            return {val: round(n * fraction[val]) for val, n in available.items() if val in fraction}
        return {val: round(n * self.fraction) for val, n in available.items()}

    def execute(
        self, eopatch: EOPatch, *, seed: Optional[int] = None, fraction: Optional[_FractionType] = None
    ) -> EOPatch:
        """Execute random spatial sampling of specified features of eopatch

        :param eopatch: Input eopatch to be sampled
        :param seed: Setting seed of random sampling. If None a random seed will be used.
        :param fraction: Override the sampling fraction of the task. If None the value from task initialization will
            be used.
        :return: An EOPatch with additional spatially sampled features
        """
        rng = np.random.default_rng(seed)
        sampling_image = eopatch[self.sampling_feature].squeeze(axis=-1)

        if fraction is not None:
            self._validate_fraction_input(fraction)
        fraction = fraction or self.fraction
        amount_per_value = self._calculate_amount_per_value(sampling_image, fraction)

        rows, columns = sample_by_values(sampling_image, amount_per_value, rng=rng, replace=self.replace)
        row_grid, column_grid = expand_to_grids(rows, columns, sample_size=(1, 1))

        eopatch = self._apply_sampling(eopatch, row_grid, column_grid)
        return eopatch


class BlockSamplingTask(BaseSamplingTask):
    """A task to randomly sample pixels or blocks of any size.

    The task has no option to add data validity masks, because when sampling a fixed amount of objects it can cause
    uneven distribution density across different eopatches. For any purposes that require fine-tuning use
    `FractionSamplingTask` instead.
    """

    def __init__(
        self,
        features_to_sample: FeaturesSpecification,
        amount: float,
        sample_size: Tuple[int, int] = (1, 1),
        replace: bool = False,
        **kwargs: Any,
    ):
        """
        :param features_to_sample: Features that will be spatially sampled according to given sampling parameters.
        :param amount: The number of points to sample if integer valued and the fraction of all points if `float`
        :param sample_size: A tuple describing a size of sampled blocks. The size is defined as a tuple of number of
            rows and number of columns.
        :param replace: Whether to sample with replacement. False means each value can only be chosen once.
        :param kwargs: Arguments for :class:`BaseSamplingTask<eolearn.geometry.sampling.BaseSamplingTask>`
        """
        super().__init__(features_to_sample, **kwargs)

        self.amount = amount
        if not (
            isinstance(sample_size, tuple)
            and len(sample_size) == 2
            and all(isinstance(value, int) for value in sample_size)
        ):
            raise ValueError(f"Parameter sample_size should be a tuple of 2 integers but {sample_size} found")

        self.sample_size = tuple(sample_size)
        self.replace = replace

    def _generate_dummy_mask(self, eopatch: EOPatch) -> np.ndarray:
        """Generate a mask consisting entirely of `values` entries, used for sampling on whole raster"""

        feature_type, feature_name = self.features_parser.get_features(eopatch)[0]
        if feature_name is None:
            raise ValueError(
                f"Encountered {feature_type} when calculating spatial dimension, please report bug to eo-learn"
                " developers."
            )

        height, width = eopatch.get_spatial_dimension(feature_type, feature_name)
        height -= self.sample_size[0] - 1
        width -= self.sample_size[1] - 1

        return np.ones((height, width), dtype=np.uint8)

    def execute(self, eopatch: EOPatch, *, seed: Optional[int] = None, amount: Optional[float] = None) -> EOPatch:
        """Execute a spatial sampling on features from a given EOPatch

        :param eopatch: Input eopatch to be sampled
        :param seed: Setting seed of random sampling. If None a random seed will be used.
        :param amount: A number of points to sample if integer valued and a fraction of all points if `float`. If `None`
            the value from task initialization will be used.
        :return: An EOPatch with additional spatially sampled features
        """
        rng = np.random.default_rng(seed)

        sampling_image = self._generate_dummy_mask(eopatch)

        amount = self.amount if amount is None else amount
        if isinstance(amount, (int, np.integer)):  # type: ignore[unreachable]
            n_samples_per_value = {1: amount}  # type: ignore[unreachable]
        else:
            n_samples_per_value = {1: round(sampling_image.size * amount)}

        rows, columns = sample_by_values(sampling_image, n_samples_per_value, rng=rng, replace=self.replace)
        size_x, size_y = self.sample_size  # this way it also works for lists
        row_grid, column_grid = expand_to_grids(rows, columns, sample_size=(size_x, size_y))

        eopatch = self._apply_sampling(eopatch, row_grid, column_grid)
        return eopatch


class GridSamplingTask(BaseSamplingTask):
    """A task to sample blocks of a given size in a regular grid.

    This task doesn't use any randomness and always produces the same results.
    """

    def __init__(
        self,
        features_to_sample: FeaturesSpecification,
        sample_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        **kwargs: Any,
    ):
        """
        :param features_to_sample: Features that will be spatially sampled according to given sampling parameters.
        :param sample_size: A tuple describing a size of sampled blocks. The size is defined as a tuple of number of
            rows and number of columns.
        :param stride: A tuple describing a distance between upper left corners of two consecutive sampled blocks.
            The first number is the vertical distance and the second number the horizontal distance. If stride in
            smaller than sample_size in any dimensions then sampled blocks will overlap.
        :param kwargs: Arguments for :class:`BaseSamplingTask<eolearn.geometry.sampling.BaseSamplingTask>`
        """
        super().__init__(features_to_sample, **kwargs)

        self.sample_size = tuple(sample_size)
        self.stride = tuple(stride)

        if not all(value > 0 for value in self.sample_size + self.stride):
            raise ValueError("Both sample_size and stride should have only positive values")

    def _sample_regular_grid(self, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Samples points from a regular grid and returns indices of rows and columns"""
        rows = np.arange(0, image_shape[0] - self.sample_size[0] + 1, self.stride[0])
        columns = np.arange(0, image_shape[1] - self.sample_size[1] + 1, self.stride[1])

        rows, columns = np.meshgrid(rows, columns)
        return np.transpose(rows).flatten(), np.transpose(columns).flatten()

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute a spatial sampling on features from a given EOPatch

        :param eopatch: Input eopatch to be sampled
        :return: An EOPatch with additional spatially sampled features
        """
        feature_type, feature_name = self.features_parser.get_features(eopatch)[0]
        if feature_name is None:
            raise ValueError(
                f"Encountered {feature_type} when calculating spatial dimension, please report bug to eo-learn"
                " developers."
            )

        image_shape = eopatch.get_spatial_dimension(feature_type, feature_name)
        rows, columns = self._sample_regular_grid(image_shape)
        size_x, size_y = self.sample_size  # this way it also works for lists
        row_grid, column_grid = expand_to_grids(rows, columns, sample_size=(size_x, size_y))

        eopatch = self._apply_sampling(eopatch, row_grid, column_grid)
        return eopatch
