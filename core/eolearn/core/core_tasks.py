"""
A collection of most basic EOTasks

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import copy
from abc import ABCMeta
from typing import Any, Callable, Iterable, Tuple, Union, cast

import fs
import numpy as np
from fs.base import FS

from sentinelhub import SHConfig

from .constants import FeatureType, OverwritePermission
from .eodata import EOPatch
from .eodata_merge import merge_eopatches
from .eotask import EOTask
from .types import FeatureSpec, FeaturesSpecification, SingleFeatureSpec
from .utils.fs import get_filesystem, pickle_fs, unpickle_fs


class CopyTask(EOTask):
    """Makes a shallow copy of the given EOPatch.

    It copies feature type dictionaries but not the data itself.
    """

    def __init__(self, features: FeaturesSpecification = ...):
        """
        :param features: A collection of features or feature types that will be copied into a new EOPatch.
        """
        self.features = features

    def execute(self, eopatch: EOPatch) -> EOPatch:
        return eopatch.copy(features=self.features)


class DeepCopyTask(CopyTask):
    """Makes a deep copy of the given EOPatch."""

    def execute(self, eopatch: EOPatch) -> EOPatch:
        return eopatch.copy(features=self.features, deep=True)


class IOTask(EOTask, metaclass=ABCMeta):
    """An abstract Input/Output task that can handle a path and a filesystem object."""

    def __init__(self, path: str, filesystem: FS | None = None, create: bool = False, config: SHConfig | None = None):
        """
        :param path: root path where all EOPatches are saved
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the EOPatch
            path.
        :param create: If the filesystem path doesn't exist this flag indicates to either create it or raise an error
        :param config: A configuration object with AWS credentials. By default, is set to None and in this case the
            default configuration will be taken.
        """
        self.path = path
        self.filesystem_path = "/" if filesystem is None else self.path

        self._pickled_filesystem = None if filesystem is None else pickle_fs(filesystem)
        self._create_path = create
        self.config = config

    @property
    def filesystem(self) -> FS:
        """A filesystem property that unpickles an existing filesystem definition or creates a new one."""
        if self._pickled_filesystem is None:
            filesystem = get_filesystem(self.path, create=self._create_path, config=self.config)
            self._pickled_filesystem = pickle_fs(filesystem)
            return filesystem

        return unpickle_fs(self._pickled_filesystem)


class SaveTask(IOTask):
    """Saves the given EOPatch to a filesystem."""

    def __init__(
        self,
        path: str,
        filesystem: FS | None = None,
        config: SHConfig | None = None,
        features: FeaturesSpecification = ...,
        overwrite_permission: OverwritePermission = OverwritePermission.ADD_ONLY,
        compress_level: int = 0,
        use_zarr: bool = False,
    ):
        """
        :param path: root path where all EOPatches are saved
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the path.
        :param config: A configuration object with AWS credentials. By default, is set to None and in this case the
            default configuration will be taken.
        :param features: A collection of features types specifying features of which type will be saved. By default,
            all features will be saved.
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :param compress_level: A level of data compression and can be specified with an integer from 0 (no compression)
            to 9 (highest compression).
        :param use_zarr: Saves numpy-array based features into Zarr files. Requires ZARR extra dependencies.
        """
        self.features = features
        self.overwrite_permission = overwrite_permission
        self.compress_level = compress_level
        self.use_zarr = use_zarr
        super().__init__(path, filesystem=filesystem, create=True, config=config)

    def execute(self, eopatch: EOPatch, *, eopatch_folder: str = "") -> EOPatch:
        """Saves the EOPatch to disk: `folder/eopatch_folder`.

        :param eopatch: EOPatch which will be saved
        :param eopatch_folder: Name of EOPatch folder containing data.
        :return: The same EOPatch
        """
        path = fs.path.combine(self.filesystem_path, eopatch_folder)
        eopatch.save(
            path,
            filesystem=self.filesystem,
            features=self.features,
            overwrite_permission=self.overwrite_permission,
            compress_level=self.compress_level,
            use_zarr=self.use_zarr,
        )
        return eopatch


class LoadTask(IOTask):
    """Loads an EOPatch from a filesystem."""

    def __init__(
        self,
        path: str,
        filesystem: FS | None = None,
        config: SHConfig | None = None,
        features: FeaturesSpecification = ...,
        lazy_loading: bool = False,
    ):
        """
        :param path: root directory where all EOPatches are saved
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the EOPatch
            path. If you intend to run this task in multiprocessing mode you shouldn't specify this parameter.
        :param config: A configuration object with AWS credentials. By default, is set to None and in this case the
            default configuration will be taken.
        :param features: A collection of features to be loaded. By default, all features will be loaded.
        :param lazy_loading: If `True` features will be lazy loaded.
        """
        self.features = features
        self.lazy_loading = lazy_loading
        super().__init__(path, filesystem=filesystem, create=False, config=config)

    def execute(self, eopatch: EOPatch | None = None, *, eopatch_folder: str = "") -> EOPatch:
        """Loads the EOPatch from disk: `folder/eopatch_folder`.

        :param eopatch: Optional input EOPatch. If given the loaded features are merged onto it, otherwise a new EOPatch
            is created.
        :param eopatch_folder: Name of EOPatch folder containing data.
        :return: EOPatch loaded from disk
        """
        path = fs.path.combine(self.filesystem_path, eopatch_folder)
        loaded_patch = EOPatch.load(
            path, filesystem=self.filesystem, features=self.features, lazy_loading=self.lazy_loading
        )
        return loaded_patch if eopatch is None else merge_eopatches(eopatch, loaded_patch)


class AddFeatureTask(EOTask):
    """Adds a feature to the given EOPatch."""

    def __init__(self, feature: FeatureSpec):
        """
        :param feature: Feature to be added
        """
        self.feature_type, self.feature_name = self.parse_feature(feature)

    def execute(self, eopatch: EOPatch, data: object) -> EOPatch:
        """Returns the EOPatch with added features.

        :param eopatch: input EOPatch
        :param data: data to be added to the feature
        :return: input EOPatch with the specified feature
        """
        if self.feature_name is None:
            eopatch[self.feature_type] = data
        else:
            eopatch[self.feature_type][self.feature_name] = data

        return eopatch


class RemoveFeatureTask(EOTask):
    """Removes one or multiple features from the given EOPatch."""

    def __init__(self, features: FeaturesSpecification):
        """
        :param features: A collection of features to be removed.
        """
        self.feature_parser = self.get_feature_parser(features)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Returns the EOPatch with removed features.

        :param eopatch: input EOPatch
        :return: input EOPatch without the specified feature
        """
        for feature in self.feature_parser.get_features(eopatch):
            del eopatch[feature]

        return eopatch


class RenameFeatureTask(EOTask):
    """Renames one or multiple features from the given EOPatch."""

    def __init__(self, features: FeaturesSpecification):
        """
        :param features: A collection of features to be renamed.
        """
        self.feature_parser = self.get_feature_parser(features)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Returns the EOPatch with renamed features.

        :param eopatch: input EOPatch
        :return: input EOPatch with the renamed features
        """
        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            eopatch[feature_type][new_feature_name] = eopatch[feature_type][feature_name]
            del eopatch[feature_type][feature_name]

        return eopatch


class DuplicateFeatureTask(EOTask):
    """Duplicates one or multiple features in an EOPatch."""

    def __init__(self, features: FeaturesSpecification, deep_copy: bool = False):
        """
        :param features: A collection of features to be copied.
        :param deep_copy: Make a deep copy of feature's data if set to true, else just assign it.
        """
        self.feature_parser = self.get_feature_parser(features)
        self.deep = deep_copy

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Returns the EOPatch with copied features.

        :param eopatch: Input EOPatch
        :return: Input EOPatch with the duplicated features.
        :raises ValueError: Raises an exception when trying to duplicate a feature with an
            already existing feature name.
        """

        for feature_type, feature_name, new_feature_name in self.feature_parser.get_renamed_features(eopatch):
            if new_feature_name in eopatch[feature_type]:
                raise ValueError(f"A feature named '{new_feature_name}' already exists.")

            if self.deep:
                eopatch[feature_type][new_feature_name] = copy.deepcopy(eopatch[feature_type][feature_name])
            else:
                eopatch[feature_type][new_feature_name] = eopatch[feature_type][feature_name]

        return eopatch


class InitializeFeatureTask(EOTask):
    """Initializes the values of a feature.

    Example:

    .. code-block:: python

        InitializeFeature((FeatureType.DATA, 'data1'), shape=(5, 10, 10, 3), init_value=3)

        # Initialize data of the same shape as (FeatureType.DATA, 'data1')
        InitializeFeature((FeatureType.MASK, 'mask1'), shape=(FeatureType.DATA, 'data1'), init_value=1)
    """

    def __init__(
        self,
        features: FeaturesSpecification,
        shape: tuple[int, ...] | FeatureSpec,
        init_value: int = 0,
        dtype: np.dtype | type = np.uint8,
    ):
        """
        :param features: A collection of features to initialize.
        :param shape: A shape object (t, n, m, d) or a feature from which to read the shape.
        :param init_value: A value with which to initialize the array of the new feature.
        :param dtype: Type of array values.
        :raises ValueError: Raises an exception when passing the wrong shape argument.
        """

        self.features = self.parse_features(features)
        self.shape_feature: tuple[FeatureType, str | None] | None
        self.shape: None | tuple[int, int, int] | tuple[int, int, int, int]

        try:
            self.shape_feature = self.parse_feature(shape)  # type: ignore[arg-type]
        except ValueError:
            self.shape_feature = None

        if self.shape_feature:
            self.shape = None
        elif isinstance(shape, tuple) and len(shape) in (3, 4) and all(isinstance(x, int) for x in shape):
            self.shape = cast(Union[Tuple[int, int, int], Tuple[int, int, int, int]], shape)
        else:
            raise ValueError("shape argument is not a shape tuple or a feature containing one.")

        self.init_value = init_value
        self.dtype = dtype

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: Input EOPatch.
        :return: Input EOPatch with the initialized additional features.
        """
        shape = eopatch[self.shape_feature].shape if self.shape_feature else self.shape

        add_features = set(self.features) - set(self.parse_features(eopatch.get_features()))

        for feature in add_features:
            eopatch[feature] = np.ones(shape, dtype=self.dtype) * self.init_value

        return eopatch


class MoveFeatureTask(EOTask):
    """Task to copy/deepcopy fields from one EOPatch to another."""

    def __init__(self, features: FeaturesSpecification, deep_copy: bool = False):
        """
        :param features: A collection of features to be moved.
        :param deep_copy: Make a deep copy of feature's data if set to true, else just assign it.
        """
        self.feature_parser = self.get_feature_parser(features)
        self.deep = deep_copy

    def execute(self, src_eopatch: EOPatch, dst_eopatch: EOPatch) -> EOPatch:
        """
        :param src_eopatch: Source EOPatch from which to take features.
        :param dst_eopatch: Destination EOPatch to which to move/copy features.
        :return: dst_eopatch with the additional features from src_eopatch.
        """

        for feature in self.feature_parser.get_features(src_eopatch):
            if self.deep:
                dst_eopatch[feature] = copy.deepcopy(src_eopatch[feature])
            else:
                dst_eopatch[feature] = src_eopatch[feature]

        return dst_eopatch


class MapFeatureTask(EOTask):
    """Applies a function to each feature in input_features of a patch and stores the results in a set of
    output_features.

    Example using inheritance:

    .. code-block:: python

        class MultiplyFeatures(MapFeatureTask):
            def map_method(self, f):
                return f * 2

        multiply = MultiplyFeatures({FeatureType.DATA: ['f1', 'f2', 'f3']},  # input features
                                    {FeatureType.MASK: ['m1', 'm2', 'm3']})  # output features

        result = multiply(patch)

    Example using lambda:

    .. code-block:: python

        multiply = MapFeatureTask({FeatureType.DATA: ['f1', 'f2', 'f3']},  # input features
                                  {FeatureType.MASK: ['m1', 'm2', 'm3']},  # output features
                                  lambda f: f*2)                           # a function to apply to each feature

        result = multiply(patch)

    Example using a `np.max` and it's kwargs passed as arguments to the `MapFeatureTask`:

    .. code-block:: python

        maximum = MapFeatureTask((FeatureType.DATA: 'f1'),  # input features
                                 (FeatureType.MASK, 'm1'),  # output feature
                                 np.max,                    # a function to apply to each feature
                                 axis=0)                    # function's kwargs

        result = maximum(patch)
    """

    def __init__(
        self,
        input_features: FeaturesSpecification,
        output_features: FeaturesSpecification,
        map_function: Callable | None = None,
        **kwargs: Any,
    ):
        """
        :param input_features: A collection of the input features to be mapped.
        :param output_features: A collection of the output features to which to assign the output data.
        :param map_function: A function or lambda to be applied to the input data.
        :raises ValueError: Raises an exception when passing feature collections with different lengths.
        :param kwargs: kwargs to be passed to the map function.
        """
        self.input_features = self.parse_features(input_features)
        self.output_features = self.parse_features(output_features)
        self.kwargs = kwargs

        if len(self.input_features) != len(self.output_features):
            raise ValueError("The number of input and output features must match.")

        if map_function:  # mypy 0.981 has issues with inlined conditional and functions
            self.function: Callable = map_function
        else:
            self.function = self.map_method

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: Source EOPatch from which to read the data of input features.
        :return: An eopatch with the additional mapped features.
        """
        for input_feature, output_feature in zip(self.input_features, self.output_features):
            eopatch[output_feature] = self.function(eopatch[input_feature], **self.kwargs)

        return eopatch

    def map_method(self, feature: Any) -> Any:
        """A function that will be applied to the input features."""
        raise NotImplementedError("map_method should be overridden.")


class ZipFeatureTask(EOTask):
    """Passes a set of input_features to a function, which returns a single features as a result and stores it in
    the given EOPatch.

        Example using inheritance:

        .. code-block:: python

            class CalculateFeatures(ZipFeatureTask):
                def zip_method(self, *f):
                    return f[0] / (f[1] + f[2])

            calc = CalculateFeatures({FeatureType.DATA: ['f1', 'f2', 'f3']}, # input features
                                     (FeatureType.MASK, 'm1'))               # output feature

            result = calc(patch)

        Example using lambda:

        .. code-block:: python

            calc = ZipFeatureTask({FeatureType.DATA: ['f1', 'f2', 'f3']},  # input features
                                  (FeatureType.MASK, 'm1'),                # output feature
                                  lambda f0, f1, f2: f0 / (f1 + f2))       # a function to apply to each feature

            result = calc(patch)

        Example using a `np.maximum` and it's kwargs passed as arguments to the `ZipFeatureTask`:

        .. code-block:: python

            maximum = ZipFeatureTask({FeatureType.DATA: ['f1', 'f2']},  # input features
                                     (FeatureType.MASK, 'm1'),          # output feature
                                     np.maximum,                        # a function to apply to each feature
                                     dtype=np.float64)                  # function's kwargs

            result = maximum(patch)
    """

    def __init__(
        self,
        input_features: FeaturesSpecification,
        output_feature: SingleFeatureSpec,
        zip_function: Callable | None = None,
        **kwargs: Any,
    ):
        """
        :param input_features: A collection of the input features to be mapped.
        :param output_feature: An output feature object to which to assign the data.
        :param zip_function: A function or lambda to be applied to the input data.
        :param kwargs: kwargs to be passed to the zip function.
        """
        self.input_features = self.parse_features(input_features)
        self.output_feature = self.parse_feature(output_feature)
        self.kwargs = kwargs

        if zip_function:  # mypy 0.981 has issues with inlined conditional and functions
            self.function: Callable = zip_function
        else:
            self.function = self.zip_method

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: Source EOPatch from which to read the data of input features.
        :return: An eopatch with the additional zipped features.
        """
        data = [eopatch[feature] for feature in self.input_features]

        eopatch[self.output_feature] = self.function(*data, **self.kwargs)

        return eopatch

    def zip_method(self, *features: Any) -> Any:
        """A function that will be applied to the input features if overridden."""
        raise NotImplementedError("zip_method should be overridden.")


class MergeFeatureTask(ZipFeatureTask):
    """Merges multiple features together by concatenating their data along the specified axis."""

    def zip_method(self, *f: np.ndarray, dtype: None | np.dtype | type = None, axis: int = -1) -> np.ndarray:
        """Concatenates the data of features along the specified axis."""
        return np.concatenate(f, axis=axis, dtype=dtype)  # pylint: disable=unexpected-keyword-arg


class ExtractBandsTask(MapFeatureTask):
    """Moves a subset of bands from one feature to a new one."""

    def __init__(self, input_feature: FeaturesSpecification, output_feature: FeaturesSpecification, bands: list[int]):
        """
        :param input_feature: A source feature from which to take the subset of bands.
        :param output_feature: An output feature to which to write the bands.
        :param bands: A list of bands to be moved.
        """
        super().__init__(input_feature, output_feature)
        self.bands = bands

    def map_method(self, feature: np.ndarray) -> np.ndarray:
        if not all(band < feature.shape[-1] for band in self.bands):
            raise ValueError("Band index out of feature's dimensions.")

        return feature[..., self.bands]


class ExplodeBandsTask(EOTask):
    """Explode a subset of bands from one feature to multiple new features."""

    def __init__(
        self,
        input_feature: tuple[FeatureType, str],
        output_mapping: dict[tuple[FeatureType, str], int | Iterable[int]],
    ):
        """
        :param input_feature: A source feature from which to take the subset of bands.
        :param output_mapping: A mapping of output features into the band indices used to explode the input feature.
        """
        self.input_feature = input_feature
        self.output_mapping = output_mapping

    def execute(self, eopatch: EOPatch) -> EOPatch:
        for output_feature, bands in self.output_mapping.items():
            new_bands = list(bands) if isinstance(bands, Iterable) else [bands]
            eopatch = ExtractBandsTask(
                input_feature=self.input_feature, output_feature=output_feature, bands=new_bands
            ).execute(eopatch)
        return eopatch


class CreateEOPatchTask(EOTask):
    """Creates an EOPatch."""

    def execute(self, **kwargs: Any) -> EOPatch:
        """Returns a newly created EOPatch with the given kwargs.

        :param kwargs: Any valid kwargs accepted by :class:`EOPatch.__init__<eolearn.core.eodata.EOPatch>`
        :return: A new eopatch.
        """
        return EOPatch(**kwargs)


class MergeEOPatchesTask(EOTask):
    """Merge content from multiple EOPatches into a single EOPatch.

    Check :func:`merge_eopatches<eolearn.core.eodata_merge.merge_eopatches>` for more information.
    """

    def __init__(self, **merge_kwargs: Any):
        """
        :param merge_kwargs: Keyword arguments defined for `merge_eopatches` function.
        """
        self.merge_kwargs = merge_kwargs

    def execute(self, *eopatches: EOPatch) -> EOPatch:
        """
        :param eopatches: EOPatches to be merged
        :return: A new EOPatch with merged content
        """
        if not eopatches:
            raise ValueError("At least one EOPatch should be given")

        return merge_eopatches(*eopatches, **self.merge_kwargs)
