"""
A collection of most basic EOTasks

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import copy
import warnings
from abc import abstractmethod

import fs
import numpy as np

from .eodata import EOPatch
from .eotask import EOTask
from .fs_utils import get_filesystem

warnings.simplefilter('default', DeprecationWarning)


class CopyTask(EOTask):
    """Makes a shallow copy of the given EOPatch.

    It copies feature type dictionaries but not the data itself.

    """
    def __init__(self, features=...):
        """
        :param features: A collection of features or feature types that will be copied into a new EOPatch.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        self.features = features

    def execute(self, eopatch):
        return eopatch.__copy__(features=self.features)


class DeepCopyTask(CopyTask):
    """ Makes a deep copy of the given EOPatch.
    """
    def execute(self, eopatch):
        return eopatch.__deepcopy__(features=self.features)


class IOTask(EOTask):
    """ An abstract Input/Output task that can handle a path and a filesystem object
    """
    def __init__(self, path, filesystem=None, create=False):
        """
        :param path: root path where all EOPatches are saved
        :type path: str
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the EOPatch
            path. If you intend to run this task in multiprocessing mode you shouldn't specify this parameter.
        :type filesystem: fs.base.FS or None
        :param create: If the filesystem path doesn't exist this flag indicates to either create it or raise an error
        :type create: bool
        """
        self.path = path
        self._filesystem = filesystem
        self._create = create

        self.filesystem_path = '/' if self._filesystem is None else self.path

    @property
    def filesystem(self):
        """ A filesystem property that is being lazy-loaded the first time it is needed
        """
        if self._filesystem is None:
            self._filesystem = get_filesystem(self.path, create=self._create)

        return self._filesystem

    @abstractmethod
    def execute(self, *eopatches, **kwargs):
        """ Implement execute function
        """
        raise NotImplementedError


class SaveTask(IOTask):
    """ Saves the given EOPatch to a filesystem
    """
    def __init__(self, path, filesystem=None, **kwargs):
        """
        :param path: root path where all EOPatches are saved
        :type path: str
        :param features: A collection of features types specifying features of which type will be saved. By default
            all features will be saved.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified with an integer from 0 (no compression)
            to 9 (highest compression).
        :type compress_level: int
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the EOPatch
            path. If you intend to run this task in multiprocessing mode you shouldn't specify this parameter.
        :type filesystem: fs.base.FS or None
        """
        self.kwargs = kwargs
        super().__init__(path, filesystem=filesystem, create=True)

    def execute(self, eopatch, *, eopatch_folder=''):
        """Saves the EOPatch to disk: `folder/eopatch_folder`.

        :param eopatch: EOPatch which will be saved
        :type eopatch: EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        :return: The same EOPatch
        :rtype: EOPatch
        """
        path = fs.path.combine(self.filesystem_path, eopatch_folder)

        eopatch.save(path, filesystem=self.filesystem, **self.kwargs)
        return eopatch


class SaveToDisk(SaveTask):
    """ A deprecated version of SaveTask
    """
    def __init__(self, folder, *args, **kwargs):
        warnings.warn('This task is deprecated, use SaveTask instead', DeprecationWarning)
        super().__init__(folder, *args, **kwargs)


class LoadTask(IOTask):
    """ Loads an EOPatch from a filesystem
    """
    def __init__(self, path, filesystem=None, **kwargs):
        """
        :param folder: root directory where all EOPatches are saved
        :type folder: str
        :param features: A collection of features to be loaded. By default all features will be loaded.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param lazy_loading: If `True` features will be lazy loaded. Default is `False`
        :type lazy_loading: bool
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the EOPatch
            path. If you intend to run this task in multiprocessing mode you shouldn't specify this parameter.
        :type filesystem: fs.base.FS or None
        """
        self.kwargs = kwargs
        super().__init__(path, filesystem=filesystem, create=False)

    def execute(self, *, eopatch_folder=''):
        """Loads the EOPatch from disk: `folder/eopatch_folder`.

        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        :return: EOPatch loaded from disk
        :rtype: EOPatch
        """
        path = fs.path.combine(self.filesystem_path, eopatch_folder)

        return EOPatch.load(path, filesystem=self.filesystem, **self.kwargs)


class LoadFromDisk(LoadTask):
    """ A deprecated version of LoadTask
    """
    def __init__(self, folder, *args, **kwargs):
        warnings.warn('This task is deprecated, use LoadTask instead', DeprecationWarning)
        super().__init__(folder, *args, **kwargs)


class AddFeature(EOTask):
    """Adds a feature to the given EOPatch.
    """
    def __init__(self, feature):
        """
        :param feature: Feature to be added
        :type feature: (FeatureType, feature_name) or FeatureType
        """
        self.feature_type, self.feature_name = next(self._parse_features(feature)())

    def execute(self, eopatch, data):
        """Returns the EOPatch with added features.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param data: data to be added to the feature
        :type data: object
        :return: input EOPatch with the specified feature
        :rtype: EOPatch
        """
        if self.feature_name is None:
            eopatch[self.feature_type] = data
        else:
            eopatch[self.feature_type][self.feature_name] = data

        return eopatch


class RemoveFeature(EOTask):
    """Removes one or multiple features from the given EOPatch.
    """
    def __init__(self, features):
        """
        :param features: A collection of features to be removed.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        self.feature_gen = self._parse_features(features)

    def execute(self, eopatch):
        """Returns the EOPatch with removed features.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: input EOPatch without the specified feature
        :rtype: EOPatch
        """
        for feature_type, feature_name in list(self.feature_gen(eopatch)):
            if feature_name is ...:
                eopatch.reset_feature_type(feature_type)
            else:
                del eopatch[feature_type][feature_name]

        return eopatch


class RenameFeature(EOTask):
    """Renames one or multiple features from the given EOPatch.
    """
    def __init__(self, features):
        """
        :param features: A collection of features to be renamed.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        self.feature_gen = self._parse_features(features, new_names=True)

    def execute(self, eopatch):
        """Returns the EOPatch with renamed features.

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: input EOPatch with the renamed features
        :rtype: EOPatch
        """
        for feature_type, feature_name, new_feature_name in self.feature_gen(eopatch):
            eopatch[feature_type][new_feature_name] = eopatch[feature_type][feature_name]
            del eopatch[feature_type][feature_name]

        return eopatch


class DuplicateFeature(EOTask):
    """Duplicates one or multiple features in an EOPatch.
    """

    def __init__(self, features, deep_copy=False):
        """
        :param features: A collection of features to be copied.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param deep_copy: Make a deep copy of feature's data if set to true, else just assign it.
        :type deep_copy: bool
        """
        self.feature_gen = self._parse_features(features, new_names=True)
        self.deep = deep_copy

    def execute(self, eopatch):
        """Returns the EOPatch with copied features.

        :param eopatch: Input EOPatch
        :type eopatch: EOPatch
        :return: Input EOPatch with the duplicated features.
        :rtype: EOPatch
        :raises ValueError: Raises an exception when trying to duplicate a feature with an
            already existing feature name.
        """

        for feature_type, feature_name, new_feature_name in self.feature_gen(eopatch):
            if new_feature_name in eopatch[feature_type]:
                raise ValueError("A feature named '{}' already exists.".format(new_feature_name))

            if self.deep:
                eopatch[feature_type][new_feature_name] = copy.deepcopy(eopatch[feature_type][feature_name])
            else:
                eopatch[feature_type][new_feature_name] = eopatch[feature_type][feature_name]

        return eopatch


class InitializeFeature(EOTask):
    """ Initializes the values of a feature.

    Example:

    .. code-block:: python

        InitializeFeature((FeatureType.DATA, 'data1'), shape=(5, 10, 10, 3), init_value=3)

        # Initialize data of the same shape as (FeatureType.DATA, 'data1')
        InitializeFeature((FeatureType.MASK, 'mask1'), shape=(FeatureType.DATA, 'data1'), init_value=1)
    """
    def __init__(self, features, shape, init_value=0, dtype=np.uint8):
        """
        :param features: A collection of features to initialize.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param shape: A shape object (t, n, m, d) or a feature from which to read the shape.
        :type shape: A tuple or an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param init_value: A value with which to initialize the array of the new feature.
        :type init_value: int
        :param dtype: Type of array values.
        :type dtype: NumPy dtype
        :raises ValueError: Raises an exception when passing the wrong shape argument.
        """

        self.features = self._parse_features(features)

        try:
            self.shape_feature = next(self._parse_features(shape)())
        except ValueError:
            self.shape_feature = None

        if self.shape_feature:
            self.shape = None
        elif isinstance(shape, tuple) and len(shape) in (3, 4) and all(isinstance(x, int) for x in shape):
            self.shape = shape
        else:
            raise ValueError("shape argument is not a shape tuple or a feature containing one.")

        self.init_value = init_value
        self.dtype = dtype

    def execute(self, eopatch):
        """
        :param eopatch: Input EOPatch.
        :type eopatch: EOPatch
        :return: Input EOPatch with the initialized additional features.
        :rtype: EOPatch
        """
        shape = eopatch[self.shape_feature].shape if self.shape_feature else self.shape

        add_features = set(self.features) - set(eopatch.get_feature_list())

        for feature in add_features:
            eopatch[feature] = np.ones(shape, dtype=self.dtype) * self.init_value

        return eopatch


class MoveFeature(EOTask):
    """ Task to copy/deepcopy fields from one eopatch to another.
    """
    def __init__(self, features, deep_copy=False):
        """
        :param features: A collection of features to be moved.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param deep_copy: Make a deep copy of feature's data if set to true, else just assign it.
        :type deep_copy: bool
        """
        self.feature_gen = self._parse_features(features)
        self.deep = deep_copy

    def execute(self, src_eopatch, dst_eopatch):
        """
        :param src_eopatch: Source EOPatch from which to take features.
        :type src_eopatch: EOPatch
        :param dst_eopatch: Destination EOPatch to which to move/copy features.
        :type dst_eopatch: EOPatch
        :return: dst_eopatch with the additional features from src_eopatch.
        :rtype: EOPatch
        """

        for feature in self.feature_gen(src_eopatch):
            if self.deep:
                dst_eopatch[feature] = copy.deepcopy(src_eopatch[feature])
            else:
                dst_eopatch[feature] = src_eopatch[feature]

        return dst_eopatch


class MapFeatureTask(EOTask):
    """ Applies a function to each feature in input_features of a patch and stores the results in a set of
        output_features.

        Example using inheritance:

        .. code-block:: python

            class MultiplyFeatures(MapFeatureTask):
                def map_function(self, f):
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

        Example using a np.max and it's kwargs passed as arguments to the MapFeatureTask:

        .. code-block:: python

            maximum = MapFeatureTask((FeatureType.DATA: 'f1'),  # input features
                                     (FeatureType.MASK, 'm1'),  # output feature
                                     np.max,                    # a function to apply to each feature
                                     axis=0)                    # function's kwargs

            result = multiply(patch)
    """
    def __init__(self, input_features, output_features, map_function=None, **kwargs):
        """
        :param input_features: A collection of the input features to be mapped.
        :type input_features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_features: A collection of the output features to which to assign the output data.
        :type output_features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param map_function: A function or lambda to be applied to the input data.
        :raises ValueError: Raises an exception when passing feature collections with different lengths.
        :param kwargs: kwargs to be passed to the map function.
        """
        self.input_features = list(self._parse_features(input_features))
        self.output_feature = list(self._parse_features(output_features))
        self.kwargs = kwargs

        if len(self.input_features) != len(self.output_feature):
            raise ValueError('The number of input and output features must match.')

        self.function = map_function if map_function else self.map_method

    def execute(self, eopatch):
        """
        :param eopatch: Source EOPatch from which to read the data of input features.
        :type eopatch: EOPatch
        :return: An eopatch with the additional mapped features.
        :rtype: EOPatch
        """
        for input_features, output_feature in zip(self.input_features, self.output_feature):
            eopatch[output_feature] = self.function(eopatch[input_features], **self.kwargs)

        return eopatch

    def map_method(self, feature):
        """
        A function that will be applied to the input features.
        """
        raise NotImplementedError('map_method should be overridden.')


class ZipFeatureTask(EOTask):
    """ Passes a set of input_features to a function, which returns a single features as a result and stores it in
        the eopatch.

        Example using inheritance:

        .. code-block:: python

            class CalculateFeatures(ZipFeatureTask):
                def map_function(self, *f):
                    return f[0] / (f[1] + f[2])

            calc = CalculateFeatures({FeatureType.DATA: ['f1', 'f2', 'f3']}, # input features
                                     (FeatureType.MASK, 'm1'))               # output feature

            result = calc(patch)

        Example using lambda:

        .. code-block:: python

            calc = ZipFeatureTask({FeatureType.DATA: ['f1', 'f2', 'f3']},  # input features
                                  (FeatureType.MASK, 'm1'),                # output feature
                                  lambda f0, f1, f2: f0 / (f1 + f2))       # a function to apply to each feature

            result = multiply(patch)

        Example using a np.maximum and it's kwargs passed as arguments to the ZipFeatureTask:

        .. code-block:: python

            maximum = ZipFeatureTask({FeatureType.DATA: ['f1', 'f2']},  # input features
                                     (FeatureType.MASK, 'm1'),          # output feature
                                     np.maximum,                        # a function to apply to each feature
                                     dtype=np.float64)                  # function's kwargs

            result = multiply(patch)
    """
    def __init__(self, input_features, output_feature, zip_function=None, **kwargs):
        """
        :param input_features: A collection of the input features to be mapped.
        :type input_features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature object to which to assign the the data.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param zip_function: A function or lambda to be applied to the input data.
        :param kwargs: kwargs to be passed to the zip function.
        """
        self.input_features = list(self._parse_features(input_features))
        self.output_feature = next(self._parse_features(output_feature)())
        self.function = zip_function if zip_function else self.zip_method
        self.kwargs = kwargs

    def execute(self, eopatch):
        """
        :param eopatch: Source EOPatch from which to read the data of input features.
        :type eopatch: EOPatch
        :return: An eopatch with the additional zipped features.
        :rtype: EOPatch
        """
        data = [eopatch[feature] for feature in self.input_features]

        eopatch[self.output_feature] = self.function(*data, **self.kwargs)

        return eopatch

    def zip_method(self, *f):
        """A function that will be applied to the input features if overridden.

        :raises NotImplementedError: When called and was neither overridden nor function argument was provided in
        __init__.
        """
        raise NotImplementedError('zip_method should be overridden.')


class MergeFeatureTask(ZipFeatureTask):
    """ Merges multiple features together by concatenating their data along the last axis.
    """
    def zip_method(self, *f):
        """Concatenates the data of features along the last axis.
        """
        return np.concatenate(f, axis=-1)


class ExtractBandsTask(MapFeatureTask):
    """ Moves a subset of bands from one feature to a new one.
    """
    def __init__(self, input_feature, output_feature, bands):
        """
        :param input_feature: A source feature from which to take the subset of bands.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature to which to write the bands.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param bands: A list of bands to be moved.
        :type bands: list
        """
        super().__init__(input_feature, output_feature)
        self.bands = bands

    def map_method(self, feature):
        if not all(band < feature.shape[-1] for band in self.bands):
            raise ValueError("Band index out of feature's dimensions.")

        return feature[..., self.bands]


class CreateEOPatchTask(EOTask):
    """Creates an EOPatch
    """
    def execute(self, **kwargs):
        """Returns a newly created EOPatch with the given kwargs.

        :param kwargs: Any valid kwargs accepted by :class:`EOPatch.__init__<eolearn.core.eodata.EOPatch>`
        :return: A new eopatch.
        :rtype: EOPatch
        """
        return EOPatch(**kwargs)
