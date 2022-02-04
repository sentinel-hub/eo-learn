"""
The eodata module provides core objects for handling remote sensing multi-temporal data (such as satellite imagery).

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import logging
import copy
import datetime
from typing import Tuple, Union, List, Optional, TYPE_CHECKING

import attr
import dateutil.parser
import numpy as np
import geopandas as gpd

from sentinelhub import BBox, CRS

from .constants import FeatureType, OverwritePermission
from .eodata_io import save_eopatch, load_eopatch, FeatureIO
from .eodata_merge import merge_eopatches
from .utils.fs import get_filesystem
from .utils.common import deep_eq, is_discrete_type
from .utils.parsing import parse_features


LOGGER = logging.getLogger(__name__)

MAX_DATA_REPR_LEN = 100

if TYPE_CHECKING:
    try:
        from eolearn.visualization import PlotBackend
        from eolearn.visualization.eopatch_base import BasePlotConfig
    except ImportError:
        pass


@attr.s(repr=False, eq=False, kw_only=True)
class EOPatch:
    """The basic data object for multi-temporal remotely sensed data, such as satellite imagery and its derivatives.

    The EOPatch contains multi-temporal remotely sensed data of a single patch of earth's surface defined by the
    bounding box in specific coordinate reference system. The patch can be a rectangle, polygon, or pixel in space.
    The EOPatch object can also be used to store derived quantities, such as for example means, standard deviations,
    etc., of a patch. In this case the 'space' dimension is equivalent to a pixel.

    Primary goal of EOPatch is to store remotely sensed data, usually of a shape n_time x height x width x n_features
    images, where height and width are the numbers of pixels in y and x, n_features is the number of features
    (i.e. bands/channels, cloud probability, etc.), and n_time is the number of time-slices (the number of times this
    patch was recorded by the satellite; can also be a single image)

    In addition to that other auxiliary information is also needed and can be stored in additional attributes of the
    EOPatch (thus extending the functionality of numpy ndarray). These attributes are listed in the FeatureType enum.

    Currently the EOPatch object doesn't enforce that the length of timestamp be equal to n_times dimensions of numpy
    arrays in other attributes.
    """

    data = attr.ib(factory=dict)
    mask = attr.ib(factory=dict)
    scalar = attr.ib(factory=dict)
    label = attr.ib(factory=dict)
    vector = attr.ib(factory=dict)
    data_timeless = attr.ib(factory=dict)
    mask_timeless = attr.ib(factory=dict)
    scalar_timeless = attr.ib(factory=dict)
    label_timeless = attr.ib(factory=dict)
    vector_timeless = attr.ib(factory=dict)
    meta_info = attr.ib(factory=dict)
    bbox = attr.ib(default=None)
    timestamp = attr.ib(factory=list)

    def __setattr__(self, key, value, feature_name=None):
        """Raises TypeError if feature type attributes are not of correct type.

        In case they are a dictionary they are cast to _FeatureDict class.
        """
        if feature_name not in (None, Ellipsis) and FeatureType.has_value(key):
            self[key][feature_name] = value
            return

        if FeatureType.has_value(key) and not isinstance(value, FeatureIO):
            feature_type = FeatureType(key)
            value = self._parse_feature_type_value(feature_type, value)

        super().__setattr__(key, value)

    @staticmethod
    def _parse_feature_type_value(feature_type, value):
        """Checks or parses value which will be assigned to a feature type attribute of `EOPatch`. If the value
        cannot be parsed correctly it raises an error.

        :raises: TypeError, ValueError
        """
        if feature_type.has_dict() and isinstance(value, dict):
            return value if isinstance(value, _FeatureDict) else _FeatureDict(value, feature_type)

        if feature_type is FeatureType.BBOX:
            if value is None or isinstance(value, BBox):
                return value
            if isinstance(value, (tuple, list)) and len(value) == 5:
                return BBox(value[:4], crs=value[4])

        if feature_type is FeatureType.TIMESTAMP:
            if isinstance(value, (tuple, list)):
                return [
                    timestamp if isinstance(timestamp, datetime.date) else dateutil.parser.parse(timestamp)
                    for timestamp in value
                ]

        raise TypeError(
            f"Attribute {feature_type} requires value of type {feature_type.type()} - "
            f"failed to parse given value {value}"
        )

    def __getattribute__(self, key, load=True, feature_name=None):
        """Handles lazy loading and it can even provide a single feature from _FeatureDict."""
        value = super().__getattribute__(key)

        if isinstance(value, FeatureIO) and load:
            value = value.load()
            setattr(self, key, value)
            value = getattr(self, key)

        if feature_name not in (None, Ellipsis) and isinstance(value, _FeatureDict):
            return value[feature_name]

        return value

    def __getitem__(self, feature_type):
        """Provides features of requested feature type. It can also accept a tuple of (feature_type, feature_name).

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str or (FeatureType, str)
        :return: Dictionary of features
        """
        feature_name = None
        if isinstance(feature_type, tuple):
            self._check_tuple_key(feature_type)
            feature_type, feature_name = feature_type

        return self.__getattribute__(FeatureType(feature_type).value, feature_name=feature_name)

    def __setitem__(self, feature_type, value):
        """Sets a new dictionary / list to the given FeatureType. As a key it can also accept a tuple of
        (feature_type, feature_name).

        :param feature_type: Type of EOPatch feature
        :type feature_type: FeatureType or str or (FeatureType, str)
        :param value: New dictionary or list
        :type value: dict or list
        :return: Dictionary of features
        """
        feature_name = None
        if isinstance(feature_type, tuple):
            self._check_tuple_key(feature_type)
            feature_type, feature_name = feature_type

        return self.__setattr__(FeatureType(feature_type).value, value, feature_name=feature_name)

    def __delitem__(self, feature):
        """Deletes the selected feature.

        :param feature: EOPatch feature
        :type feature: (FeatureType, str)
        """
        self._check_tuple_key(feature)
        feature_type, feature_name = feature
        del self[feature_type][feature_name]

    @staticmethod
    def _check_tuple_key(key):
        """A helper function that checks a tuple, which should hold (feature_type, feature_name)."""
        if len(key) != 2:
            raise ValueError(f"Given element should be a tuple of (feature_type, feature_name), but {key} found.")

    def __eq__(self, other):
        """True if FeatureType attributes, bbox, and timestamps of both EOPatches are equal by value."""
        if not isinstance(self, type(other)):
            return False

        for feature_type in FeatureType:
            if not deep_eq(self[feature_type], other[feature_type]):
                return False
        return True

    def __contains__(self, feature: Union[FeatureType, Tuple[FeatureType, str]]):
        if isinstance(feature, FeatureType):
            return bool(self[feature])
        if isinstance(feature, tuple) and len(feature) == 2:
            ftype, fname = FeatureType(feature[0]), feature[1]
            if ftype.has_dict():
                return fname in self[ftype]
            return bool(self[ftype])
        raise ValueError(
            f"Membership checking is only implemented elements of type `{FeatureType.__name__}` and for "
            "`(feature_type, feature_name)` tuples."
        )

    def __add__(self, other):
        """Merges two EOPatches into a new EOPatch."""
        return self.merge(other)

    def __repr__(self):
        feature_repr_list = []
        for feature_type in FeatureType:
            content = self[feature_type]
            if not content:
                continue

            if isinstance(content, dict) and content:
                content_str = (
                    "{\n    "
                    + "\n    ".join([f"{label}: {self._repr_value(value)}" for label, value in sorted(content.items())])
                    + "\n  }"
                )
            else:
                content_str = self._repr_value(content)
            feature_repr_list.append(f"{feature_type.value}={content_str}")

        feature_repr = "\n  ".join(feature_repr_list)
        if feature_repr:
            feature_repr = f"\n  {feature_repr}\n"
        return f"{self.__class__.__name__}({feature_repr})"

    @staticmethod
    def _repr_value(value):
        """Creates a representation string for different types of data.

        :param value: data in any type
        :return: representation string
        :rtype: str
        """
        if isinstance(value, np.ndarray):
            return f"{EOPatch._repr_value_class(value)}(shape={value.shape}, dtype={value.dtype})"

        if isinstance(value, gpd.GeoDataFrame):
            crs = CRS(value.crs).ogc_string() if value.crs else value.crs
            return f"{EOPatch._repr_value_class(value)}(columns={list(value)}, length={len(value)}, crs={crs})"

        if isinstance(value, (list, tuple, dict)) and value:
            repr_str = str(value)
            if len(repr_str) <= MAX_DATA_REPR_LEN:
                return repr_str

            l_bracket, r_bracket = ("[", "]") if isinstance(value, list) else ("(", ")")
            if isinstance(value, (list, tuple)) and len(value) > 2:
                repr_str = f"{l_bracket}{repr(value[0])}, ..., {repr(value[-1])}{r_bracket}"

            if len(repr_str) > MAX_DATA_REPR_LEN and isinstance(value, (list, tuple)) and len(value) > 1:
                repr_str = f"{l_bracket}{repr(value[0])}, ...{r_bracket}"

            if len(repr_str) > MAX_DATA_REPR_LEN:
                repr_str = str(type(value))

            return f"{repr_str}, length={len(value)}"

        return repr(value)

    @staticmethod
    def _repr_value_class(value):
        """A representation of a class of a given value"""
        cls = value.__class__
        return ".".join([cls.__module__.split(".")[0], cls.__name__])

    def __copy__(self, features=...):
        """Returns a new EOPatch with shallow copies of given features.

        :param features: A collection of features or feature types that will be copied into new EOPatch.
        :type features: object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        if not features:  # For some reason deepcopy and copy pass {} by default
            features = ...

        new_eopatch = EOPatch()
        for feature_type, feature_name in parse_features(features, eopatch=self):
            if feature_type.has_dict():
                new_eopatch[feature_type][feature_name] = self[feature_type].__getitem__(feature_name, load=False)
            else:
                new_eopatch[feature_type] = copy.copy(self[feature_type])
        return new_eopatch

    def __deepcopy__(self, memo=None, features=...):
        """Returns a new EOPatch with deep copies of given features.

        :param memo: built-in parameter for memoization
        :type memo: dict
        :param features: A collection of features or feature types that will be copied into new EOPatch.
        :type features: object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        """
        if not features:  # For some reason deepcopy and copy pass {} by default
            features = ...

        new_eopatch = EOPatch()
        for feature_type, feature_name in parse_features(features, eopatch=self):
            if feature_type.has_dict():
                value = self[feature_type].__getitem__(feature_name, load=False)

                if isinstance(value, FeatureIO):
                    # We cannot deepcopy the entire object because of the filesystem attribute
                    value = copy.copy(value)
                    value.loaded_value = copy.deepcopy(value.loaded_value, memo=memo)
                else:
                    value = copy.deepcopy(value, memo=memo)

                new_eopatch[feature_type][feature_name] = value
            else:
                new_eopatch[feature_type] = copy.deepcopy(self[feature_type], memo=memo)

        return new_eopatch

    def copy(self, features=..., deep=False):
        """Get a copy of the current `EOPatch`.

        :param features: Features to be copied into a new `EOPatch`. By default all features will be copied.
        :type features: object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param deep: If `True` it will make a deep copy of all data inside the `EOPatch`. Otherwise only a shallow copy
            of `EOPatch` will be made. Note that `BBOX` and `TIMESTAMP` will be copied even with a shallow copy.
        :type deep: bool
        :return: An EOPatch copy.
        :rtype: EOPatch
        """
        if deep:
            return self.__deepcopy__(features=features)
        return self.__copy__(features=features)

    def reset_feature_type(self, feature_type):
        """Resets the values of the given feature type.

        :param feature_type: Type of a feature
        :type feature_type: FeatureType
        """
        feature_type = FeatureType(feature_type)
        if feature_type.has_dict():
            self[feature_type] = {}
        elif feature_type is FeatureType.BBOX:
            self[feature_type] = None
        else:
            self[feature_type] = []

    def get_features(self):
        """Returns a dictionary of all non-empty features of EOPatch.

        The elements are either sets of feature names or a boolean `True` in case feature type has no dictionary of
        feature names.

        :return: A dictionary of features
        :rtype: dict(FeatureType: str or True)
        """
        feature_dict = {}
        for feature_type in FeatureType:
            if self[feature_type]:
                feature_dict[feature_type] = set(self[feature_type]) if feature_type.has_dict() else True

        return feature_dict

    def get_spatial_dimension(self, feature_type, feature_name):
        """
        Returns a tuple of spatial dimension (height, width) of a feature.

        The feature has to be spatial or time dependent.

        :param feature_type: Enum of the attribute
        :type feature_type: FeatureType
        :param feature_name: Name of the feature
        :type feature_name: str
        """
        if feature_type.is_temporal() or feature_type.is_spatial():
            shape = self[feature_type][feature_name].shape
            return shape[1:3] if feature_type.is_temporal() else shape[0:2]

        raise ValueError(
            "FeatureType used to determine the width and height of raster must be time dependent or spatial."
        )

    def get_feature_list(self) -> List[Union[FeatureType, Tuple[FeatureType, str]]]:
        """Returns a list of all non-empty features of EOPatch.

        The elements are either only FeatureType or a pair of FeatureType and feature name.

        :return: list of features
        """
        feature_list: List[Union[FeatureType, Tuple[FeatureType, str]]] = []
        for feature_type in FeatureType:
            if feature_type.has_dict():
                for feature_name in self[feature_type]:
                    feature_list.append((feature_type, feature_name))
            elif self[feature_type]:
                feature_list.append(feature_type)
        return feature_list

    def save(
        self, path, features=..., overwrite_permission=OverwritePermission.ADD_ONLY, compress_level=0, filesystem=None
    ):
        """Method to save an EOPatch from memory to a storage.

        :param path: A location where to save EOPatch. It can be either a local path or a remote URL path.
        :type path: str
        :param features: A collection of features types specifying features of which type will be saved. By default
            all features will be saved.
        :type features: list(FeatureType) or list((FeatureType, str)) or ...
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified with an integer from 0 (no compression)
            to 9 (highest compression).
        :type compress_level: int
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the `path`
            parameter.
        :type filesystem: fs.FS or None
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=True)
            path = "/"

        save_eopatch(
            self,
            filesystem,
            path,
            features=features,
            compress_level=compress_level,
            overwrite_permission=OverwritePermission(overwrite_permission),
        )

    @staticmethod
    def load(path, features=..., lazy_loading=False, filesystem=None):
        """Method to load an EOPatch from a storage into memory.

        :param path: A location from where to load EOPatch. It can be either a local path or a remote URL path.
        :type path: str
        :param features: A collection of features to be loaded. By default all features will be loaded.
        :type features: object
        :param lazy_loading: If `True` features will be lazy loaded.
        :type lazy_loading: bool
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the `path`
            parameter.
        :type filesystem: fs.FS or None
        :return: Loaded EOPatch
        :rtype: EOPatch
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=False)
            path = "/"

        return load_eopatch(EOPatch(), filesystem, path, features=features, lazy_loading=lazy_loading)

    def merge(self, *eopatches, features=..., time_dependent_op=None, timeless_op=None):
        """Merge features of given EOPatches into a new EOPatch.

        :param eopatches: Any number of EOPatches to be merged together with the current EOPatch
        :type eopatches: EOPatch
        :param features: A collection of features to be merged together. By default all features will be merged.
        :type features: object
        :param time_dependent_op: An operation to be used to join data for any time-dependent raster feature. Before
            joining time slices of all arrays will be sorted. Supported options are:

            - None (default): If time slices with matching timestamps have the same values, take one. Raise an error
              otherwise.
            - 'concatenate': Keep all time slices, even the ones with matching timestamps
            - 'min': Join time slices with matching timestamps by taking minimum values. Ignore NaN values.
            - 'max': Join time slices with matching timestamps by taking maximum values. Ignore NaN values.
            - 'mean': Join time slices with matching timestamps by taking mean values. Ignore NaN values.
            - 'median': Join time slices with matching timestamps by taking median values. Ignore NaN values.
        :type time_dependent_op: str or Callable or None
        :param timeless_op: An operation to be used to join data for any timeless raster feature. Supported options
            are:

            - None (default): If arrays are the same, take one. Raise an error otherwise.
            - 'concatenate': Join arrays over the last (i.e. bands) dimension
            - 'min': Join arrays by taking minimum values. Ignore NaN values.
            - 'max': Join arrays by taking maximum values. Ignore NaN values.
            - 'mean': Join arrays by taking mean values. Ignore NaN values.
            - 'median': Join arrays by taking median values. Ignore NaN values.
        :type timeless_op: str or Callable or None
        :return: A dictionary with EOPatch features and values
        :rtype: Dict[(FeatureType, str), object]
        """
        eopatch_content = merge_eopatches(
            self, *eopatches, features=features, time_dependent_op=time_dependent_op, timeless_op=timeless_op
        )

        merged_eopatch = EOPatch()
        for feature, value in eopatch_content.items():
            merged_eopatch[feature] = value

        return merged_eopatch

    def get_time_series(self, ref_date=None, scale_time=1):
        """Returns a numpy array with seconds passed between the reference date and the timestamp of each image.

        An array is constructed as time_series[i] = (timestamp[i] - ref_date).total_seconds().
        If reference date is None the first date in the EOPatch's timestamp is taken.
        If EOPatch timestamp attribute is empty the method returns None.

        :param ref_date: reference date relative to which the time is measured
        :type ref_date: datetime object
        :param scale_time: scale seconds by factor. If `60`, time will be in minutes, if `3600` hours
        :type scale_time: int
        """

        if not self.timestamp:
            return np.zeros(0, dtype=np.int64)

        if ref_date is None:
            ref_date = self.timestamp[0]

        return np.asarray(
            [round((timestamp - ref_date).total_seconds() / scale_time) for timestamp in self.timestamp], dtype=np.int64
        )

    def consolidate_timestamps(self, timestamps):
        """Removes all frames from the EOPatch with a date not found in the provided timestamps list.

        :param timestamps: keep frames with date found in this list
        :type timestamps: list of datetime objects
        :return: set of removed frames' dates
        :rtype: set of datetime objects
        """
        remove_from_patch = set(self.timestamp).difference(timestamps)
        remove_from_patch_idxs = [self.timestamp.index(rm_date) for rm_date in remove_from_patch]
        good_timestamp_idxs = [idx for idx, _ in enumerate(self.timestamp) if idx not in remove_from_patch_idxs]
        good_timestamps = [date for idx, date in enumerate(self.timestamp) if idx not in remove_from_patch_idxs]

        for feature_type in [
            feature_type for feature_type in FeatureType if (feature_type.is_temporal() and feature_type.has_dict())
        ]:

            for feature_name, value in self[feature_type].items():
                if isinstance(value, np.ndarray):
                    self[feature_type][feature_name] = value[good_timestamp_idxs, ...]
                if isinstance(value, list):
                    self[feature_type][feature_name] = [value[idx] for idx in good_timestamp_idxs]

        self.timestamp = good_timestamps
        return remove_from_patch

    def plot(
        self,
        feature,
        *,
        times: Union[List[int], slice, None] = None,
        channels: Union[List[int], slice, None] = None,
        channel_names: Optional[List[str]] = None,
        rgb: Optional[Tuple[int, int, int]] = None,
        backend: Union[str, PlotBackend] = "matplotlib",
        config: Optional[BasePlotConfig] = None,
        **kwargs,
    ) -> object:
        """Plots an `EOPatch` feature.

        :param feature: A feature in the `EOPatch`.
        :param times: A list or a slice of indices on temporal axis to be used for plotting. If not provided all
            indices will be used.
        :param channels: A list or a slice of indices on channels axis to be used for plotting. If not provided all
            indices will be used.
        :param channel_names: Names of channels of the last dimension in the given raster feature.
        :param rgb: If provided, it should be a list of 3 indices of RGB channels to be plotted. It will plot only RGB
            images with these channels. This only works for raster features with spatial dimension.
        :param backend: A type of plotting backend.
        :param config: A configuration object with advanced plotting parameters.
        :param kwargs: Parameters that are specific to a specified plotting backend.
        :return: A plot object that depends on the backend used.
        """
        # pylint: disable=import-outside-toplevel,raise-missing-from
        try:
            from eolearn.visualization.eopatch import plot_eopatch
        except ImportError:
            raise RuntimeError(
                "Subpackage eo-learn-visualization has to be installed in order to use EOPatch visualization method"
            )

        return plot_eopatch(
            self,
            feature=feature,
            times=times,
            channels=channels,
            channel_names=channel_names,
            rgb=rgb,
            backend=backend,
            config=config,
            **kwargs,
        )


class _FeatureDict(dict):
    """A dictionary structure that holds features of certain feature type.

    It checks that features have a correct and dimension. It also supports lazy loading by accepting a function as a
    feature value, which is then called when the feature is accessed.
    """

    FORBIDDEN_CHARS = {".", "/", "\\", "|", ";", ":", "\n", "\t"}

    def __init__(self, feature_dict, feature_type):
        """
        :param feature_dict: A dictionary of feature names and values
        :type feature_dict: dict(str: object)
        :param feature_type: Type of features
        :type feature_type: FeatureType
        """
        super().__init__()

        self.feature_type = feature_type
        self.ndim = self.feature_type.ndim()
        self.is_vector = self.feature_type.is_vector()

        for feature_name, value in feature_dict.items():
            self[feature_name] = value

    def __setitem__(self, feature_name, value):
        """Before setting value to the dictionary it checks that value is of correct type and dimension and tries to
        transform value in correct form.
        """
        value = self._parse_feature_value(value, feature_name)
        self._check_feature_name(feature_name)
        super().__setitem__(feature_name, value)

    def _check_feature_name(self, feature_name):
        if not isinstance(feature_name, str):
            raise ValueError(f"Feature name must be a string but an object of type {type(feature_name)} was given.")

        for char in feature_name:
            if char in self.FORBIDDEN_CHARS:
                raise ValueError(
                    f"The name of feature ({self.feature_type}, {feature_name}) contains an illegal character '{char}'."
                )

        if feature_name == "":
            raise ValueError("Feature name cannot be an empty string.")

    def __getitem__(self, feature_name, load=True):
        """Implements lazy loading."""
        value = super().__getitem__(feature_name)

        if isinstance(value, FeatureIO) and load:
            value = value.load()
            self[feature_name] = value

        return value

    def __eq__(self, other):
        """Compares its content against a content of another feature type dictionary."""
        return deep_eq(self, other)

    def __ne__(self, other):
        """Compares its content against a content of another feature type dictionary."""
        return not self.__eq__(other)

    def get_dict(self):
        """Returns a Python dictionary of features and value."""
        return dict(self)

    def _parse_feature_value(self, value, feature_name):
        """Checks if value fits the feature type. If not it tries to fix it or raise an error.

        :raises: ValueError
        """
        if isinstance(value, FeatureIO):
            return value
        if not hasattr(self, "ndim"):  # Because of serialization/deserialization during multiprocessing
            return value

        if self.ndim:
            if not isinstance(value, np.ndarray):
                raise ValueError(f"{self.feature_type} feature has to be a numpy array.")
            if value.ndim != self.ndim:
                raise ValueError(
                    f"Numpy array of {self.feature_type} feature has to have {self.ndim} "
                    f"dimension{'s' if self.ndim > 1 else ''} but feature {feature_name} has {value.ndim}."
                )

            if self.feature_type.is_discrete() and not is_discrete_type(value.dtype):
                raise ValueError(
                    f"{self.feature_type} is a discrete feature type therefore dtype of data array "
                    f"has to be either integer or boolean type but feature {feature_name} has dtype {value.dtype.type}."
                )

            return value

        if self.is_vector:
            if isinstance(value, gpd.GeoSeries):
                value = gpd.GeoDataFrame(dict(geometry=value), crs=value.crs)

            if isinstance(value, gpd.GeoDataFrame):
                if self.feature_type is FeatureType.VECTOR and FeatureType.TIMESTAMP.value.upper() not in value:
                    raise ValueError(
                        f"{self.feature_type} feature has to contain a column 'TIMESTAMP' with timestamps but "
                        f"feature {feature_name} doesn't not have it."
                    )

                return value

            raise ValueError(
                f"{self.feature_type} feature works with data of type {gpd.GeoDataFrame.__name__} but feature "
                f"{feature_name} has data of type {type(value)}."
            )

        return value
