"""
The eodata module provides core objects for handling remote sensing multi-temporal data (such as satellite imagery).

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import datetime as dt
import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    TypeVar,
    cast,
)
from warnings import warn

import geopandas as gpd
import numpy as np
from fs.base import FS
from typing_extensions import deprecated

from sentinelhub import CRS, BBox, parse_time

from .constants import FEATURETYPE_DEPRECATION_MSG, TIMESTAMP_COLUMN, FeatureType, OverwritePermission
from .eodata_io import FeatureIO, load_eopatch_content, save_eopatch
from .exceptions import EODeprecationWarning, TemporalDimensionWarning
from .types import EllipsisType, Feature, FeaturesSpecification
from .utils.common import deep_eq, is_discrete_type
from .utils.fs import get_filesystem
from .utils.parsing import parse_features

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)
MISSING_BBOX_WARNING = (
    "Initializing an EOPatch without providing a BBox will no longer be possible in the future."
    " EOPatches represent geolocated data and so any EOPatch without a BBox is ill-formed. Consider"
    " using a different data structure for non-geolocated data."
)
TIMESTAMP_RENAME_WARNING = "The attribute `timestamp` is deprecated, use `timestamps` instead."

MAX_DATA_REPR_LEN = 100

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from eolearn.visualization import PlotBackend, PlotConfig


class _FeatureDict(MutableMapping[str, T], metaclass=ABCMeta):
    """A dictionary structure that holds features of certain feature type.

    It checks that features have a correct and dimension. It also supports lazy loading by accepting a function as a
    feature value, which is then called when the feature is accessed.
    """

    FORBIDDEN_CHARS: ClassVar[set[str]] = {".", "/", "\\", "|", ";", ":", "\n", "\t"}

    def __init__(self, feature_dict: Mapping[str, T | FeatureIO[T]], feature_type: FeatureType):
        """
        :param feature_dict: A dictionary of feature names and values
        :param feature_type: Type of features
        """
        super().__init__()

        self.feature_type = feature_type

        # we need trigger parsing and validation
        self._content: dict[str, T | FeatureIO[T]] = {}
        for key, value in feature_dict.items():
            self[key] = value

    def __setitem__(self, feature_name: str, value: T | FeatureIO[T]) -> None:
        """Before setting value to the dictionary it checks that value is of correct type and dimension and tries to
        transform value in correct form.
        """
        if not isinstance(value, FeatureIO):
            value = self._parse_feature_value(value, feature_name)
        self._check_feature_name(feature_name)
        self._content[feature_name] = value

    def _check_feature_name(self, feature_name: str) -> None:
        """Ensures that feature names are strings and do not contain forbidden characters."""
        if not isinstance(feature_name, str):
            raise ValueError(f"Feature name must be a string but an object of type {type(feature_name)} was given.")

        for char in feature_name:
            if char in self.FORBIDDEN_CHARS:
                raise ValueError(f"The feature name of {feature_name} contains an illegal character '{char}'.")

        if feature_name == "":
            raise ValueError("Feature name cannot be an empty string.")

    def __getitem__(self, feature_name: str) -> T:
        """Implements lazy loading."""
        value = self._content[feature_name]

        if isinstance(value, FeatureIO):
            value = cast(FeatureIO[T], value)  # not sure why mypy borks this one
            value = value.load()
            self._content[feature_name] = value

        return value

    def _get_unloaded(self, feature_name: str) -> T | FeatureIO[T]:
        """Returns the value, bypassing lazy-loading mechanisms."""
        return self._content[feature_name]

    def __delitem__(self, feature_name: str) -> None:
        del self._content[feature_name]

    def __eq__(self, other: object) -> bool:
        # default doesn't know how to compare numpy arrays
        return deep_eq(self, other)

    def __len__(self) -> int:
        return len(self._content)

    def __contains__(self, key: object) -> bool:
        return key in self._content

    def __iter__(self) -> Iterator[str]:
        return iter(self._content)

    @abstractmethod
    def _parse_feature_value(self, value: object, feature_name: str) -> T:
        """Checks if value fits the feature type. If not it tries to fix it or raise an error.

        :raises: ValueError
        """


class _FeatureDictNumpy(_FeatureDict[np.ndarray]):
    """_FeatureDict object specialized for Numpy arrays."""

    def __init__(
        self,
        feature_dict: Mapping[str, np.ndarray | FeatureIO[np.ndarray]],
        feature_type: FeatureType,
        *,
        temporal_dim: int | None,
    ):
        self._temporal_dim = temporal_dim
        super().__init__(feature_dict, feature_type)

    def _parse_feature_value(self, value: object, feature_name: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{self.feature_type} feature has to be a numpy array.")

        expected_ndim = cast(int, self.feature_type.ndim())  # numpy features have ndim
        if value.ndim != expected_ndim:
            raise ValueError(
                f"Numpy array of {self.feature_type} feature has to have {expected_ndim} "
                f"dimension{'s' if expected_ndim > 1 else ''} but feature {feature_name} has {value.ndim}."
            )

        if self.feature_type.is_temporal():
            if self._temporal_dim is None:
                msg = (
                    f"Adding temporal feature {(self.feature_type, feature_name)} to EOPatch without a temporal"
                    " definition (no timestamps)."
                )
                warnings.warn(msg, category=TemporalDimensionWarning, stacklevel=4)
            elif self._temporal_dim != value.shape[0]:
                msg = (
                    f"Missmatch in temporal dimensions when adding {(self.feature_type, feature_name)}: EOPatch has"
                    f" {self._temporal_dim} timestamps while the value has a temporal size of {value.shape[0]}."
                )
                warnings.warn(msg, category=TemporalDimensionWarning, stacklevel=4)

        if self.feature_type.is_discrete() and not is_discrete_type(value.dtype):
            raise ValueError(
                f"{self.feature_type} is a discrete feature type therefore dtype of data array "
                f"has to be either integer or boolean type but feature {feature_name} has dtype {value.dtype.type}."
            )

        return value

    def _update_temporal_dim(self, new_value: int | None) -> None:
        self._temporal_dim = new_value
        if not self.feature_type.is_temporal():
            return
        if self._temporal_dim is None:
            if self._content:
                warnings.warn(
                    f"EOPatch does not have timestamps, but has temporal features of type {self.feature_type}.",
                    category=TemporalDimensionWarning,
                    stacklevel=4,
                )
            return

        for feature_name, value in self._content.items():
            if not isinstance(value, FeatureIO) and value.shape[0] != self._temporal_dim:
                msg = (
                    f"Missmatch in temporal dimensions. The EOPatch has {self._temporal_dim} timestamps while"
                    f" {(self.feature_type, feature_name)} has a temporal size of {value.shape[0]}."
                )
                warnings.warn(msg, category=TemporalDimensionWarning, stacklevel=4)


class _FeatureDictGeoDf(_FeatureDict[gpd.GeoDataFrame]):
    """_FeatureDict object specialized for GeoDataFrames."""

    def _parse_feature_value(self, value: object, feature_name: str) -> gpd.GeoDataFrame:
        if isinstance(value, gpd.GeoSeries):
            value = gpd.GeoDataFrame(geometry=value, crs=value.crs)

        if isinstance(value, gpd.GeoDataFrame):
            if self.feature_type is FeatureType.VECTOR and TIMESTAMP_COLUMN not in value:
                raise ValueError(
                    f"{self.feature_type} feature has to contain a column '{TIMESTAMP_COLUMN}' with timestamps but "
                    f"feature {feature_name} does not not have it."
                )

            return value

        raise ValueError(
            f"{self.feature_type} feature works with data of type {gpd.GeoDataFrame.__name__} but feature "
            f"{feature_name} has data of type {type(value)}."
        )


class _FeatureDictJson(_FeatureDict[Any]):
    """_FeatureDict object specialized for meta-info."""

    def _parse_feature_value(self, value: object, _: str) -> Any:
        return value


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

    Currently, the EOPatch object doesn't enforce that the length of timestamps be equal to n_times dimensions of numpy
    arrays in other attributes.
    """

    # establish types of property value holders
    _timestamps: list[dt.datetime] | None
    _bbox: BBox | None

    def __init__(
        self,
        *,
        data: Mapping[str, np.ndarray] | None = None,
        mask: Mapping[str, np.ndarray] | None = None,
        scalar: Mapping[str, np.ndarray] | None = None,
        label: Mapping[str, np.ndarray] | None = None,
        vector: Mapping[str, gpd.GeoDataFrame] | None = None,
        data_timeless: Mapping[str, np.ndarray] | None = None,
        mask_timeless: Mapping[str, np.ndarray] | None = None,
        scalar_timeless: Mapping[str, np.ndarray] | None = None,
        label_timeless: Mapping[str, np.ndarray] | None = None,
        vector_timeless: Mapping[str, gpd.GeoDataFrame] | None = None,
        meta_info: Mapping[str, Any] | None = None,
        bbox: BBox | None = None,
        timestamps: list[dt.datetime] | None = None,
    ):
        self.bbox = bbox
        self._timestamps = self._parse_timestamps(timestamps)  # see timestamps setter why this is direct

        # the __setattr__ transforms the below to FeatureDicts and checks temporal dimensions if necessary
        self.data: MutableMapping[str, np.ndarray] = dict(data or {})
        self.mask: MutableMapping[str, np.ndarray] = dict(mask or {})
        self.scalar: MutableMapping[str, np.ndarray] = dict(scalar or {})
        self.label: MutableMapping[str, np.ndarray] = dict(label or {})
        self.vector: MutableMapping[str, gpd.GeoDataFrame] = dict(vector or {})
        self.data_timeless: MutableMapping[str, np.ndarray] = dict(data_timeless or {})
        self.mask_timeless: MutableMapping[str, np.ndarray] = dict(mask_timeless or {})
        self.scalar_timeless: MutableMapping[str, np.ndarray] = dict(scalar_timeless or {})
        self.label_timeless: MutableMapping[str, np.ndarray] = dict(label_timeless or {})
        self.vector_timeless: MutableMapping[str, gpd.GeoDataFrame] = dict(vector_timeless or {})
        self.meta_info: MutableMapping[str, Any] = dict(meta_info or {})

    @property
    def timestamp(self) -> list[dt.datetime] | None:
        """A property for handling the deprecated timestamp attribute."""
        warn(TIMESTAMP_RENAME_WARNING, category=EODeprecationWarning, stacklevel=2)
        return self.timestamps

    @timestamp.setter
    def timestamp(self, value: list[dt.datetime]) -> None:
        warn(TIMESTAMP_RENAME_WARNING, category=EODeprecationWarning, stacklevel=2)
        self.timestamps = value

    @property
    def timestamps(self) -> list[dt.datetime] | None:
        """A property for handling the `timestamps` attribute."""
        return self._timestamps

    @timestamps.setter
    def timestamps(self, value: Iterable[dt.datetime] | None) -> None:
        # The first setting of timestamps is done directly to `_timestamps` to avoid updating temporal dims of
        # nonexisting feature-dict attributes
        self._timestamps = self._parse_timestamps(value)

        temporal_dim = len(self._timestamps) if self._timestamps is not None else None
        for ftype in FeatureType:
            if ftype.is_temporal() and ftype.is_array():
                self[ftype]._update_temporal_dim(temporal_dim)  # noqa: SLF001 # pylint: disable=protected-access

    def _parse_timestamps(self, value: Iterable[dt.datetime | str] | None) -> list[dt.datetime] | None:
        if value is None:
            return None
        if isinstance(value, Iterable) and all(isinstance(time, (dt.date, str)) for time in value):
            return [parse_time(time, force_datetime=True) for time in value]
        raise TypeError(f"Cannot assign {value} as timestamps. Should be a sequence of datetime.datetime objects.")

    def get_timestamps(
        self, message_on_failure: str = "This EOPatch does not contain timestamps."
    ) -> list[dt.datetime]:
        """Returns the `timestamps` attribute if the EOPatch is temporally defined. Fails otherwise."""
        if self._timestamps is None:
            raise RuntimeError(message_on_failure)
        return self._timestamps

    @property
    def bbox(self) -> BBox | None:
        """A property for handling the `bbox` attribute."""
        return self._bbox

    @bbox.setter
    def bbox(self, value: BBox | None) -> None:
        if not (isinstance(value, BBox) or value is None):
            raise TypeError(f"Cannot assign {value} as bbox. Should be a `BBox` object.")
        if value is None:
            warn(MISSING_BBOX_WARNING, category=EODeprecationWarning, stacklevel=2)
        self._bbox = value

    def __setattr__(self, key: str, value: object) -> None:
        """Casts dictionaries to _FeatureDict objects for non-meta features."""

        if FeatureType.has_value(key) and key not in ("bbox", "timestamps"):
            if not isinstance(value, (dict, _FeatureDict)):
                raise TypeError(f"Cannot parse {value} for attribute {key}. Should be a dictionary.")

            feature_type = FeatureType(key)
            if feature_type.is_vector():
                value = _FeatureDictGeoDf(value, feature_type)
            elif feature_type is FeatureType.META_INFO:
                value = _FeatureDictJson(value, feature_type)
            else:
                temporal_dim = None if self.timestamps is None else len(self.timestamps)
                value = _FeatureDictNumpy(value, feature_type, temporal_dim=temporal_dim)

        super().__setattr__(key, value)

    def __getitem__(self, key: FeatureType | tuple[FeatureType, str | None | EllipsisType]) -> Any:
        """Provides features of requested feature type. It can also accept a tuple of (feature_type, feature_name).

        :param key: Feature type or a (feature_type, feature_name) pair.
        """
        feature_type, feature_name = key if isinstance(key, tuple) else (key, None)
        value = getattr(self, FeatureType(feature_type).value)
        if feature_name not in (None, Ellipsis) and isinstance(value, _FeatureDict):
            feature_name = cast(str, feature_name)  # the above check deals with ... and None
            return value[feature_name]
        return value

    def __setitem__(self, key: FeatureType | tuple[FeatureType, str | None | EllipsisType], value: Any) -> None:
        """Sets a new value to the given FeatureType or tuple of (feature_type, feature_name)."""
        feature_type, feature_name = key if isinstance(key, tuple) else (key, None)
        ftype_attr = FeatureType(feature_type).value

        if feature_name not in (None, Ellipsis):
            getattr(self, ftype_attr)[feature_name] = value
        else:
            setattr(self, ftype_attr, value)

    def __delitem__(self, feature: FeatureType | Feature) -> None:
        """Deletes the selected feature type or feature."""
        if isinstance(feature, tuple):
            feature_type, feature_name = feature
            del self[feature_type][feature_name]
            return

        self[FeatureType(feature)] = {}

    def __eq__(self, other: object) -> bool:
        """True if FeatureType attributes, bbox, and timestamps of both EOPatches are equal by value."""
        if not isinstance(other, type(self)):
            return False

        if self.bbox != other.bbox or self.timestamps != other.timestamps:
            return False

        return all(deep_eq(self[feature_type], other[feature_type]) for feature_type in FeatureType)

    def __contains__(self, key: object) -> bool:
        # `key` does not have a precise type, because otherwise `mypy` defaults to inclusion using `__iter__` and
        # the error message becomes incomprehensible.
        if isinstance(key, FeatureType):
            return bool(self[key])
        if isinstance(key, tuple) and len(key) == 2:
            ftype, fname = key
            return fname in self[ftype]
        raise ValueError(
            f"Membership checking is only implemented for elements of type `{FeatureType.__name__}` and for "
            "`(feature_type, feature_name)` pairs."
        )

    @deprecated(
        "The `+` operator for EOPatches has been deprecated. Use the function `eolearn.core.merge_eopatches` instead.",
        category=EODeprecationWarning,
    )
    def __add__(self, other: EOPatch) -> EOPatch:
        """Merges two EOPatches into a new EOPatch."""
        return self.merge(other)

    def __repr__(self) -> str:
        feature_repr_list = []
        if self.bbox is not None:
            feature_repr_list.append(f"bbox={self.bbox!r}")
        if self.timestamps is not None:
            feature_repr_list.append(f"timestamps={self._repr_value(self.timestamps)}")
        for feature_type in {ftype for ftype, _ in self.get_features()}:
            content = self[feature_type]

            content = {k: content._get_unloaded(k) for k in content}  # noqa: SLF001
            inner_content_repr = "\n    ".join(
                [f"{label}: {self._repr_value(value)}" for label, value in sorted(content.items())]
            )
            content_str = "{\n    " + inner_content_repr + "\n  }"

            feature_repr_list.append(f"{feature_type.value}={content_str}")

        feature_repr = "\n  ".join(feature_repr_list)
        if feature_repr:
            feature_repr = f"\n  {feature_repr}\n"
        return f"{self.__class__.__name__}({feature_repr})"

    @staticmethod
    def _repr_value(value: object) -> str:
        """Creates a representation string for different types of data."""
        if isinstance(value, np.ndarray):
            return f"{EOPatch._repr_value_class(value)}(shape={value.shape}, dtype={value.dtype})"

        if isinstance(value, gpd.GeoDataFrame):
            crs = CRS(value.crs).ogc_string() if value.crs else value.crs
            return f"{EOPatch._repr_value_class(value)}(columns={list(value)}, length={len(value)}, crs={crs})"

        repr_str = str(value)
        if len(repr_str) <= MAX_DATA_REPR_LEN:
            return repr_str

        if isinstance(value, (list, tuple, dict)) and value:
            lb, rb = ("[", "]") if isinstance(value, list) else ("(", ")") if isinstance(value, tuple) else ("{", "}")

            if isinstance(value, dict):  # generate representation of first element or (key, value) pair
                some_key = next(iter(value))
                repr_of_el = f"{EOPatch._repr_value(some_key)}: {EOPatch._repr_value(value[some_key])}"
            else:
                repr_of_el = EOPatch._repr_value(value[0])

            many_elements_visual = ", ..." if len(value) > 1 else ""  # add ellipsis if there are multiple elements
            repr_str = f"{lb}{repr_of_el}{many_elements_visual}{rb}"

            if len(repr_str) > MAX_DATA_REPR_LEN:
                repr_str = str(type(value))

            return f"{repr_str}<length={len(value)}>"

        return str(type(value))

    @staticmethod
    def _repr_value_class(value: object) -> str:
        """A representation of a class of a given value"""
        cls = value.__class__
        return ".".join([cls.__module__.split(".")[0], cls.__name__])

    def __copy__(
        self, features: FeaturesSpecification = ..., copy_timestamps: bool | Literal["auto"] = "auto"
    ) -> EOPatch:
        """Returns a new EOPatch with shallow copies of given features.

        :param features: A collection of features or feature types that will be copied into new EOPatch.
        :param copy_timestamps: Copy timestamps to the new EOPatch. By default copies them over if all
            features are copied or if any temporal features are getting copied.
        """
        if not features:  # For some reason deepcopy and copy pass {} by default
            features = ...
        patch_features = parse_features(features, eopatch=self)
        if copy_timestamps == "auto":
            copy_timestamps = features is ... or any(ftype.is_temporal() for ftype, _ in patch_features)

        new_eopatch = EOPatch(bbox=copy.copy(self.bbox))
        if copy_timestamps:
            new_eopatch.timestamps = copy.copy(self.timestamps)

        for feature_type, feature_name in patch_features:
            new_eopatch[feature_type][feature_name] = self[feature_type]._get_unloaded(feature_name)  # noqa: SLF001
        return new_eopatch

    def __deepcopy__(
        self,
        memo: dict | None = None,
        features: FeaturesSpecification = ...,
        copy_timestamps: bool | Literal["auto"] = "auto",
    ) -> EOPatch:
        """Returns a new EOPatch with deep copies of given features.

        :param memo: built-in parameter for memoization
        :param features: A collection of features or feature types that will be copied into new EOPatch.
        :param copy_timestamps: Copy timestamps to the new EOPatch. By default copies them over if all
            features are copied or if any temporal features are getting copied.
        """
        if not features:  # For some reason deepcopy and copy pass {} by default
            features = ...
        patch_features = parse_features(features, eopatch=self)
        if copy_timestamps == "auto":
            copy_timestamps = features is ... or any(ftype.is_temporal() for ftype, _ in patch_features)

        new_eopatch = EOPatch(bbox=copy.deepcopy(self.bbox))
        if copy_timestamps:
            new_eopatch.timestamps = copy.deepcopy(self.timestamps)

        for feature_type, feature_name in parse_features(features, eopatch=self):
            value = self[feature_type]._get_unloaded(feature_name)  # noqa: SLF001

            if isinstance(value, FeatureIO):
                # We cannot deepcopy the entire object because of the filesystem attribute
                value = copy.copy(value)
                value.loaded_value = copy.deepcopy(value.loaded_value, memo=memo)
            else:
                value = copy.deepcopy(value, memo=memo)

            new_eopatch[feature_type][feature_name] = value

        return new_eopatch

    def copy(
        self,
        features: FeaturesSpecification = ...,
        deep: bool = False,
        copy_timestamps: bool | Literal["auto"] = "auto",
    ) -> EOPatch:
        """Get a copy of the current `EOPatch`.

        :param features: Features to be copied into a new `EOPatch`. By default, all features will be copied.
        :param deep: If `True` it will make a deep copy of all data inside the `EOPatch`. Otherwise, only a shallow copy
            of `EOPatch` will be made. Note that `BBOX` and `TIMESTAMPS` will be copied even with a shallow copy.
        :param copy_timestamps: Copy timestamps to the new EOPatch. By default copies them over if all
            features are copied or if any temporal features are getting copied.
        :return: An EOPatch copy.
        """
        # pylint: disable=unnecessary-dunder-call
        if deep:
            return self.__deepcopy__(features=features, copy_timestamps=copy_timestamps)
        return self.__copy__(features=features, copy_timestamps=copy_timestamps)

    def get_spatial_dimension(self, feature_type: FeatureType, feature_name: str) -> tuple[int, int]:
        """
        Returns a tuple of spatial dimensions (height, width) of a feature.

        :param feature_type: Type of the feature
        :param feature_name: Name of the feature
        """
        if feature_type.is_array() and feature_type.is_spatial():
            shape = self[feature_type][feature_name].shape
            return shape[1:3] if feature_type.is_temporal() else shape[0:2]

        raise ValueError(f"Features of type {feature_type} do not have a spatial dimension or are not arrays.")

    def get_features(self) -> list[Feature]:
        """Returns a list of all non-empty features of EOPatch.

        :return: List of non-empty features
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=FEATURETYPE_DEPRECATION_MSG.format(".*?", ".*?"))
            removed_ftypes = {FeatureType.BBOX, FeatureType.TIMESTAMPS}  # list comprehensions make ignoring hard
        return [
            (feature_type, feature_name)
            for feature_type in (ftype for ftype in FeatureType if ftype not in removed_ftypes)
            for feature_name in self[feature_type]
        ]

    def save(
        self,
        path: str,
        features: FeaturesSpecification = ...,
        overwrite_permission: OverwritePermission = OverwritePermission.ADD_ONLY,
        filesystem: FS | None = None,
        *,
        save_timestamps: bool | Literal["auto"] = "auto",
        use_zarr: bool = False,
        temporal_selection: None | slice | list[int] | Literal["infer"] = None,
        compress_level: int | None = None,
    ) -> None:
        """Method to save an EOPatch from memory to a storage.

        :param path: A location where to save EOPatch. It can be either a local path or a remote URL path.
        :param features: A collection of features types specifying features of which type will be saved. By default,
            all features will be saved.
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the `path`
            parameter.
        :save_timestamps: Save the timestamps of the EOPatch. With the `"auto"` setting timestamps are saved
            if `features=...` or if other temporal features are being saved.
        :param use_zarr: Saves numpy-array based features into Zarr files. Requires ZARR extra dependencies.
        :param temporal_selection: Writes all of the data to the chosen temporal indices of preexisting arrays. Can be
            used for saving data in multiple steps for memory optimization. When set to `"infer"` it will match the
            timestamps of the EOPatch to the timestamps of the stored EOPatch to calculate indices.
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=True)
            path = "/"

        if compress_level is not None:
            warnings.warn(
                "The `compress_level` parameter has been deprecated, data is now compressed by default.",
                category=EODeprecationWarning,
                stacklevel=2,
            )

        save_eopatch(
            self,
            filesystem,
            path,
            features=features,
            overwrite_permission=OverwritePermission(overwrite_permission),
            save_timestamps=save_timestamps,
            use_zarr=use_zarr,
            temporal_selection=temporal_selection,
        )

    @staticmethod
    def load(
        path: str,
        features: FeaturesSpecification = ...,
        lazy_loading: bool = False,
        filesystem: FS | None = None,
        *,
        load_timestamps: bool | Literal["auto"] = "auto",
        temporal_selection: None | slice | list[int] | Callable[[list[dt.datetime]], list[bool]] = None,
    ) -> EOPatch:
        """Method to load an EOPatch from a storage into memory.

        :param path: A location from where to load EOPatch. It can be either a local path or a remote URL path.
        :param features: A collection of features to be loaded. By default, all features will be loaded.
        :param lazy_loading: If `True` features will be lazy loaded.
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the `path`
            parameter.
        :load_timestamps: Load the timestamps of the EOPatch. With the `"auto"` setting timestamps are loaded
            if `features=...` or if other temporal features are being loaded.
        :param temporal_selection: Only loads data corresponding to the chosen indices. Can also be a callable that,
            given a list of timestamps, returns a list of booleans declaring which temporal slices to load.
        :return: Loaded EOPatch
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=False)
            path = "/"

        bbox, timestamps, features_dict = load_eopatch_content(
            filesystem, path, features=features, temporal_selection=temporal_selection, load_timestamps=load_timestamps
        )
        eopatch = EOPatch(bbox=bbox, timestamps=timestamps)

        for feature, feature_io in features_dict.items():
            eopatch[feature] = feature_io

        if not lazy_loading:
            _trigger_loading_for_eopatch_features(eopatch)
        return eopatch

    @deprecated(
        "The EOPatch method `merge` has been deprecated. Use the function `eolearn.core.merge_eopatches` instead.",
        category=EODeprecationWarning,
    )
    def merge(
        self,
        *eopatches: EOPatch,
        features: FeaturesSpecification = ...,
        time_dependent_op: Literal[None, "concatenate", "min", "max", "mean", "median"] | Callable = None,
        timeless_op: Literal[None, "concatenate", "min", "max", "mean", "median"] | Callable = None,
    ) -> EOPatch:
        """Merge features of given EOPatches into a new EOPatch.

        :param eopatches: Any number of EOPatches to be merged together with the current EOPatch
        :param features: A collection of features to be merged together. By default, all features will be merged.
        :param time_dependent_op: An operation to be used to join data for any time-dependent raster feature. Before
            joining time slices of all arrays will be sorted. Supported options are:

            - None (default): If time slices with matching timestamps have the same values, take one. Raise an error
              otherwise.
            - 'concatenate': Keep all time slices, even the ones with matching timestamps
            - 'min': Join time slices with matching timestamps by taking minimum values. Ignore NaN values.
            - 'max': Join time slices with matching timestamps by taking maximum values. Ignore NaN values.
            - 'mean': Join time slices with matching timestamps by taking mean values. Ignore NaN values.
            - 'median': Join time slices with matching timestamps by taking median values. Ignore NaN values.
        :param timeless_op: An operation to be used to join data for any timeless raster feature. Supported options
            are:

            - None (default): If arrays are the same, take one. Raise an error otherwise.
            - 'concatenate': Join arrays over the last (i.e. bands) dimension
            - 'min': Join arrays by taking minimum values. Ignore NaN values.
            - 'max': Join arrays by taking maximum values. Ignore NaN values.
            - 'mean': Join arrays by taking mean values. Ignore NaN values.
            - 'median': Join arrays by taking median values. Ignore NaN values.
        :return: A merged EOPatch
        """
        from .eodata_merge import merge_eopatches  # pylint: disable=import-outside-toplevel, cyclic-import

        return merge_eopatches(
            self, *eopatches, features=features, time_dependent_op=time_dependent_op, timeless_op=timeless_op
        )

    @deprecated(
        "The method `consolidate_timestamps` has been deprecated. Use the method `temporal_subset` instead.",
        category=EODeprecationWarning,
    )
    def consolidate_timestamps(self, timestamps: list[dt.datetime]) -> set[dt.datetime]:
        """Removes all frames from the EOPatch with a date not found in the provided timestamps list.

        :param timestamps: keep frames with date found in this list
        :return: set of removed frames' dates
        """
        old_timestamps = self.get_timestamps()
        remove_from_patch = set(old_timestamps).difference(timestamps)
        remove_from_patch_idxs = [old_timestamps.index(rm_date) for rm_date in remove_from_patch]
        good_timestamp_idxs = [idx for idx, _ in enumerate(old_timestamps) if idx not in remove_from_patch_idxs]
        good_timestamps = [date for idx, date in enumerate(old_timestamps) if idx not in remove_from_patch_idxs]

        with warnings.catch_warnings():  # catch all temporal dimension related warnings
            warnings.simplefilter("ignore", category=TemporalDimensionWarning)
            self.timestamps = good_timestamps

        for ftype in FeatureType:
            if ftype.is_timeless() or ftype.is_meta() or ftype.is_vector():
                continue
            for feature_name, value in self[ftype].items():
                self[ftype, feature_name] = value[good_timestamp_idxs, ...]

        return remove_from_patch

    def temporal_subset(
        self, timestamps: Iterable[dt.datetime] | Iterable[int] | Callable[[list[dt.datetime]], Iterable[bool]]
    ) -> EOPatch:
        """Returns an EOPatch that only contains data for the temporal subset corresponding to `timestamps`.

        For array-based data appropriate temporal slices are extracted. For vector data a filtration is performed.

        :param timestamps: Parameter that defines the temporal subset. Can be a collection of timestamps, a
            collection of timestamp indices. It is possible to also provide a callable that maps a list of timestamps
            to a sequence of booleans, which determine if a given timestamp is included in the subset or not.
        """
        timestamp_indices = self._parse_temporal_subset_input(timestamps)
        new_timestamps = [ts for i, ts in enumerate(self.get_timestamps()) if i in timestamp_indices]
        new_patch = EOPatch(bbox=self.bbox, timestamps=new_timestamps)

        for ftype, fname in self.get_features():
            if ftype.is_timeless() or ftype.is_meta():
                new_patch[ftype, fname] = self[ftype, fname]
            elif ftype.is_vector():
                gdf: gpd.GeoDataFrame = self[ftype, fname]
                new_patch[ftype, fname] = gdf[gdf[TIMESTAMP_COLUMN].isin(new_timestamps)]
            else:
                new_patch[ftype, fname] = self[ftype, fname][timestamp_indices]

        return new_patch

    def _parse_temporal_subset_input(
        self, timestamps: Iterable[dt.datetime] | Iterable[int] | Callable[[list[dt.datetime]], Iterable[bool]]
    ) -> list[int]:
        """Parses input into a list of timestamp indices. Also adds implicit support for strings via `parse_time`."""
        if callable(timestamps):
            accepted_timestamps = timestamps(self.get_timestamps())
            return [i for i, accepted in enumerate(accepted_timestamps) if accepted]
        ts_or_idx = list(timestamps)
        if all(isinstance(ts, int) for ts in ts_or_idx):
            return ts_or_idx  # type: ignore[return-value]
        parsed_timestamps = {parse_time(ts, force_datetime=True) for ts in ts_or_idx}  # type: ignore[call-overload]
        return [i for i, ts in enumerate(self.get_timestamps()) if ts in parsed_timestamps]

    def plot(
        self,
        feature: Feature,
        *,
        times: list[int] | slice | None = None,
        channels: list[int] | slice | None = None,
        channel_names: list[str] | None = None,
        rgb: tuple[int, int, int] | None = None,
        backend: str | PlotBackend = "matplotlib",
        config: PlotConfig | None = None,
        **kwargs: Any,
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
                "Dependencies `eo-learn[VISUALIZATION]` have to be installed in order to use EOPatch plotting."
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


def _trigger_loading_for_eopatch_features(eopatch: EOPatch) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(lambda: eopatch.bbox)
        executor.submit(lambda: eopatch.timestamps)
        list(executor.map(lambda feature: eopatch[feature], eopatch.get_features()))
