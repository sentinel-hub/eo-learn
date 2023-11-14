"""
This module implements feature types used in EOPatch objects

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import warnings
from enum import Enum, EnumMeta
from typing import Any, TypeVar

from typing_extensions import deprecated

from sentinelhub import BBox, MimeType

from .exceptions import EODeprecationWarning

TIMESTAMP_COLUMN = "TIMESTAMP"
T = TypeVar("T")

FEATURETYPE_DEPRECATION_MSG = (
    "The `FeatureType.{}` has been deprecated and will be removed in the future. Use the EOPatch attribute `{}`"
    " directly."
)


def _warn_and_adjust(name: T) -> T:
    # since we stick with `UPPER` for attributes and `lower` for values, we include both to reuse function
    if isinstance(name, str):  # to avoid type issues
        if name in ("TIMESTAMP", "timestamp"):
            name = "TIMESTAMPS" if name == "TIMESTAMP" else "timestamps"  # type: ignore[assignment]

        if name in ("TIMESTAMPS", "BBOX", "timestamps", "bbox"):
            warnings.warn(
                FEATURETYPE_DEPRECATION_MSG.format(name.upper(), name.lower()),
                category=EODeprecationWarning,
                stacklevel=3,
            )
    return name


class EnumWithDeprecations(EnumMeta):
    """A custom EnumMeta class for catching the deprecated Enum members of the FeatureType Enum class."""

    def __getattribute__(cls, name: str) -> Any:  # noqa: N805
        return super().__getattribute__(_warn_and_adjust(name))

    def __getitem__(cls, name: str) -> Any:  # noqa: N805
        return super().__getitem__(_warn_and_adjust(name))

    def __call__(cls, value: str, *args: Any, **kwargs: Any) -> Any:  # noqa: N805
        return super().__call__(_warn_and_adjust(value), *args, **kwargs)


class FeatureType(Enum, metaclass=EnumWithDeprecations):
    """The Enum class of all possible feature types that can be included in EOPatch.

    List of feature types:
     - DATA with shape t x n x m x d: time- and position-dependent remote sensing data (e.g. bands) of type float
     - MASK with shape t x n x m x d: time- and position-dependent mask (e.g. ground truth, cloud/shadow mask,
       super pixel identifier) of type int
     - SCALAR with shape t x s: time-dependent and position-independent remote sensing data (e.g. weather data,) of
       type float
     - LABEL with shape t x s: time-dependent and position-independent label (e.g. ground truth) of type int
     - VECTOR: a list of time-dependent vector shapes in shapely.geometry classes
     - DATA_TIMELESS with shape n x m x d: time-independent and position-dependent remote sensing data (e.g.
       elevation model) of type float
     - MASK_TIMELESS with shape n x m x d: time-independent and position-dependent mask (e.g. ground truth,
       region of interest mask) of type int
     - SCALAR_TIMELESS with shape s:  time-independent and position-independent remote sensing data of type float
     - LABEL_TIMELESS with shape s: time-independent and position-independent label of type int
     - VECTOR_TIMELESS: time-independent vector shapes in shapely.geometry classes
     - META_INFO: dictionary of additional info (e.g. resolution, time difference)
     - BBOX: bounding box of the patch which is an instance of sentinelhub.BBox
     - TIMESTAMPS: list of dates which are instances of datetime.datetime
    """

    # IMPORTANT: these feature names must exactly match those in EOPatch constructor
    DATA = "data"
    MASK = "mask"
    SCALAR = "scalar"
    LABEL = "label"
    VECTOR = "vector"
    DATA_TIMELESS = "data_timeless"
    MASK_TIMELESS = "mask_timeless"
    SCALAR_TIMELESS = "scalar_timeless"
    LABEL_TIMELESS = "label_timeless"
    VECTOR_TIMELESS = "vector_timeless"
    META_INFO = "meta_info"
    BBOX = "bbox"
    TIMESTAMPS = "timestamps"

    @classmethod
    def has_value(cls, value: str) -> bool:
        """True if value is in FeatureType values. False otherwise."""
        return value in cls._value2member_map_

    def is_spatial(self) -> bool:
        """True if FeatureType has a spatial component. False otherwise."""
        return self in [
            FeatureType.DATA,
            FeatureType.MASK,
            FeatureType.VECTOR,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK_TIMELESS,
            FeatureType.VECTOR_TIMELESS,
        ]

    def is_temporal(self) -> bool:
        """True if FeatureType has a time component. False otherwise."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=FEATURETYPE_DEPRECATION_MSG.format(".*?", ".*?"))
            return self in [
                FeatureType.DATA,
                FeatureType.MASK,
                FeatureType.SCALAR,
                FeatureType.LABEL,
                FeatureType.VECTOR,
                FeatureType.TIMESTAMPS,
            ]

    def is_timeless(self) -> bool:
        """True if FeatureType doesn't have a time component and is not a meta feature. False otherwise."""
        return not (self.is_temporal() or self.is_meta())

    def is_discrete(self) -> bool:
        """True if FeatureType should have discrete (integer) values. False otherwise."""
        return self in [FeatureType.MASK, FeatureType.MASK_TIMELESS, FeatureType.LABEL, FeatureType.LABEL_TIMELESS]

    def is_meta(self) -> bool:
        """True if FeatureType is for storing metadata info and False otherwise."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=FEATURETYPE_DEPRECATION_MSG.format(".*?", ".*?"))
            return self in [FeatureType.META_INFO, FeatureType.BBOX, FeatureType.TIMESTAMPS]

    def is_vector(self) -> bool:
        """True if FeatureType is vector feature type. False otherwise."""
        return self in [FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]

    def is_array(self) -> bool:
        """True if FeatureType stores a dictionary with array data. False otherwise."""
        return self in [
            FeatureType.DATA,
            FeatureType.MASK,
            FeatureType.SCALAR,
            FeatureType.LABEL,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK_TIMELESS,
            FeatureType.SCALAR_TIMELESS,
            FeatureType.LABEL_TIMELESS,
        ]

    def is_image(self) -> bool:
        """True if FeatureType stores a dictionary with arrays that represent images. False otherwise."""
        return self.is_array() and self.is_spatial()

    @deprecated(
        "The method `is_raster` has been deprecated. Use the equivalent `is_array` method, or consider if `is_image`"
        " fits better.",
        category=EODeprecationWarning,
    )
    def is_raster(self) -> bool:
        """True if FeatureType stores a dictionary with raster data. False otherwise."""
        return self.is_array()

    @deprecated("The method `has_dict` has been deprecated.", category=EODeprecationWarning)
    def has_dict(self) -> bool:
        """True if FeatureType stores a dictionary. False otherwise."""
        return self in [
            FeatureType.DATA,
            FeatureType.MASK,
            FeatureType.SCALAR,
            FeatureType.LABEL,
            FeatureType.VECTOR,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK_TIMELESS,
            FeatureType.SCALAR_TIMELESS,
            FeatureType.LABEL_TIMELESS,
            FeatureType.VECTOR_TIMELESS,
            FeatureType.META_INFO,
        ]

    @deprecated("The method `contains_ndarrays` has been deprecated.", category=EODeprecationWarning)
    def contains_ndarrays(self) -> bool:
        """True if FeatureType stores a dictionary of numpy.ndarrays. False otherwise."""
        return self.is_array()

    def ndim(self) -> int | None:
        """If given FeatureType stores a dictionary of numpy.ndarrays it returns dimensions of such arrays."""
        if self.is_array():
            return {
                FeatureType.DATA: 4,
                FeatureType.MASK: 4,
                FeatureType.SCALAR: 2,
                FeatureType.LABEL: 2,
                FeatureType.DATA_TIMELESS: 3,
                FeatureType.MASK_TIMELESS: 3,
                FeatureType.SCALAR_TIMELESS: 1,
                FeatureType.LABEL_TIMELESS: 1,
            }[self]
        return None

    @deprecated("The method `type` has been deprecated.", category=EODeprecationWarning)
    def type(self) -> type:
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMPS:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

    @deprecated("The method `file_format` has been deprecated.", category=EODeprecationWarning)
    def file_format(self) -> MimeType:
        """Returns a mime type enum of a file format into which data of the feature type will be serialized"""
        if self.is_array():
            return MimeType.NPY
        if self.is_vector():
            return MimeType.GPKG
        if self is FeatureType.BBOX:
            return MimeType.GEOJSON
        return MimeType.JSON


def _warn_and_adjust_permissions(name: T) -> T:
    if isinstance(name, str) and name.upper() == "OVERWRITE_PATCH":
        warnings.warn(
            '"OVERWRITE_PATCH" permission is deprecated and will be removed in a future version',
            category=EODeprecationWarning,
            stacklevel=3,
        )
    return name


class PermissionsWithDeprecations(EnumMeta):
    """A custom EnumMeta class for catching the deprecated Enum members of the OverwritePermission Enum class."""

    def __getattribute__(cls, name: str) -> Any:  # noqa: N805
        return super().__getattribute__(_warn_and_adjust_permissions(name))

    def __getitem__(cls, name: str) -> Any:  # noqa: N805
        return super().__getitem__(_warn_and_adjust_permissions(name))

    def __call__(cls, value: str, *args: Any, **kwargs: Any) -> Any:  # noqa: N805
        return super().__call__(_warn_and_adjust_permissions(value), *args, **kwargs)


class OverwritePermission(Enum, metaclass=PermissionsWithDeprecations):
    """Enum class which specifies which content of the saved EOPatch can be overwritten when saving new content.

    Permissions are in the following hierarchy:

    - `ADD_ONLY` - Only new features can be added, anything that is already saved cannot be changed.
    - `OVERWRITE_FEATURES` - Overwrite only data for features which have to be saved. The remaining content of saved
      EOPatch will stay unchanged.
    """

    ADD_ONLY = "ADD_ONLY"
    OVERWRITE_FEATURES = "OVERWRITE_FEATURES"
    OVERWRITE_PATCH = "OVERWRITE_PATCH"

    @classmethod
    def _missing_(cls, value: object) -> OverwritePermission:
        permissions_mapping = {0: "ADD_ONLY", 1: "OVERWRITE_FEATURES", 2: "OVERWRITE_PATCH"}
        if isinstance(value, int) and value in permissions_mapping:
            deprecation_msg = (
                f"Please use strings to instantiate overwrite permissions, e.g., instead of {value} use"
                f" {permissions_mapping[value]!r}"
            )
            warnings.warn(deprecation_msg, category=EODeprecationWarning, stacklevel=3)

            return cls(permissions_mapping[value])
        if isinstance(value, str) and value.upper() in cls._value2member_map_:
            return cls(value.upper())
        return super()._missing_(value)
