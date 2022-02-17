"""
This module implements feature types used in EOPatch objects

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from enum import Enum
from typing import Optional

from sentinelhub import BBox, MimeType


class FeatureType(Enum):
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
     - TIMESTAMP: list of dates which are instances of datetime.datetime
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
    TIMESTAMP = "timestamp"

    @classmethod
    def has_value(cls, value: str) -> bool:
        """True if value is in FeatureType values. False otherwise."""
        return value in cls._value2member_map_

    def is_spatial(self) -> bool:
        """True if FeatureType has a spatial component. False otherwise."""
        return self in FeatureTypeSet.SPATIAL_TYPES

    def is_temporal(self) -> bool:
        """True if FeatureType has a time component. False otherwise."""
        return self in FeatureTypeSet.TEMPORAL_TYPES

    def is_timeless(self) -> bool:
        """True if FeatureType doesn't have a time component and is not a meta feature. False otherwise."""
        return self in FeatureTypeSet.TIMELESS_TYPES

    def is_discrete(self) -> bool:
        """True if FeatureType should have discrete (integer) values. False otherwise."""
        return self in FeatureTypeSet.DISCRETE_TYPES

    def is_meta(self) -> bool:
        """True if FeatureType is for storing metadata info and False otherwise."""
        return self in FeatureTypeSet.META_TYPES

    def is_vector(self) -> bool:
        """True if FeatureType is vector feature type. False otherwise."""
        return self in FeatureTypeSet.VECTOR_TYPES

    def has_dict(self) -> bool:
        """True if FeatureType stores a dictionary. False otherwise."""
        return self in FeatureTypeSet.DICT_TYPES

    def is_raster(self) -> bool:
        """True if FeatureType stores a dictionary with raster data. False otherwise."""
        return self in FeatureTypeSet.RASTER_TYPES

    def contains_ndarrays(self) -> bool:
        """True if FeatureType stores a dictionary of numpy.ndarrays. False otherwise."""
        return self in FeatureTypeSet.RASTER_TYPES

    def ndim(self) -> Optional[int]:
        """If given FeatureType stores a dictionary of numpy.ndarrays it returns dimensions of such arrays."""
        if self.is_raster():
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

    def type(self) -> type:
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

    def file_format(self) -> MimeType:
        """Returns a mime type enum of a file format into which data of the feature type will be serialized"""
        if self.is_raster():
            return MimeType.NPY
        if self.is_vector():
            return MimeType.GPKG
        if self is FeatureType.BBOX:
            return MimeType.GEOJSON
        return MimeType.JSON


class FeatureTypeSet:
    """A collection of immutable sets of feature types, grouped together by certain properties."""

    SPATIAL_TYPES = frozenset(
        [
            FeatureType.DATA,
            FeatureType.MASK,
            FeatureType.VECTOR,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK_TIMELESS,
            FeatureType.VECTOR_TIMELESS,
        ]
    )
    TEMPORAL_TYPES = frozenset(
        [
            FeatureType.DATA,
            FeatureType.MASK,
            FeatureType.SCALAR,
            FeatureType.LABEL,
            FeatureType.VECTOR,
            FeatureType.TIMESTAMP,
        ]
    )
    TIMELESS_TYPES = frozenset(
        [
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK_TIMELESS,
            FeatureType.SCALAR_TIMELESS,
            FeatureType.LABEL_TIMELESS,
            FeatureType.VECTOR_TIMELESS,
        ]
    )
    DISCRETE_TYPES = frozenset(
        [FeatureType.MASK, FeatureType.MASK_TIMELESS, FeatureType.LABEL, FeatureType.LABEL_TIMELESS]
    )
    META_TYPES = frozenset([FeatureType.META_INFO, FeatureType.BBOX, FeatureType.TIMESTAMP])
    VECTOR_TYPES = frozenset([FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS])
    RASTER_TYPES = frozenset(
        [
            FeatureType.DATA,
            FeatureType.MASK,
            FeatureType.SCALAR,
            FeatureType.LABEL,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK_TIMELESS,
            FeatureType.SCALAR_TIMELESS,
            FeatureType.LABEL_TIMELESS,
        ]
    )
    DICT_TYPES = frozenset(
        [
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
    )
    RASTER_TYPES_4D = frozenset([FeatureType.DATA, FeatureType.MASK])
    RASTER_TYPES_3D = frozenset([FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS])
    RASTER_TYPES_2D = frozenset([FeatureType.SCALAR, FeatureType.LABEL])
    RASTER_TYPES_1D = frozenset([FeatureType.SCALAR_TIMELESS, FeatureType.LABEL_TIMELESS])


class OverwritePermission(Enum):
    """Enum class which specifies which content of saved EOPatch can be overwritten when saving new content.

    Permissions are in the following hierarchy:

    - `ADD_ONLY` - Only new features can be added, anything that is already saved cannot be changed.
    - `OVERWRITE_FEATURES` - Overwrite only data for features which have to be saved. The remaining content of saved
      EOPatch will stay unchanged.
    - `OVERWRITE_PATCH` - Overwrite entire content of saved EOPatch and replace it with the new content.
    """

    ADD_ONLY = 0
    OVERWRITE_FEATURES = 1
    OVERWRITE_PATCH = 2
