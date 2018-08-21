"""
This module implements feature types used in EOPatch objects
"""

from enum import Enum


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
    DATA = 'data'
    MASK = 'mask'
    SCALAR = 'scalar'
    LABEL = 'label'
    VECTOR = 'vector'
    DATA_TIMELESS = 'data_timeless'
    MASK_TIMELESS = 'mask_timeless'
    SCALAR_TIMELESS = 'scalar_timeless'
    LABEL_TIMELESS = 'label_timeless'
    VECTOR_TIMELESS = 'vector_timeless'
    META_INFO = 'meta_info'
    BBOX = 'bbox'
    TIMESTAMP = 'timestamp'

    @classmethod
    def has_value(cls, value):
        """True if value is in FeatureType values. False otherwise."""
        return any(value == item.value for item in cls)

    def is_spatial(self):
        """True if FeatureType has a spatial component. False otherwise."""
        return self in frozenset([FeatureType.DATA, FeatureType.MASK, FeatureType.VECTOR, FeatureType.DATA_TIMELESS,
                                  FeatureType.MASK_TIMELESS, FeatureType.VECTOR_TIMELESS])

    def is_time_dependent(self):
        """True if FeatureType has a time component. False otherwise."""
        return self in frozenset([FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL,
                                  FeatureType.VECTOR, FeatureType.TIMESTAMP])

    def is_discrete(self):
        """True if FeatureType should have discrete (integer) values. False otherwise."""
        return self in frozenset([FeatureType.MASK, FeatureType.MASK_TIMELESS, FeatureType.LABEL,
                                  FeatureType.LABEL_TIMELESS])

    def is_vector(self):
        """True if FeatureType is vector feature type. False otherwise. """
        return self in frozenset([FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS])

    def has_dict(self):
        """True if FeatureType stores a dictionary. False otherwise."""
        return self not in frozenset([FeatureType.TIMESTAMP, FeatureType.BBOX])

    def contains_ndarrays(self):
        """True if FeatureType stores a dictionary of numpy.ndarrays. False otherwise."""
        return self in frozenset([FeatureType.DATA, FeatureType.MASK, FeatureType.SCALAR, FeatureType.LABEL,
                                  FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS, FeatureType.SCALAR_TIMELESS,
                                  FeatureType.LABEL_TIMELESS])

    def ndim(self):
        """If given FeatureType stores a dictionary of numpy.ndarrays it returns dimensions of such arrays."""
        if self.contains_ndarrays():
            return {
                FeatureType.DATA: 4,
                FeatureType.MASK: 4,
                FeatureType.SCALAR: 2,
                FeatureType.LABEL: 2,
                FeatureType.DATA_TIMELESS: 3,
                FeatureType.MASK_TIMELESS: 3,
                FeatureType.SCALAR_TIMELESS: 1,
                FeatureType.LABEL_TIMELESS: 1
            }[self]
        return None

    def type(self):
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return object
        return dict
