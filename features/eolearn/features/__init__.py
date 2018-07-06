"""
A collection of EOTasks for feature manipulation
"""

from .temporal_features import AddSpatioTemporalFeaturesTask, AddMaxMinTemporalIndicesTask, \
    AddMaxMinNDVISlopeIndicesTask
from .interp_smooth import BSplineInterpolation
from .feature_extractor import FeatureExtractionTask, FeatureExtendedExtractor
from .feature_manipulation import RemoveFeature, SimpleFilterTask
from .compute_haralick import AddHaralickTask
from .feature_manipulation import RemoveFeature, SimpleFilterTask, FilterTimeSeries


__version__ = '0.1.0'
1