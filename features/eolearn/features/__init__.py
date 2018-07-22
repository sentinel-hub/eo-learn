"""
A collection of EOTasks for feature manipulation
"""

from .temporal_features import AddSpatioTemporalFeaturesTask, AddMaxMinTemporalIndicesTask, \
    AddMaxMinNDVISlopeIndicesTask
from .interpolation import InterpolationTask, LinearInterpolation, CubicInterpolation, SplineInterpolation, \
    BSplineInterpolation, AkimaInterpolation, ResamplingTask, LinearResampling, CubicResampling
from .feature_extractor import FeatureExtractionTask, FeatureExtendedExtractor
from .feature_manipulation import SimpleFilterTask, FilterTimeSeries
from .compute_haralick import AddHaralickTask


__version__ = '0.2.0'
