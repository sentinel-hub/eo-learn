"""
A collection of EOTasks for feature manipulation
"""

from .temporal_features import AddSpatioTemporalFeaturesTask, AddMaxMinTemporalIndicesTask, \
    AddMaxMinNDVISlopeIndicesTask
from .interpolation import InterpolationTask, LinearInterpolation, CubicInterpolation, SplineInterpolation, \
    BSplineInterpolation, AkimaInterpolation, ResamplingTask, NearestResampling, LinearResampling, CubicResampling
from .feature_extractor import FeatureExtractionTask, FeatureExtendedExtractor
from .feature_manipulation import SimpleFilterTask, FilterTimeSeries
from .haralick import HaralickTask


__version__ = '0.3.1'
