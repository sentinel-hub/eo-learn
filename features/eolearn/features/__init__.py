"""
A collection of EOTasks for feature manipulation
"""

from .temporal_features import AddSpatioTemporalFeaturesTask, AddMaxMinTemporalIndicesTask, \
    AddMaxMinNDVISlopeIndicesTask
from .interpolation import InterpolationTask, LinearInterpolation, CubicInterpolation, SplineInterpolation, \
    BSplineInterpolation, AkimaInterpolation, ResamplingTask, NearestResampling, LinearResampling, CubicResampling, \
    KrigingInterpolation, LegacyInterpolation
from .feature_extractor import FeatureExtractionTask, FeatureExtendedExtractor
from .feature_manipulation import SimpleFilterTask, FilterTimeSeries, ValueFilloutTask
from .haralick import HaralickTask
from .radiometric_normalization import ReferenceScenes, HistogramMatching, BlueCompositing, HOTCompositing, \
    MaxNDVICompositing, MaxNDWICompositing, MaxRatioCompositing
from .blob import BlobTask, DoGBlobTask, DoHBlobTask, LoGBlobTask
from .hog import HOGTask
from .local_binary_pattern import LocalBinaryPatternTask
from .bands_extraction import EuclideanNormTask, NormalizedDifferenceIndexTask


__version__ = '0.7.0'
