"""
A collection of EOTasks for feature manipulation
"""

from .temporal_features import (
    AddSpatioTemporalFeaturesTask, AddMaxMinTemporalIndicesTask, AddMaxMinNDVISlopeIndicesTask
)
from .interpolation import (
    InterpolationTask, LinearInterpolation, CubicInterpolation, SplineInterpolation, BSplineInterpolation,
    AkimaInterpolation, ResamplingTask, NearestResampling, LinearResampling, CubicResampling, KrigingInterpolation,
    LegacyInterpolation, LinearInterpolationTask, CubicInterpolationTask, SplineInterpolationTask,
    BSplineInterpolationTask, AkimaInterpolationTask, NearestResamplingTask, LinearResamplingTask, CubicResamplingTask,
    KrigingInterpolationTask, LegacyInterpolationTask
)
from .feature_extractor import FeatureExtractionTask, FeatureExtendedExtractor
from .feature_manipulation import SimpleFilterTask, FilterTimeSeriesTask, FilterTimeSeries, ValueFilloutTask
from .haralick import HaralickTask
from .radiometric_normalization import (
    ReferenceScenes, HistogramMatching, BlueCompositing, HOTCompositing, MaxNDVICompositing, MaxNDWICompositing,
    MaxRatioCompositing, ReferenceScenesTask, HistogramMatchingTask, BlueCompositingTask, HOTCompositingTask,
    MaxNDVICompositingTask, MaxNDWICompositingTask, MaxRatioCompositingTask
)
from .blob import BlobTask, DoGBlobTask, DoHBlobTask, LoGBlobTask
from .hog import HOGTask
from .local_binary_pattern import LocalBinaryPatternTask
from .bands_extraction import EuclideanNormTask, NormalizedDifferenceIndexTask
from .clustering import ClusteringTask
from .doubly_logistic_approximation import DoublyLogisticApproximationTask


__version__ = '0.10.1'
