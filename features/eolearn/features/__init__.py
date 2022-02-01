"""
A collection of EOTasks for feature manipulation
"""

from .temporal_features import (
    AddSpatioTemporalFeaturesTask,
    AddMaxMinTemporalIndicesTask,
    AddMaxMinNDVISlopeIndicesTask,
)
from .interpolation import (
    InterpolationTask,
    ResamplingTask,
    LinearInterpolationTask,
    CubicInterpolationTask,
    SplineInterpolationTask,
    BSplineInterpolationTask,
    AkimaInterpolationTask,
    NearestResamplingTask,
    LinearResamplingTask,
    CubicResamplingTask,
    KrigingInterpolationTask,
)
from .feature_manipulation import (
    SimpleFilterTask,
    FilterTimeSeriesTask,
    ValueFilloutTask,
    LinearFunctionTask,
)
from .haralick import HaralickTask
from .radiometric_normalization import (
    ReferenceScenesTask,
    HistogramMatchingTask,
    BlueCompositingTask,
    HOTCompositingTask,
    MaxNDVICompositingTask,
    MaxNDWICompositingTask,
    MaxRatioCompositingTask,
)
from .blob import BlobTask, DoGBlobTask, DoHBlobTask, LoGBlobTask
from .hog import HOGTask
from .local_binary_pattern import LocalBinaryPatternTask
from .bands_extraction import EuclideanNormTask, NormalizedDifferenceIndexTask
from .clustering import ClusteringTask
from .doubly_logistic_approximation import DoublyLogisticApproximationTask


__version__ = "1.0.0"
