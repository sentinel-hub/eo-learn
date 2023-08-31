"""
A collection of EOTasks for feature manipulation
"""

from .bands_extraction import EuclideanNormTask, NormalizedDifferenceIndexTask
from .blob import BlobTask, DoGBlobTask, DoHBlobTask, LoGBlobTask
from .doubly_logistic_approximation import DoublyLogisticApproximationTask
from .feature_manipulation import FilterTimeSeriesTask, LinearFunctionTask, SimpleFilterTask, ValueFilloutTask
from .haralick import HaralickTask
from .hog import HOGTask
from .local_binary_pattern import LocalBinaryPatternTask
from .radiometric_normalization import (
    BlueCompositingTask,
    HistogramMatchingTask,
    HOTCompositingTask,
    MaxNDVICompositingTask,
    MaxNDWICompositingTask,
    MaxRatioCompositingTask,
    ReferenceScenesTask,
)
from .temporal_features import (
    AddMaxMinNDVISlopeIndicesTask,
    AddMaxMinTemporalIndicesTask,
    AddSpatioTemporalFeaturesTask,
)
