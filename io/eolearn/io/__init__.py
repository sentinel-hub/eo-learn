"""
A collection of input and output EOTasks
"""

from .geopedia import AddGeopediaFeatureTask
from .local_io import ExportToTiffTask, ImportFromTiffTask
from .geometry_io import VectorImportTask, GeopediaVectorImportTask
from .sentinelhub_process import (
    SentinelHubDemTask,
    SentinelHubEvalscriptTask,
    SentinelHubInputTask,
    SentinelHubSen2corTask,
    get_available_timestamps,
)

__version__ = "1.0.0"
