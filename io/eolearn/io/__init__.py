"""
A collection of input and output EOTasks
"""

from .geopedia import AddGeopediaFeature, AddGeopediaFeatureTask
from .local_io import ExportToTiff, ImportFromTiff, ExportToTiffTask, ImportFromTiffTask
from .geometry_io import VectorImportTask, GeopediaVectorImportTask, GeoDBVectorImportTask
from .sentinelhub_process import SentinelHubDemTask, SentinelHubEvalscriptTask, SentinelHubInputTask, \
    SentinelHubSen2corTask, get_available_timestamps
from .meteoblue import MeteoblueVectorTask, MeteoblueRasterTask


__version__ = '0.10.1'
