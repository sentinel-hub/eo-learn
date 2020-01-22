"""
A collection of input and output EOTasks
"""

from .sentinelhub_service import SentinelHubOGCInput, SentinelHubWMSInput, SentinelHubWCSInput, S2L1CWMSInput, \
    S2L1CWCSInput, L8L1CWMSInput, L8L1CWCSInput, S2L2AWMSInput, S2L2AWCSInput, S1IWWMSInput, S1IWWCSInput, \
    DEMWMSInput, DEMWCSInput, AddSen2CorClassificationFeature
from .geopedia import AddGeopediaFeature
from .local_io import ExportToTiff, ImportFromTiff
from .processing_api import SentinelHubInputTask, SentinelHubDemTask

__version__ = '0.7.0'
