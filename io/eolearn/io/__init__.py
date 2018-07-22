"""
A collection of input and output EOTasks
"""

from .sh_input import SentinelHubOGCInput, SentinelHubWMSInput, SentinelHubWCSInput, S2L1CWMSInput, S2L1CWCSInput,\
    L8L1CWMSInput, L8L1CWCSInput, S2L2AWMSInput, S2L2AWCSInput
from .sh_add import AddSentinelHubOGCFeature, AddDEMFeature, AddS2L1CFeature, AddS2L2AFeature, AddL8Feature, \
    AddSen2CorClassificationFeature, AddGeopediaFeature
from .local_io import SaveToDisk, LoadFromDisk, ExportToTiff

__version__ = '0.2.0'
