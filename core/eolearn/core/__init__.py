"""
The following objects and functions are the core of eo-learn package
"""

from sentinelhub import BBox, CRS

from .constants import FeatureType, FileFormat, OverwritePermission
from .eodata import EOPatch
from .eotask import EOTask, CompositeTask
from .eoworkflow import EOWorkflow, LinearWorkflow, Dependency, WorkflowResults
from .eoexecution import EOExecutor

from .core_tasks import CopyTask, DeepCopyTask, SaveToDisk, LoadFromDisk, AddFeature, RemoveFeature, RenameFeature
from .plots import bgr_to_rgb, IndexTracker, PatchShowTask
from .utilities import deep_eq, negate_mask, constant_pad, get_common_timestamps


__version__ = '0.4.0'
