"""
The following objects and functions are the core of eo-learn package
"""

from sentinelhub import BBox, CRS

from .constants import FeatureType, FeatureTypeSet, FileFormat, OverwritePermission
from .eodata import EOPatch
from .eotask import EOTask, CompositeTask
from .eoworkflow import EOWorkflow, LinearWorkflow, Dependency, WorkflowResults
from .eoexecution import EOExecutor

from .core_tasks import CopyTask, DeepCopyTask, SaveToDisk, LoadFromDisk, AddFeature, RemoveFeature, RenameFeature
from .utilities import deep_eq, negate_mask, constant_pad, get_common_timestamps, bgr_to_rgb, FeatureParser


__version__ = '0.5.2'
