"""
The following objects and functions are the core of eo-learn package
"""

from sentinelhub import BBox, CRS

from .constants import FeatureType, FeatureTypeSet, FileFormat, OverwritePermission
from .eodata import EOPatch
from .eotask import EOTask, CompositeTask
from .eoworkflow import EOWorkflow, LinearWorkflow, Dependency, WorkflowResults
from .eoexecution import EOExecutor, execute_with_mp_lock

from .core_tasks import CopyTask, DeepCopyTask, SaveTask, LoadTask, AddFeature, RemoveFeature, RenameFeature,\
    DuplicateFeature, InitializeFeature, MoveFeature, MergeFeatureTask, MapFeatureTask, ZipFeatureTask,\
    ExtractBandsTask, CreateEOPatchTask, SaveToDisk, LoadFromDisk

from .fs_utils import get_filesystem, load_s3_filesystem
from .utilities import deep_eq, negate_mask, constant_pad, get_common_timestamps, bgr_to_rgb, FeatureParser


__version__ = '0.7.0'
