"""
The following objects and functions are the core of eo-learn package
"""
from .constants import FeatureType, FeatureTypeSet, OverwritePermission
from .eodata import EOPatch
from .eotask import EOTask
from .eonode import EONode, linearly_connect_tasks
from .eoworkflow import EOWorkflow, WorkflowResults
from .eoworkflow_tasks import OutputTask
from .eoexecution import EOExecutor, execute_with_mp_lock

from .core_tasks import (
    CopyTask,
    DeepCopyTask,
    SaveTask,
    LoadTask,
    AddFeatureTask,
    RemoveFeatureTask,
    RenameFeatureTask,
    DuplicateFeatureTask,
    InitializeFeatureTask,
    MoveFeatureTask,
    MergeFeatureTask,
    MapFeatureTask,
    ZipFeatureTask,
    ExtractBandsTask,
    CreateEOPatchTask,
    MergeEOPatchesTask,
)

from .utils.fs import get_filesystem, load_s3_filesystem
from .utils.parsing import FeatureParser
from .utils.common import deep_eq, constant_pad

__version__ = "1.0.0"
