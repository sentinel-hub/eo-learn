"""
The following objects and functions are the core of eo-learn package
"""
from .constants import FeatureType, FeatureTypeSet, OverwritePermission
from .core_tasks import (
    AddFeatureTask,
    CopyTask,
    CreateEOPatchTask,
    DeepCopyTask,
    DuplicateFeatureTask,
    ExplodeBandsTask,
    ExtractBandsTask,
    InitializeFeatureTask,
    LoadTask,
    MapFeatureTask,
    MergeEOPatchesTask,
    MergeFeatureTask,
    MoveFeatureTask,
    RemoveFeatureTask,
    RenameFeatureTask,
    SaveTask,
    ZipFeatureTask,
)
from .eodata import EOPatch
from .eoexecution import EOExecutor
from .eonode import EONode, linearly_connect_tasks
from .eotask import EOTask
from .eoworkflow import EOWorkflow, WorkflowResults
from .eoworkflow_tasks import OutputTask
from .utils.common import deep_eq
from .utils.fs import get_filesystem, load_s3_filesystem, pickle_fs, unpickle_fs
from .utils.parallelize import execute_with_mp_lock, join_futures, join_futures_iter, parallelize
from .utils.parsing import FeatureParser

__version__ = "1.4.0"
