"""
The following objects and functions are the core of eo-learn package
"""

from .eodata import FeatureType, EOPatch
from .eotask import EOTask, EOChainedTask
from .eoworkflow import EOWorkflow, Dependency, WorkflowResult
from .graph import DirectedGraph
from .plots import bgr_to_rgb, IndexTracker, PatchShowTask
from .utilities import deep_eq, negate_mask, constant_pad


__version__ = '0.1.0'
