"""
Subpackage containing EOTasks for geometrical transformations
"""

from .morphology import ErosionTask
from .superpixel import (
    SuperpixelSegmentationTask,
    FelzenszwalbSegmentationTask,
    SlicSegmentationTask,
    MarkSegmentationBoundariesTask,
)
from .transformations import VectorToRasterTask, RasterToVectorTask

__version__ = "1.0.0"
