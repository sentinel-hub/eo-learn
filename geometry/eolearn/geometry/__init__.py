"""
Subpackage containing EOTasks for geometrical transformations
"""

from .morphology import ErosionTask, MorphologicalFilterTask, MorphologicalOperations, MorphologicalStructFactory
from .superpixel import (
    FelzenszwalbSegmentationTask,
    MarkSegmentationBoundariesTask,
    SlicSegmentationTask,
    SuperpixelSegmentationTask,
)
from .transformations import RasterToVectorTask, VectorToRasterTask

__version__ = "1.4.0"
