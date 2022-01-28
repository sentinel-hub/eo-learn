"""
Subpackage containing EOTasks for geometrical transformations
"""

from .morphology import ErosionTask
from .superpixel import (
    SuperpixelSegmentationTask,
    FelzenszwalbSegmentationTask,
    SlicSegmentationTask,
    MarkSegmentationBoundariesTask,
    SuperpixelSegmentation,
    FelzenszwalbSegmentation,
    SlicSegmentation,
    MarkSegmentationBoundaries,
)
from .transformations import VectorToRasterTask, RasterToVectorTask, VectorToRaster, RasterToVector

__version__ = "1.0.0"
