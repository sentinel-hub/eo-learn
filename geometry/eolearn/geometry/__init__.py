"""
Subpackage containing EOTasks for geometrical transformations
"""

from .morphology import ErosionTask
from .sampling import PointSamplingTask, PointSampler, PointRasterSampler
from .superpixel import SuperpixelSegmentation, FelzenszwalbSegmentation, SlicSegmentation, MarkSegmentationBoundaries
from .transformations import VectorToRaster, RasterToVector

__version__ = '0.7.0'
