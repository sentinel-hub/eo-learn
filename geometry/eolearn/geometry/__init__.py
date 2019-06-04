"""
Subpackage containing EOTasks for geometrical transformations
"""

from .utilities import ErosionTask, VectorToRaster, RasterToVector
from .sampling import PointSamplingTask, PointSampler, PointRasterSampler
from .superpixel import SuperpixelSegmentation, FelzenszwalbSegmentation, SlicSegmentation, MarkSegmentationBoundaries

__version__ = '0.4.2'
