"""
Module containing tasks used for reading and writing to disk
"""

import os.path
import rasterio
import numpy as np

from sentinelhub import CRS

from eolearn.core import SaveToDisk


class ExportToTiff(SaveToDisk):
    """ Task exports specified feature to Geo-Tiff.

    :param feature: Feature which will be exported
    :type feature: (FeatureType, str)
    :param folder: root directory where all Geo-Tiff images will be saved
    :type folder: str
    :param band_count: Number of bands to be added to tiff image
    :type band_count: int
    :param image_dtype: Type of data to be saved into tiff image
    :type image_dtype: numpy.dtype
    :param no_data_value: Value of pixels of tiff image with no data in EOPatch
    :type no_data_value: int or float
    """

    def __init__(self, feature, folder='.', *, band_count=1, image_dtype=np.uint8, no_data_value=0):
        super().__init__(folder)

        self.feature = self._parse_features(feature)
        self.band_count = band_count
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value

    def execute(self, eopatch, *, filename):

        feature_type, feature_name = next(self.feature(eopatch))
        array = eopatch[feature_type][feature_name]

        if self.band_count == 1:
            array = array[..., 0]

        dst_shape = array.shape
        dst_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=dst_shape[1], height=dst_shape[0])
        dst_crs = {'init': CRS.ogc_string(eopatch.bbox.crs)}

        # Write it out to a file.
        with rasterio.open(os.path.join(self.folder, filename), 'w', driver='GTiff',
                           width=dst_shape[1], height=dst_shape[0],
                           count=self.band_count, dtype=self.image_dtype, nodata=self.no_data_value,
                           transform=dst_transform, crs=dst_crs) as dst:
            dst.write(array.astype(self.image_dtype), indexes=self.band_count)

        return eopatch
