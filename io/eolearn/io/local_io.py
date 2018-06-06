"""
Module containing tasks used for reading and writing to disk
"""

import os.path
import rasterio
import numpy as np

from sentinelhub import CRS, make_folder

from eolearn.core import EOPatch, EOTask


class SaveToDisk(EOTask):
    """ Saves EOPatch to disk.

    :param folder: root directory where all EOPatches are saved
    :type folder: str
    """

    def __init__(self, folder):
        self.folder = folder.rstrip('/')
        make_folder(folder)

    def execute(self, eopatch, *, eopatch_folder):
        """ Saves the EOPatch to disk: `folder/eopatch_folder`.

        :param eopatch: EOPatch which will be saved
        :type eopatch: eolearn.core.EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        :return: The same EOPatch
        :rtype: eolearn.core.EOPatch
        """
        eopatch.save(os.path.join(self.folder, eopatch_folder))
        return eopatch


class LoadFromDisk(EOTask):
    """ Loads EOPatch from disk.

    :param folder: root directory where all EOPatches are saved
    :type folder: str
    """
    def __init__(self, folder):
        self.folder = folder.rstrip('/')

    def execute(self, *, eopatch_folder):
        """ Loads the EOPatch from disk: `folder/eopatch_folder`.

        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        :return: EOPatch loaded from disk
        :rtype: eolearn.core.EOPatch
        """
        eopatch = EOPatch.load(os.path.join(self.folder, eopatch_folder))
        return eopatch


class ExportToTiff(EOTask):
    """ Task exports specified feature to Geo-Tiff.

    :param feature_type: Type of the raster feature which will be exported
    :type feature_type: eolearn.core.FeatureType
    :param feature_name: Name of the raster feature which will be exported
    :type feature_name: str
    :param band_count: Number of bands to be added to tiff image
    :type band_count: int
    :param image_dtype: Type of data to be saved into tiff image
    :type image_dtype: numpy.dtype
    :param no_data_value: Value of pixels of tiff image with no data in EOPatch
    :type no_data_value: int or float
    """

    def __init__(self, feature_type, feature_name, *, band_count=1, image_dtype=np.uint8, no_data_value=0):
        self.feature_type = feature_type
        self.feature_name = feature_name
        self.band_count = band_count
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value

    def execute(self, eopatch, *, filename):
        array = eopatch.get_feature(self.feature_type, self.feature_name)

        if self.band_count == 1:
            array = array[..., 0]

        dst_shape = array.shape
        dst_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=dst_shape[1], height=dst_shape[0])
        dst_crs = {'init': CRS.ogc_string(eopatch.bbox.crs)}

        # Write it out to a file.
        with rasterio.open(filename, 'w', driver='GTiff',
                           width=dst_shape[1], height=dst_shape[0],
                           count=self.band_count, dtype=self.image_dtype, nodata=self.no_data_value,
                           transform=dst_transform, crs=dst_crs) as dst:
            dst.write(array.astype(self.image_dtype), indexes=self.band_count)

        return eopatch
