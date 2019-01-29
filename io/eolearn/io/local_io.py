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
    :param band_count: Bands to be added to tiff image. Bands are represented by an integer `band_n`, 
    a tuple in the form `(start_band, end_band)` or a list in the form `[band_1, band_2,...,band_n]`.
    :type band_count: int, tuple or list
    :param time_count: Dates to be added to tiff image. Dates are represented by an integer `date_n`, 
    a tuple in the form `(start_date, end_date)` or a list in the form `[date_1, date_2,...,date_n]`.
    :type time_count: int, tuple or list
    :param image_dtype: Type of data to be saved into tiff image
    :type image_dtype: numpy.dtype
    :param no_data_value: Value of pixels of tiff image with no data in EOPatch
    :type no_data_value: int or float
    """

    def __init__(self, feature, folder='.', *, band_count=1, time_count=1, image_dtype=np.uint8, no_data_value=0):
        super().__init__(folder)

        self.feature = self._parse_features(feature)
        self.band_count = band_count
        self.time_count = time_count
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value

    def execute(self, eopatch, *, filename):

        feature_type, feature_name = next(self.feature(eopatch))
        array = eopatch[feature_type][feature_name]
        dates = eopatch.timestamp
        bands = range(array.shape[-1])

        filename_list = []
        
        if type(self.band_count) is list: 
            if [date for date in self.band_count if type(elem) != int]:
                raise ValueError('Invalid format in {} list, expected integers'.format(self.band_count))
            array_sub = array[...,np.array(self.band_count)-1]
        elif type(self.band_count) is tuple:
            if tuple(map(type, self.band_count)) != (int, int):
                raise ValueError('Invalid format in {} tuple, expected integers'.format(self.band_count))
            array_sub = array[...,np.nonzero(np.where(bands >= self.band_count[0] and bands <= self.band_count[1],bands,0))]
        elif type(self.band_count) == int:
            array_sub = array[...,self.band_count-1]
        else:
            raise ValueError('Invalid format in {}, expected int, tuple or list'.format(self.band_count))
            
        band_dim = len(array_sub.shape[-1])

        if feature_type in [FeatureType.DATA, FeatureType.MASK]:
            if type(self.time_count) == list:
                if [date for date in self.time_count if type(elem) != int]:
                    raise ValueError('Invalid format in {} list, expected integers'.format(self.time_count))
                array_sub = array_sub[np.array(self.time_count)-1,...]
            elif type(self.time_count) == tuple:
                if tuple(map(type, self.time_count)) == (str, str):
                    start_date = parser.parse(self.time_count[0])
                    end_date = parser.parse(self.time_count[1])
                elif tuple(map(type, self.time_count)) == (datetime, datetime):
                    start_date = self.time_count[0]
                    end_date = self.time_count[1]
                else:
                    raise ValueError('Invalid format in {} tuple, expected datetimes or strings'.format(self.time_count))
                array_sub = array_sub[np.nonzero(np.where(dates >= start_date and dates <= end_date, dates, 0)),...]
            elif type(self.time_count) == int:
                array_sub = array_sub[self.time_count-1,...]
            else:
                raise ValueError('Invalid format in {}, expected int, tuple or list'.format(self.time_count))
                
            time_dim = len(array_sub.shape[0])
            width = array_sub.shape[2]
            height = array_sub.shape[1]
        else:
            time_dim = 1
            width = array_sub.shape[1]
            height = array_sub.shape[0]

        index = time_dim * band_dim
        dst_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)
        dst_crs = {'init': CRS.ogc_string(eopatch.bbox.crs)}

        # Write it out to a file.
        with rasterio.open(os.path.join(self.folder, filename), 'w', driver='GTiff',
                           width=width, height=height,
                           count=index,
                           dtype=self.image_dtype, nodata=self.no_data_value,
                           transform=dst_transform, crs=dst_crs) as dst:
            dst.write(array.astype(self.image_dtype).reshape(index, width, height).squeeze(), indexes=index)

        return eopatch
