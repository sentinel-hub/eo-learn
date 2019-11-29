"""
Module containing tasks used for reading and writing to disk

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)
Copyright (c) 2018-2019 William Ouellette (TomTom)
Copyright (c) 2019 Drew Bollinger (DevelopmentSeed)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import datetime
import logging
import warnings
from abc import abstractmethod

import dateutil
import rasterio
import numpy as np

from sentinelhub import CRS, BBox

from eolearn.core import EOTask, EOPatch

LOGGER = logging.getLogger(__name__)


class BaseLocalIo(EOTask):
    """ Base abstract class for local IO tasks
    """
    def __init__(self, feature, folder=None, *, image_dtype=None, no_data_value=0):
        """
        :param feature: Feature which will be exported or imported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a path of an image file
        :type folder: str
        :param image_dtype: Type of data to be exported into tiff image or imported from tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Value of undefined pixels
        :type no_data_value: int or float
        """
        self.feature = self._parse_features(feature)
        self.folder = folder
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value

    def _get_file_path(self, filename, create_dir=False):
        """ Builds a file path from values obtained at class initialization and in execute method.
        If create_dir is set to True, non existing directories are automatically created.
        """
        if self.folder is None:
            if filename is None:
                raise ValueError("At least one of parameters 'folder' and 'filename' has to be specified")
            path = filename
        elif filename is None:
            path = self.folder
        else:
            path = os.path.join(self.folder, filename)

        # Create directory of path if it doesn't exist
        if create_dir:
            path_dir = os.path.dirname(path)
            if path_dir != '' and not os.path.exists(path_dir):
                os.makedirs(path_dir, exist_ok=True)

        return path

    @abstractmethod
    def execute(self, eopatch, **kwargs):
        """ Execute of a base class is not implemented
        """
        raise NotImplementedError


class ExportToTiff(BaseLocalIo):
    """ Task exports specified feature to Geo-Tiff.

    When exporting multiple times OR bands, the Geo-Tiff `band` counts are in the expected order.
    However, when exporting multiple times AND bands, the order obeys the following pattern:

    T(1)B(1), T(1)B(2), ..., T(1)B(N), T(2)B(1), T(2)B(2), ..., T(2)B(N), ..., ..., T(M)B(N)

    where T and B are the time and band indices of the array,
    and M and N are the lengths of these indices, respectively
    """
    def __init__(self, feature, folder=None, *, band_indices=None, date_indices=None, crs=None, fail_on_missing=True,
                 **kwargs):
        """
        :param feature: Feature which will be exported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a path of an image file
        :type folder: str
        :param band_indices: Bands to be added to tiff image. Bands are represented by their 0-based index as tuple
            in the inclusive interval form `(start_band, end_band)` or as list in the form
            `[band_1, band_2,...,band_n]`.
        :type band_indices: tuple or list or None
        :param date_indices: Dates to be added to tiff image. Dates are represented by their 0-based index as tuple
            in the inclusive interval form `(start_date, end_date)` or a list in the form `[date_1, date_2,...,date_n]`.
        :type date_indices: tuple or list or None
        :param crs: CRS in which to reproject the feature before writing it to GeoTiff
        :type crs: CRS or string of the form authority:id representing the CRS
        :param fail_on_missing: should the pipeline fail if a feature is missing or just log warning and return
        :type fail_on_missing: bool
        :param image_dtype: Type of data to be exported into tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Value of pixels of tiff image with no data in EOPatch
        :type no_data_value: int or float
        """
        super().__init__(feature, folder=folder, **kwargs)

        self.band_indices = band_indices
        self.date_indices = date_indices
        self.crs = crs
        self.fail_on_missing = fail_on_missing

    def _prepare_image_array(self, eopatch, feature):
        """ Collects a feature from EOPatch and prepares the array of an image which will be rasterized. The resulting
        array has shape (channels, height, width) and is of correct dtype.
        """
        data_array = self._get_bands_subset(eopatch[feature])

        feature_type = feature[0]
        if feature_type.is_time_dependent():
            data_array = self._get_dates_subset(data_array, eopatch.timestamp)
        else:
            # add temporal dimension
            data_array = np.expand_dims(data_array, axis=0)

        if not feature_type.is_spatial():
            # add height and width dimensions
            data_array = np.expand_dims(np.expand_dims(data_array, axis=1), axis=1)

        data_array = self._set_export_dtype(data_array, feature)

        return self._reshape_to_image_array(data_array)

    def _get_bands_subset(self, array):
        """ Reduce array by selecting a subset of bands
        """
        if self.band_indices is None:
            return array
        if isinstance(self.band_indices, list):
            if [band for band in self.band_indices if not isinstance(band, int)]:
                raise ValueError('Invalid format in {} list, expected integers'.format(self.band_indices))
            return array[..., self.band_indices]
        if isinstance(self.band_indices, tuple):
            if tuple(map(type, self.band_indices)) != (int, int):
                raise ValueError('Invalid format in {} tuple, expected integers'.format(self.band_indices))
            return array[..., self.band_indices[0]: self.band_indices[1] + 1]

        raise ValueError('Invalid format in {}, expected tuple or list'.format(self.band_indices))

    def _get_dates_subset(self, array, dates):
        """ Reduce array by selecting a subset of times
        """
        if self.date_indices is None:
            return array
        if isinstance(self.date_indices, list):
            if [date for date in self.date_indices if not isinstance(date, int)]:
                raise ValueError('Invalid format in {} list, expected integers'.format(self.date_indices))
            return array[np.array(self.date_indices), ...]
        if isinstance(self.date_indices, tuple):
            dates = np.array(dates)
            if tuple(map(type, self.date_indices)) == (int, int):
                start_date = dates[self.date_indices[0]]
                end_date = dates[self.date_indices[1]]
            elif tuple(map(type, self.date_indices)) == (str, str):
                start_date = dateutil.parser.parse(self.date_indices[0])
                end_date = dateutil.parser.parse(self.date_indices[1])
            elif tuple(map(type, self.date_indices)) == (datetime.datetime, datetime.datetime):
                start_date = self.date_indices[0]
                end_date = self.date_indices[1]
            else:
                raise ValueError('Invalid format in {} tuple, expected ints, strings, or datetimes'.format(
                    self.date_indices))
            return array[np.nonzero(np.where((dates >= start_date) & (dates <= end_date), dates, 0))[0]]

        raise ValueError('Invalid format in {}, expected tuple or list'.format(self.date_indices))

    def _set_export_dtype(self, data_array, feature):
        """ To a given array it sets a dtype in which data will be exported
        """
        image_dtype = data_array.dtype if self.image_dtype is None else self.image_dtype

        if image_dtype == np.int64:
            image_dtype = np.int32
            warnings.warn('Data from feature {} cannot be exported to tiff with dtype numpy.int64. Will export as '
                          'numpy.int32 instead'.format(feature))

        if image_dtype == data_array.dtype:
            return data_array
        return data_array.astype(image_dtype)

    @staticmethod
    def _reshape_to_image_array(data_array):
        """ Reshapes an array in form of (times, height, width, bands) into array in form of (channels, height, width)
        """
        time_dim, height, width, band_dim = data_array.shape

        return np.moveaxis(data_array, -1, 1).reshape(time_dim * band_dim, height, width)

    def execute(self, eopatch, *, filename=None):
        """ Execute method

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param filename: filename of tiff file or None if entire path has already been specified in `folder` parameter
        of task initialization.
        :type filename: str or None
        :return: Unchanged input EOPatch
        :rtype: EOPatch
        """
        try:
            feature = next(self.feature(eopatch))
        except ValueError as error:
            LOGGER.warning(error)

            if self.fail_on_missing:
                raise ValueError(error)
            return eopatch

        image_array = self._prepare_image_array(eopatch, feature)

        channel_count, height, width = image_array.shape

        src_crs = {'init': CRS.ogc_string(eopatch.bbox.crs)}
        src_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)

        if self.crs:
            dst_crs = {'init': CRS.ogc_string(self.crs)}
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src_crs, dst_crs, width, height, *eopatch.bbox
            )
        else:
            dst_crs = src_crs
            dst_transform = src_transform
            dst_width, dst_height = width, height

        with rasterio.open(self._get_file_path(filename, create_dir=True), 'w', driver='GTiff',
                           width=dst_width, height=dst_height,
                           count=channel_count,
                           dtype=image_array.dtype, nodata=self.no_data_value,
                           transform=dst_transform, crs=dst_crs) as dst:

            if dst_crs == src_crs:
                dst.write(image_array)
            else:
                for idx in range(channel_count):
                    rasterio.warp.reproject(
                        source=image_array[idx, ...],
                        destination=rasterio.band(dst, idx + 1),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=rasterio.warp.Resampling.nearest
                    )

        return eopatch


class ImportFromTiff(BaseLocalIo):
    """ Task for importing data from a Geo-Tiff file into an EOPatch

    The task can take an existing EOPatch and read the part of Geo-Tiff image, which intersects with its bounding
    box, into a new feature. But if no EOPatch is given it will create a new EOPatch, read entire Geo-Tiff image into a
    feature and set a bounding box of the new EOPatch.

    Note that if Geo-Tiff file is not completely spatially aligned with location of given EOPatch it will try to fit it
    as best as possible. However it will not do any spatial resampling or interpolation on Geo-TIFF data.
    """
    def __init__(self, feature, folder=None, *, timestamp_size=None, **kwargs):
        """
        :param feature: EOPatch feature into which data will be imported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a path of an image file
        :type folder: str
        :param timestamp_size: In case data will be imported into time-dependant feature this parameter can be used to
            specify time dimension. If not specified, time dimension will be the same as size of FeatureType.TIMESTAMP
            feature. If FeatureType.TIMESTAMP does not exist it will be set to 1.
            When converting data into a feature channels of given tiff image should be in order
            T(1)B(1), T(1)B(2), ..., T(1)B(N), T(2)B(1), T(2)B(2), ..., T(2)B(N), ..., ..., T(M)B(N)
            where T and B are the time and band indices.
        :type timestamp_size: int
        :param image_dtype: Type of data of new feature imported from tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Values where given Geo-Tiff image does not cover EOPatch
        :type no_data_value: int or float
        """
        super().__init__(feature, folder=folder, **kwargs)

        self.timestamp_size = timestamp_size

    @staticmethod
    def _get_reading_window(width, height, data_bbox, eopatch_bbox):
        """ Calculates a window in pixel coordinates for which data will be read from an image
        """
        if eopatch_bbox.crs is not data_bbox.crs:
            eopatch_bbox = eopatch_bbox.transform(data_bbox.crs)

        # The following will be in the future moved to sentinelhub-py
        data_ul_x, data_lr_y = data_bbox.lower_left
        data_lr_x, data_ul_y = data_bbox.upper_right

        res_x = abs(data_ul_x - data_lr_x) / width
        res_y = abs(data_ul_y - data_lr_y) / height

        ul_x, lr_y = eopatch_bbox.lower_left
        lr_x, ul_y = eopatch_bbox.upper_right

        # If these coordinates wouldn't be rounded here, rasterio.io.DatasetReader.read would round
        # them in the same way
        top = round((data_ul_y - ul_y) / res_y)
        left = round((ul_x - data_ul_x) / res_x)
        bottom = round((data_ul_y - lr_y) / res_y)
        right = round((lr_x - data_ul_x) / res_x)

        return (top, bottom), (left, right)

    def execute(self, eopatch=None, *, filename=None):
        """ Execute method which adds a new feature to the EOPatch

        :param eopatch: input EOPatch or None if a new EOPatch should be created
        :type eopatch: EOPatch or None
        :param filename: filename of tiff file or None if entire path has already been specified in `folder` parameter
            of task initialization.
        :type filename: str or None
        :return: New EOPatch with added raster layer
        :rtype: EOPatch
        """
        feature_type, feature_name = next(self.feature())
        if eopatch is None:
            eopatch = EOPatch()

        with rasterio.open(self._get_file_path(filename)) as source:

            data_bbox = BBox(source.bounds, CRS(source.crs.to_epsg()))
            if eopatch.bbox is None:
                eopatch.bbox = data_bbox

            reading_window = self._get_reading_window(source.width, source.height, data_bbox, eopatch.bbox)

            data = source.read(window=reading_window, boundless=True, fill_value=self.no_data_value)

        if self.image_dtype is not None:
            data = data.astype(self.image_dtype)

        if not feature_type.is_spatial():
            data = data.flatten()

        if feature_type.is_timeless():
            data = np.moveaxis(data, 0, -1)
        else:
            channels = data.shape[0]

            times = self.timestamp_size
            if times is None:
                times = len(eopatch.timestamp) if eopatch.timestamp else 1

            if channels % times != 0:
                raise ValueError('Cannot import as a time-dependant feature because the number of tiff image channels '
                                 'is not divisible by the number of timestamps')

            data = data.reshape((times, channels // times) + data.shape[1:])
            data = np.moveaxis(data, 1, -1)

        eopatch[feature_type][feature_name] = data

        return eopatch
