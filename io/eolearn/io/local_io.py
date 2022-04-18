"""
Module containing tasks used for reading and writing to disk

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2018-2019 William Ouellette (TomTom)
Copyright (c) 2019 Drew Bollinger (DevelopmentSeed)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime
import logging
import warnings
from abc import ABCMeta

import dateutil
import fs
import numpy as np
import rasterio
from rasterio.windows import Window

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, EOTask
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.utils.fs import get_base_filesystem_and_path

LOGGER = logging.getLogger(__name__)


class BaseLocalIoTask(EOTask, metaclass=ABCMeta):
    """Base abstract class for local IO tasks"""

    def __init__(self, feature, folder=None, *, image_dtype=None, no_data_value=0, config=None):
        """
        :param feature: Feature which will be exported or imported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a folder of an image file
        :type folder: str
        :param image_dtype: Type of data to be exported into tiff image or imported from tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Value of undefined pixels
        :type no_data_value: int or float
        :param config: A configuration object containing AWS credentials
        :type config: SHConfig
        """
        self.feature = self.parse_feature(feature)
        self.folder = folder
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value
        self.config = config

    def _get_filesystem_and_paths(self, filename, timestamps, create_paths=False):
        """It takes location parameters from init and execute methods, joins them together, and creates a filesystem
        object and file paths relative to the filesystem object.
        """

        if isinstance(filename, str) or filename is None:
            filesystem, relative_path = get_base_filesystem_and_path(self.folder, filename, config=self.config)
            filename_paths = self._generate_paths(relative_path, timestamps)
        elif isinstance(filename, list):
            filename_paths = []
            for timestamp_index, path in enumerate(filename):
                filesystem, relative_path = get_base_filesystem_and_path(self.folder, path, config=self.config)
                if len(filename) == len(timestamps):
                    filename_paths.append(*self._generate_paths(relative_path, [timestamps[timestamp_index]]))
                elif not timestamps:
                    filename_paths.append(*self._generate_paths(relative_path, timestamps))
                else:
                    raise ValueError(
                        "The number of provided timestamps does not match the number of provided filenames."
                    )
        else:
            raise TypeError(f"The 'filename' parameter must either be a list or a string, but {filename} found")

        if create_paths:
            paths_to_create = {fs.path.dirname(filename_path) for filename_path in filename_paths}
            for filename_path in paths_to_create:
                filesystem.makedirs(filename_path, recreate=True)

        return filesystem, filename_paths

    @staticmethod
    def _generate_paths(path_template, timestamps):
        """Uses a filename path template to create a list of actual filename paths"""
        if not (path_template.lower().endswith(".tif") or path_template.lower().endswith(".tiff")):
            path_template = f"{path_template}.tif"

        if not timestamps:
            return [path_template]

        if "*" in path_template:
            path_template = path_template.replace("*", "%Y%m%dT%H%M%S")

        if timestamps[0].strftime(path_template) == path_template:
            return [path_template]

        return [timestamp.strftime(path_template) for timestamp in timestamps]


class ExportToTiffTask(BaseLocalIoTask):
    """Task exports specified feature to Geo-Tiff.

    When exporting multiple times OR bands, the Geo-Tiff `band` counts are in the expected order.
    However, when exporting multiple times AND bands, the order obeys the following pattern:

    T(1)B(1), T(1)B(2), ..., T(1)B(N), T(2)B(1), T(2)B(2), ..., T(2)B(N), ..., ..., T(M)B(N)

    where T and B are the time and band indices of the array,
    and M and N are the lengths of these indices, respectively
    """

    def __init__(
        self,
        feature,
        folder=None,
        *,
        band_indices=None,
        date_indices=None,
        crs=None,
        fail_on_missing=True,
        compress=None,
        **kwargs,
    ):
        """
        :param feature: Feature which will be exported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a path of an image file.
            If the file extension of the image file is not provided, it will default to ".tif".
            If a "*" wildcard or a datetime.strftime substring (e.g. "%Y%m%dT%H%M%S")  is provided in the image file,
            an EOPatch feature will be split over multiple GeoTiffs each corresponding to a timestamp,
            and the stringified datetime will be appended to the image file name.
        :type folder: str
        :param band_indices: Bands to be added to tiff image. Bands are represented by their 0-based index as tuple
            in the inclusive interval form `(start_band, end_band)` or as list in the form
            `[band_1, band_2,...,band_n]`.
        :type band_indices: tuple or list or None
        :param date_indices: Dates to be added to tiff image. Dates are represented by their 0-based index as tuple
            in the inclusive interval form `(start_date, end_date)` or a list in the form `[date_1, date_2,...,date_n]`.
        :type date_indices: tuple or list or None
        :param crs: CRS in which to reproject the feature before writing it to GeoTiff
        :type crs: CRS or str or None
        :param fail_on_missing: should the pipeline fail if a feature is missing or just log warning and return
        :type fail_on_missing: bool
        :param compress: the type of compression that rasterio should apply to exported image.
        :type compress: str or None
        :param image_dtype: Type of data to be exported into tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Value of pixels of tiff image with no data in EOPatch
        :type no_data_value: int or float
        :param config: A configuration object containing AWS credentials
        :type config: SHConfig
        """
        super().__init__(feature, folder=folder, **kwargs)

        self.band_indices = band_indices
        self.date_indices = date_indices
        self.crs = None if crs is None else CRS(crs)
        self.fail_on_missing = fail_on_missing
        self.compress = compress

    def _prepare_image_array(self, eopatch, feature):
        """Collects a feature from EOPatch and prepares the array of an image which will be rasterized. The resulting
        array has shape (channels, height, width) and is of correct dtype.
        """
        data_array = self._get_bands_subset(eopatch[feature])

        feature_type = feature[0]
        if feature_type.is_temporal():
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
        """Reduce array by selecting a subset of bands"""
        if self.band_indices is None:
            return array
        if isinstance(self.band_indices, list):
            if [band for band in self.band_indices if not isinstance(band, int)]:
                raise ValueError(f"Invalid format in {self.band_indices} list, expected integers")
            return array[..., self.band_indices]
        if isinstance(self.band_indices, tuple):
            if tuple(map(type, self.band_indices)) != (int, int):
                raise ValueError(f"Invalid format in {self.band_indices} tuple, expected integers")
            return array[..., self.band_indices[0] : self.band_indices[1] + 1]

        raise ValueError(f"Invalid format in {self.band_indices}, expected tuple or list")

    def _get_dates_subset(self, array, dates):
        """Reduce array by selecting a subset of times"""
        if self.date_indices is None:
            return array
        if isinstance(self.date_indices, list):
            if [date for date in self.date_indices if not isinstance(date, int)]:
                raise ValueError(f"Invalid format in {self.date_indices} list, expected integers")
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
                raise ValueError(f"Invalid format in {self.date_indices} tuple, expected ints, strings, or datetimes")
            return array[np.nonzero(np.where((dates >= start_date) & (dates <= end_date), dates, 0))[0]]

        raise ValueError(f"Invalid format in {self.date_indices}, expected tuple or list")

    def _set_export_dtype(self, data_array, feature):
        """To a given array it sets a dtype in which data will be exported"""
        image_dtype = data_array.dtype if self.image_dtype is None else self.image_dtype

        if image_dtype == np.int64:
            image_dtype = np.int32
            warnings.warn(
                f"Data from feature {feature} cannot be exported to tiff with dtype numpy.int64. Will export "
                "as numpy.int32 instead",
                EORuntimeWarning,
            )

        if image_dtype == data_array.dtype:
            return data_array
        return data_array.astype(image_dtype)

    @staticmethod
    def _reshape_to_image_array(data_array):
        """Reshapes an array in form of (times, height, width, bands) into array in form of (channels, height, width)"""
        time_dim, height, width, band_dim = data_array.shape

        return np.moveaxis(data_array, -1, 1).reshape(time_dim * band_dim, height, width)

    def _export_tiff(
        self,
        image_array,
        filesystem,
        path,
        channel_count,
        dst_crs,
        dst_height,
        dst_transform,
        dst_width,
        src_crs,
        src_transform,
    ):
        """Export an EOPatch feature to tiff based on input channel range."""
        with filesystem.openbin(path, "w") as file_handle:
            with rasterio.open(
                file_handle,
                "w",
                driver="GTiff",
                width=dst_width,
                height=dst_height,
                count=channel_count,
                dtype=image_array.dtype,
                nodata=self.no_data_value,
                transform=dst_transform,
                crs=dst_crs,
                compress=self.compress,
            ) as dst:

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
                            resampling=rasterio.warp.Resampling.nearest,
                        )

    def execute(self, eopatch, *, filename=None):
        """Execute method

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param filename: A filename of tiff file or None if entire path has already been specified in `folder`
            parameter of task initialization.
            If the file extension of the image file is not provided, it will default to ".tif".
            If a "*" wildcard or a datetime.strftime substring (e.g. "%Y%m%dT%H%M%S")  is provided in the image file,
            an EOPatch feature will be split over multiple GeoTiffs each corresponding to a timestamp,
            and the stringified datetime will be appended to the image file name.
        :type filename: str or None
        :return: Unchanged input EOPatch
        :rtype: EOPatch
        """
        if self.feature not in eopatch:
            error_msg = f"Feature {self.feature[1]} of type {self.feature[0]} was not found in EOPatch"
            LOGGER.warning(error_msg)
            if self.fail_on_missing:
                raise ValueError(error_msg)
            return eopatch

        image_array = self._prepare_image_array(eopatch, self.feature)

        channel_count, height, width = image_array.shape

        src_crs = eopatch.bbox.crs.ogc_string()
        src_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)

        if self.crs:
            dst_crs = self.crs.ogc_string()
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src_crs, dst_crs, width, height, *eopatch.bbox
            )
        else:
            dst_crs = src_crs
            dst_transform = src_transform
            dst_width, dst_height = width, height

        filesystem, filename_paths = self._get_filesystem_and_paths(filename, eopatch.timestamp, create_paths=True)

        with filesystem:
            if len(filename_paths) > 1:
                channel_count = channel_count // len(eopatch.timestamp)
                for timestamp_index, path in enumerate(filename_paths):
                    time_slice_array = image_array[
                        timestamp_index * channel_count : (timestamp_index + 1) * channel_count, ...
                    ]

                    self._export_tiff(
                        time_slice_array,
                        filesystem,
                        path,
                        channel_count,
                        dst_crs,
                        dst_height,
                        dst_transform,
                        dst_width,
                        src_crs,
                        src_transform,
                    )
            else:
                self._export_tiff(
                    image_array,
                    filesystem,
                    filename_paths[0],
                    channel_count,
                    dst_crs,
                    dst_height,
                    dst_transform,
                    dst_width,
                    src_crs,
                    src_transform,
                )

        return eopatch


class ImportFromTiffTask(BaseLocalIoTask):
    """Task for importing data from a Geo-Tiff file into an EOPatch

    The task can take an existing EOPatch and read the part of Geo-Tiff image, which intersects with its bounding
    box, into a new feature. But if no EOPatch is given it will create a new EOPatch, read entire Geo-Tiff image into a
    feature and set a bounding box of the new EOPatch.

    Note that if Geo-Tiff file is not completely spatially aligned with location of given EOPatch it will try to fit it
    as good as possible. However, it will not do any spatial resampling or interpolation on Geo-TIFF data.
    """

    def __init__(self, feature, folder=None, *, timestamp_size=None, **kwargs):
        """
        :param feature: EOPatch feature into which data will be imported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a path of an image file
        :type folder: str
        :param timestamp_size: In case data will be imported into time-dependant feature this parameter can be used to
            specify time dimension. If not specified, time dimension will be the same as size of `FeatureType.TIMESTAMP`
            feature. If `FeatureType.TIMESTAMP` does not exist it will be set to 1.
            When converting data into a feature channels of given tiff image should be in order
            T(1)B(1), T(1)B(2), ..., T(1)B(N), T(2)B(1), T(2)B(2), ..., T(2)B(N), ..., ..., T(M)B(N)
            where T and B are the time and band indices.
        :type timestamp_size: int
        :param image_dtype: Type of data of new feature imported from tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Values where given Geo-Tiff image does not cover EOPatch
        :type no_data_value: int or float
        :param config: A configuration object containing AWS credentials
        :type config: SHConfig
        """
        super().__init__(feature, folder=folder, **kwargs)

        self.timestamp_size = timestamp_size

    @staticmethod
    def _get_reading_window(tiff_source, eopatch_bbox):
        """Calculates a window in pixel coordinates for which data will be read from an image"""

        if eopatch_bbox.crs.epsg is not tiff_source.crs.to_epsg():
            eopatch_bbox = eopatch_bbox.transform(tiff_source.crs.to_epsg())

        tiff_upper_left = np.array([tiff_source.bounds.left, tiff_source.bounds.top])
        eopatch_upper_left = np.array([eopatch_bbox.min_x, eopatch_bbox.max_y])
        eopatch_lower_right = np.array([eopatch_bbox.max_x, eopatch_bbox.min_y])
        res = np.array(tiff_source.res)

        axis_flip = [1, -1]  # image origin is upper left, geographic origin is lower left

        col_off, row_off = axis_flip * (eopatch_upper_left - tiff_upper_left) / res
        width, height = abs(eopatch_lower_right - eopatch_upper_left) / res

        return Window(col_off, row_off, width, height)

    def execute(self, eopatch=None, *, filename=None):
        """Execute method which adds a new feature to the EOPatch

        :param eopatch: input EOPatch or None if a new EOPatch should be created
        :type eopatch: EOPatch or None
        :param filename: filename of tiff file or None if entire path has already been specified in `folder` parameter
            of task initialization.
        :type filename: str, list of str or None
        :return: New EOPatch with added raster layer
        :rtype: EOPatch
        """
        feature_type, feature_name = self.feature
        if eopatch is None:
            eopatch = EOPatch()

        filesystem, filename_paths = self._get_filesystem_and_paths(filename, eopatch.timestamp, create_paths=False)

        with filesystem:
            data = []
            for path in filename_paths:
                with filesystem.openbin(path, "r") as file_handle:
                    with rasterio.open(file_handle) as src:

                        boundless = True
                        if eopatch.bbox is None:
                            boundless = False
                            eopatch.bbox = BBox(src.bounds, CRS(src.crs.to_epsg()))

                        read_window = self._get_reading_window(src, eopatch.bbox)
                        data.append(src.read(window=read_window, boundless=boundless, fill_value=self.no_data_value))

        data = np.concatenate(data, axis=0)

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
                raise ValueError(
                    "Cannot import as a time-dependant feature because the number of tiff image channels "
                    "is not divisible by the number of timestamps"
                )

            data = data.reshape((times, channels // times) + data.shape[1:])
            data = np.moveaxis(data, 1, -1)

        eopatch[feature_type][feature_name] = data

        return eopatch
