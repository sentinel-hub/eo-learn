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
import datetime as dt
import functools
import logging
import warnings
from abc import ABCMeta
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import fs
import numpy as np
import rasterio
import rasterio.warp
from affine import Affine
from fs.base import FS
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from rasterio.io import DatasetReader
from rasterio.session import AWSSession
from rasterio.windows import Window, from_bounds

from sentinelhub import CRS, BBox, SHConfig, parse_time_interval

from eolearn.core import EOPatch, FeatureType
from eolearn.core.core_tasks import IOTask
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.utils.fs import get_base_filesystem_and_path, get_full_path

LOGGER = logging.getLogger(__name__)


class BaseRasterIoTask(IOTask, metaclass=ABCMeta):  # noqa: B024
    """Base abstract class for raster IO tasks"""

    def __init__(
        self,
        feature,
        folder: str,
        *,
        filesystem: Optional[FS] = None,
        image_dtype: Optional[Union[np.dtype, type]] = None,
        no_data_value: Optional[float] = None,
        create: bool = False,
        config: Optional[SHConfig] = None,
    ):
        """
        :param feature: Feature which will be exported or imported
        :param folder: A path to a main folder containing all image, potentially in its subfolders. If `filesystem`
            parameter is defined, then `folder` should be a path relative to filesystem object. Otherwise, it should be
            an absolute path.
        :param filesystem: An existing filesystem object. If not given it will be initialized according to `folder`
            parameter.
        :param image_dtype: A data type of data in exported images or data imported from images.
        :param no_data_value: When exporting this is the NoData value of pixels in exported images.
            When importing this value is assigned to the pixels with NoData.
        :param create: If the filesystem path doesn't exist this flag indicates to either create it or raise an error.
        :param config: A configuration object with AWS credentials. By default, is set to None and in this case the
            default configuration will be taken.
        """
        self.feature = self.parse_feature(feature)
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value

        if filesystem is None:
            filesystem, folder = get_base_filesystem_and_path(folder, create=create, config=config)

        super().__init__(folder, filesystem=filesystem, create=create, config=config)

    def _get_filename_paths(self, filename_template: Union[str, List[str]], timestamps: List[dt.datetime]) -> List[str]:
        """From a filename "template" and base path on the filesystem it generates full paths to tiff files. The paths
        are still relative to the filesystem object.
        """
        if isinstance(filename_template, str):
            filename_path = fs.path.join(self.filesystem_path, filename_template)
            filename_paths = self._generate_paths(filename_path, timestamps)

        elif isinstance(filename_template, list):
            filename_paths = []
            for timestamp_index, path in enumerate(filename_template):
                filename_path = fs.path.join(self.filesystem_path, path)
                if len(filename_template) == len(timestamps):
                    filename_paths.extend(self._generate_paths(filename_path, [timestamps[timestamp_index]]))
                elif not timestamps:
                    filename_paths.extend(self._generate_paths(filename_path, timestamps))
                else:
                    raise ValueError(
                        "The number of provided timestamps does not match the number of provided filenames."
                    )
        else:
            raise TypeError(
                f"The 'filename' parameter must either be a list or a string, but {filename_template} found"
            )

        if self._create_path:
            paths_to_create = {fs.path.dirname(filename_path) for filename_path in filename_paths}
            for filename_path in paths_to_create:
                self.filesystem.makedirs(filename_path, recreate=True)

        return filename_paths

    @classmethod
    def _generate_paths(cls, path_template: str, timestamps: List[dt.datetime]) -> List[str]:
        """Uses a filename path template to create a list of actual filename paths."""
        if not cls._has_tiff_file_extension(path_template):
            path_template = f"{path_template}.tif"

        if not timestamps:
            return [path_template]

        if "*" in path_template:
            path_template = path_template.replace("*", "%Y%m%dT%H%M%S")

        if timestamps[0].strftime(path_template) == path_template:
            return [path_template]

        return [timestamp.strftime(path_template) for timestamp in timestamps]

    @staticmethod
    def _has_tiff_file_extension(path: str) -> bool:
        """Checks if path ends with a tiff file extension."""
        path = path.lower()
        return path.endswith(".tif") or path.endswith(".tiff")


class ExportToTiffTask(BaseRasterIoTask):
    """Task exports specified feature to GeoTIFF.

    The task can export also features with sizes of both time and band dimension greater than `1`. When exporting
    only multiple times or only multiple bands, the GeoTIFF channels are in the expected order. However, when
    exporting multiple times and bands, the order obeys the following pattern:

    T(1)B(1), T(1)B(2), ..., T(1)B(N), T(2)B(1), T(2)B(2), ..., T(2)B(N), ..., ..., T(M)B(N)

    where T and B are the time and band indices of the array, and M and N are the lengths of these indices,
    respectively.
    """

    def __init__(
        self,
        feature,
        folder: str,
        *,
        date_indices: Union[List[int], Tuple[int, int], Tuple[dt.datetime, dt.datetime], Tuple[str, str], None] = None,
        band_indices: Union[List[int], Tuple[int, int], None] = None,
        crs: Union[CRS, int, str, None] = None,
        fail_on_missing: bool = True,
        compress: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        :param feature: A feature to be exported.
        :param folder: A path to a main folder containing all image, potentially in its subfolders. If `filesystem`
            parameter is defined, then `folder` should be a path relative to filesystem object. Otherwise, it should be
            an absolute path.
        :param date_indices: Indices of those time frames from the give feature that will be exported to a tiff image.
            It can be either a list of indices or a tuple of `2` indices defining an interval of indices or a tuple of
            `2` datetime object also defining a time interval. By default, all time frames will be exported.
        :param band_indices: Indices of those bands from the given feature that will be exported to a tiff image. It
            can be either a list of indices or a tuple of `2` indices defining an interval of indices. By default, all
            bands will be exported.
        :param crs: A CRS in which to reproject the feature before writing it to GeoTIFF. By default, no reprojection
            will be done.
        :param fail_on_missing: A flag to specify if the task should fail if a feature is missing or if it should
            just show a warning.
        :param compress: A type of compression that rasterio should apply to an exported image.
        :param kwargs: Keyword arguments to be propagated to `BaseRasterIoTask`.
        """
        super().__init__(feature, folder=folder, create=True, **kwargs)

        self.date_indices = date_indices
        self.band_indices = band_indices
        self.crs = None if crs is None else CRS(crs)
        self.fail_on_missing = fail_on_missing
        self.compress = compress

    def _prepare_image_array(
        self, data_array: np.ndarray, timestamps: List[dt.datetime], feature: Tuple[FeatureType, str]
    ) -> np.ndarray:
        """Collects a feature from EOPatch and prepares the array of an image which will be rasterized. The resulting
        array has shape (channels, height, width) and is of correct dtype.
        """
        data_array = self._reduce_by_bands(data_array)

        feature_type, _ = feature
        if feature_type.is_temporal():
            data_array = self._reduce_by_time(data_array, timestamps)
        else:
            # add temporal dimension
            data_array = np.expand_dims(data_array, axis=0)

        if not feature_type.is_spatial():
            # add height and width dimensions
            data_array = np.expand_dims(np.expand_dims(data_array, axis=1), axis=1)

        data_array = self._set_export_dtype(data_array, feature)

        time_dim, height, width, band_dim = data_array.shape
        new_shape = (time_dim * band_dim, height, width)
        data_array = np.moveaxis(data_array, -1, 1).reshape(new_shape)

        return data_array

    def _reduce_by_bands(self, array: np.ndarray) -> np.ndarray:
        """Reduces the array by selecting a subset of bands."""
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

    def _reduce_by_time(self, array: np.ndarray, timestamps: List[dt.datetime]) -> np.ndarray:
        """Reduce array by selecting a subset of times."""
        if self.date_indices is None:
            return array

        if isinstance(self.date_indices, list):
            if [date for date in self.date_indices if not isinstance(date, int)]:
                raise ValueError(f"Invalid format in {self.date_indices} list, expected integers")
            return array[np.array(self.date_indices), ...]

        if isinstance(self.date_indices, tuple):
            dates = np.array(timestamps)
            if tuple(map(type, self.date_indices)) == (int, int):
                start_date = dates[self.date_indices[0]]
                end_date = dates[self.date_indices[1]]
            elif tuple(map(type, self.date_indices)) == (str, str):
                start_date, end_date = parse_time_interval(self.date_indices)
            elif tuple(map(type, self.date_indices)) == (dt.datetime, dt.datetime):
                start_date = self.date_indices[0]
                end_date = self.date_indices[1]
            else:
                raise ValueError(f"Invalid format in {self.date_indices} tuple, expected ints, strings, or datetimes")
            return array[np.nonzero(np.where((dates >= start_date) & (dates <= end_date), dates, 0))[0]]

        raise ValueError(f"Invalid format in {self.date_indices}, expected tuple or list")

    def _set_export_dtype(self, data_array: np.ndarray, feature: Tuple[FeatureType, str]) -> np.ndarray:
        """To a given array it sets a dtype in which data will be exported"""
        image_dtype = data_array.dtype if self.image_dtype is None else self.image_dtype

        if image_dtype == np.int64:
            image_dtype = np.int32
            warnings.warn(
                (
                    f"Data from feature {feature} cannot be exported to tiff with dtype numpy.int64. Will export "
                    "as numpy.int32 instead"
                ),
                EORuntimeWarning,
            )

        if image_dtype == data_array.dtype:
            return data_array
        return data_array.astype(image_dtype)

    def _get_source_and_destination_params(
        self, data_array: np.ndarray, bbox: BBox
    ) -> Tuple[Tuple[str, Affine], Tuple[str, Affine], Tuple[int, int]]:
        """Calculates source and destination CRS and transforms. Additionally, it returns destination height and width
        """
        _, height, width = data_array.shape

        src_crs = bbox.crs.ogc_string()
        src_transform = rasterio.transform.from_bounds(*bbox, width=width, height=height)

        if self.crs:
            dst_crs = self.crs.ogc_string()
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src_crs, dst_crs, width, height, *bbox
            )
        else:
            dst_crs = src_crs
            dst_transform = src_transform
            dst_width, dst_height = width, height

        return (src_crs, src_transform), (dst_crs, dst_transform), (dst_height, dst_width)

    def _export_tiff(
        self,
        image_array: np.ndarray,
        filesystem: FS,
        path: str,
        channel_count: int,
        dst_crs: str,
        dst_transform: Affine,
        dst_height: int,
        dst_width: int,
        src_crs: str,
        src_transform: Affine,
    ) -> None:
        """Export an EOPatch feature to tiff based on input channel range."""
        with rasterio.Env(), filesystem.openbin(path, "w") as file_handle:  # noqa: SIM117
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

    def execute(self, eopatch: EOPatch, *, filename: Union[str, List[str], None] = "") -> EOPatch:
        """Execute method

        :param eopatch: An input EOPatch
        :param filename: A filename with a path to a tiff file. The path has to be a relative to the filesystem object
            and path from `folder` initialization parameter. If the file extension of the image file is not provided,
            it will default to `.tif`. If a "*" wildcard or a datetime.strftime substring (e.g. "%Y%m%dT%H%M%S") is
            provided in the path, an EOPatch feature will be split over multiple GeoTIFFs where each one will correspond
            to a single timestamp. Alternatively, a list of paths can be provided, one for each timestamp. Set this
            parameter to `None` to disable exporting.
        :return: Unchanged input EOPatch
        """
        if filename is None:
            return eopatch

        if self.feature not in eopatch:
            error_msg = f"Feature {self.feature[1]} of type {self.feature[0]} was not found in EOPatch"
            LOGGER.warning(error_msg)
            if self.fail_on_missing:
                raise ValueError(error_msg)
            return eopatch
        if eopatch.bbox is None:
            raise ValueError(
                "Given EOPatch is missing a bounding box and therefore no feature can be exported to GeoTIFF"
            )

        image_array = self._prepare_image_array(eopatch[self.feature], eopatch.timestamp, self.feature)

        (
            (src_crs, src_transform),
            (dst_crs, dst_transform),
            (dst_height, dst_width),
        ) = self._get_source_and_destination_params(image_array, eopatch.bbox)

        filename_paths = self._get_filename_paths(filename, eopatch.timestamp)

        with self.filesystem as filesystem:
            export_function = functools.partial(
                self._export_tiff,
                filesystem=filesystem,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                dst_height=dst_height,
                dst_width=dst_width,
                src_crs=src_crs,
                src_transform=src_transform,
            )

            channel_count = image_array.shape[0]
            if len(filename_paths) > 1:
                single_channel_count = channel_count // len(eopatch.timestamp)
                for timestamp_index, path in enumerate(filename_paths):
                    time_slice_array = image_array[
                        timestamp_index * single_channel_count : (timestamp_index + 1) * single_channel_count, ...
                    ]
                    export_function(image_array=time_slice_array, path=path, channel_count=single_channel_count)
            else:
                export_function(image_array=image_array, path=filename_paths[0], channel_count=channel_count)

        return eopatch


class ImportFromTiffTask(BaseRasterIoTask):
    """Task for importing data from a GeoTIFF file into an EOPatch

    The task can take an existing EOPatch and read the part of GeoTIFF image, which intersects with its bounding
    box, into a new feature. But if no EOPatch is given it will create a new EOPatch, read entire GeoTIFF image into a
    feature and set a bounding box of the new EOPatch.

    Note that if GeoTIFF file is not completely spatially aligned with location of given EOPatch it will try to fit it
    as good as possible. However, it will not do any spatial resampling or interpolation on GeoTIFF data.
    """

    def __init__(
        self, feature, folder: str, *, use_vsi: bool = False, timestamp_size: Optional[int] = None, **kwargs: Any
    ):
        """
        :param feature: EOPatch feature into which data will be imported
        :param folder: A directory containing image files or a path of an image file
        :param use_vsi: A flag to define if reading should be done with GDAL/rasterio virtual system (VSI)
            functionality. The flag only has an effect when the task is used to read an image from a remote storage
            (i.e. AWS S3 bucket). For a performance improvement it is recommended to set this to `True` when reading a
            smaller chunk of a larger image, especially if it is a Cloud-optimized GeoTIFF (COG). In other cases the
            reading might be faster if the flag remains set to `False`.
        :param timestamp_size: In case data will be imported into a time-dependant feature this parameter can be used to
            specify time dimension. If not specified, time dimension will be the same as size of `FeatureType.TIMESTAMP`
            feature. If `FeatureType.TIMESTAMP` does not exist it will be set to 1.
            When converting data into a feature channels of given tiff image should be in order
            T(1)B(1), T(1)B(2), ..., T(1)B(N), T(2)B(1), T(2)B(2), ..., T(2)B(N), ..., ..., T(M)B(N)
            where T and B are the time and band indices.
        :param kwargs: Keyword arguments to be propagated to `BaseRasterIoTask`.
        """
        super().__init__(feature, folder=folder, **kwargs)

        self.use_vsi = use_vsi
        self.timestamp_size = timestamp_size

    def _get_session(self, filesystem: FS) -> AWSSession:
        """Creates a session object with credentials from a config object."""
        if not isinstance(filesystem, S3FS):
            raise NotImplementedError("A rasterio session for VSI reading for now only works for AWS S3 filesystems")

        return AWSSession(
            aws_access_key_id=filesystem.aws_access_key_id,
            aws_secret_access_key=filesystem.aws_secret_access_key,
            aws_session_token=filesystem.aws_session_token,
            region_name=filesystem.region,
            endpoint_url=filesystem.endpoint_url,
        )

    def _load_from_image(self, path: str, filesystem: FS, bbox: Optional[BBox]) -> Tuple[np.ndarray, Optional[BBox]]:
        """The method decides in what way data will be loaded the image.

        The method always uses `rasterio.Env` to suppress any low-level warnings. In case of a local filesystem
        benchmarks show that without `filesystem.openbin` in some cases `rasterio` can read much faster. Otherwise,
        reading depends on `use_vsi` flag. In some cases where a sub-image window is read and the image is in certain
        format (e.g. COG), benchmarks show that reading with virtual system (VSI) is much faster. In other cases,
        reading with `filesystem.openbin` is faster.
        """
        if isinstance(filesystem, (OSFS, TempFS)):
            full_path = filesystem.getsyspath(path)
            with rasterio.Env():
                return self._read_image(full_path, bbox)

        if self.use_vsi:
            session = self._get_session(filesystem)
            with rasterio.Env(session=session):
                full_path = get_full_path(filesystem, path)
                return self._read_image(full_path, bbox)

        with rasterio.Env(), filesystem.openbin(path, "r") as file_handle:
            return self._read_image(file_handle, bbox)

    def _read_image(self, file_object: Union[str, BinaryIO], bbox: Optional[BBox]) -> Tuple[np.ndarray, Optional[BBox]]:
        """Reads data from the image."""
        with rasterio.open(file_object) as src:
            src: DatasetReader

            read_window, read_bbox = self._get_reading_window_and_bbox(src, bbox)
            boundless_reading = read_window is not None
            return src.read(window=read_window, boundless=boundless_reading, fill_value=self.no_data_value), read_bbox

    @staticmethod
    def _get_reading_window_and_bbox(
        reader: DatasetReader, bbox: Optional[BBox]
    ) -> Tuple[Optional[Window], Optional[BBox]]:
        """Provides a reading window for which data will be read from image. If it returns `None` this means that the
        whole image should be read. Those cases are when bbox is not defined, image is not geo-referenced, or
        bbox coordinates exactly match image coordinates. Additionally, it provides a bounding box of reading window
        if an image is geo-referenced."""
        image_crs = reader.crs
        image_transform = reader.transform
        image_bounds = reader.bounds
        if image_crs is None or image_transform is None or image_bounds is None:
            return None, bbox

        image_bbox = BBox(list(image_bounds), crs=image_crs.to_epsg())
        if bbox is None:
            return None, image_bbox

        original_bbox = bbox
        if bbox.crs is not image_bbox.crs:
            bbox = bbox.transform(image_crs.to_epsg())

        if bbox == image_bbox:
            return None, original_bbox

        return from_bounds(*iter(bbox), transform=image_transform), original_bbox

    def _load_data(self, filename_paths: List[str], initial_bbox: Optional[BBox]) -> Tuple[np.ndarray, Optional[BBox]]:
        """Load data from images, join them, and provide their bounding box."""
        data_per_path: List[np.ndarray] = []
        final_bbox: Optional[BBox] = None

        with self.filesystem as filesystem:
            for path in filename_paths:
                data, bbox = self._load_from_image(path, filesystem, initial_bbox)
                data_per_path.append(data)

                if bbox is None:
                    continue
                if final_bbox and bbox != final_bbox:
                    raise RuntimeError(
                        "Given images have different geo-references. They can't be imported into an EOPatch that"
                        " doesn't have a bounding box."
                    )
                final_bbox = bbox

        return np.concatenate(data_per_path, axis=0), final_bbox

    def execute(self, eopatch: Optional[EOPatch] = None, *, filename: Optional[str] = "") -> EOPatch:
        """Execute method which adds a new feature to the EOPatch

        :param eopatch: input EOPatch or None if a new EOPatch should be created
        :type eopatch: EOPatch or None
        :param filename: filename of tiff file or None if entire path has already been specified in `folder` parameter
            of task initialization.
        :type filename: str, list of str or None
        :return: New EOPatch with added raster layer
        :rtype: EOPatch
        """
        if filename is None:
            if eopatch is None:
                raise ValueError("Both eopatch and filename parameters cannot be set to None")
            return eopatch

        feature_type, feature_name = self.feature
        eopatch = eopatch or EOPatch()

        filename_paths = self._get_filename_paths(filename, eopatch.timestamp)

        data, bbox = self._load_data(filename_paths, eopatch.bbox)

        if eopatch.bbox is None:
            eopatch.bbox = bbox

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
