"""
A module implementing utilities for working with saving and loading EOPatch data

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import gzip
import itertools
import json
import platform
import sys
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
)

import dateutil.parser
import fs
import fs.move
import fs_s3fs
import geopandas as gpd
import numpy as np
import pandas as pd
from fs.base import FS
from fs.osfs import OSFS
from fs.tempfs import TempFS
from typing_extensions import TypeAlias

from sentinelhub import CRS, BBox, Geometry, MimeType
from sentinelhub.exceptions import SHUserWarning, deprecated_function

from .constants import TIMESTAMP_COLUMN, FeatureType, OverwritePermission
from .exceptions import EODeprecationWarning
from .types import EllipsisType, FeatureSpec, FeaturesSpecification
from .utils.parsing import FeatureParser

try:
    import s3fs
    import zarr
except ImportError:
    pass

if TYPE_CHECKING:
    from .eodata import EOPatch


T = TypeVar("T")
Self = TypeVar("Self", bound="FeatureIO")
PatchContentType: TypeAlias = Tuple[
    Optional["FeatureIOBBox"],
    Optional["FeatureIOTimestamps"],
    Optional["FeatureIOJson"],
    Dict[Tuple[FeatureType, str], "FeatureIO"],
]


BBOX_FILENAME = "bbox"
TIMESTAMPS_FILENAME = "timestamps"


@dataclass
class FilesystemDataInfo:
    """Information about data that is present on the filesystem. Fields represent paths to relevant file."""

    timestamps: str | None = None
    bbox: str | None = None
    meta_info: str | None = None
    features: dict[FeatureType, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))

    def iterate_features(self) -> Iterator[tuple[tuple[FeatureType, str], str]]:
        """Yields `(ftype, fname), path` tuples from `features`."""
        for ftype, ftype_dict in self.features.items():
            for fname, path in ftype_dict.items():
                yield (ftype, fname), path


def save_eopatch(
    eopatch: EOPatch,
    filesystem: FS,
    patch_location: str,
    *,
    features: FeaturesSpecification,
    overwrite_permission: OverwritePermission,
    compress_level: int,
    use_zarr: bool,
) -> None:
    """A utility function used by `EOPatch.save` method."""
    patch_exists = filesystem.exists(patch_location)

    eopatch_features = FeatureParser(features).get_features(eopatch)
    file_information = get_filesystem_data_info(filesystem, patch_location) if patch_exists else FilesystemDataInfo()

    _check_collisions(overwrite_permission, eopatch_features, file_information)

    # Data must be collected before any tinkering with files due to lazy-loading
    data_for_saving = list(_yield_features_to_save(eopatch, eopatch_features, patch_location, use_zarr))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=EODeprecationWarning)
        if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
            _remove_old_eopatch(filesystem, patch_location)

    ftype_folders = {fs.path.dirname(path) for _, _, path in data_for_saving}
    for folder in ftype_folders:
        filesystem.makedirs(folder, recreate=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        save_function = partial(_save_single_feature, filesystem=filesystem, compress_level=compress_level)
        list(executor.map(save_function, data_for_saving))  # Wrapped in a list to get better exceptions

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=EODeprecationWarning)
        if overwrite_permission is not OverwritePermission.OVERWRITE_PATCH:
            remove_redundant_files(filesystem, data_for_saving, file_information, compress_level)


def _remove_old_eopatch(filesystem: FS, patch_location: str) -> None:
    filesystem.removetree(patch_location)
    filesystem.makedirs(patch_location, recreate=True)


def _yield_features_to_save(
    eopatch: EOPatch, eopatch_features: list[FeatureSpec], patch_location: str, use_zarr: bool
) -> Iterator[tuple[type[FeatureIO], Any, str]]:
    """Prepares a triple `(featureIO, data, path)` so that the `featureIO` can save `data` to `path`."""
    get_file_path = partial(fs.path.join, patch_location)
    meta_features = {ftype for ftype, _ in eopatch_features if ftype.is_meta()}

    if eopatch.bbox is not None:  # remove after BBox is never None
        yield (FeatureIOBBox, eopatch.bbox, get_file_path(BBOX_FILENAME))

    if eopatch.timestamps and FeatureType.TIMESTAMPS in meta_features:
        yield (FeatureIOTimestamps, eopatch.timestamps, get_file_path(TIMESTAMPS_FILENAME))

    if eopatch.meta_info and FeatureType.META_INFO in meta_features:
        yield (FeatureIOJson, eopatch.meta_info, get_file_path(FeatureType.META_INFO.value))

    for ftype, fname in eopatch_features:
        if ftype.is_meta():
            continue
        yield (_get_feature_io_constructor(ftype, use_zarr), eopatch[(ftype, fname)], get_file_path(ftype.value, fname))


def _save_single_feature(save_spec: tuple[type[FeatureIO[T]], T, str], *, filesystem: FS, compress_level: int) -> None:
    feature_io, data, feature_path = save_spec
    feature_io.save(data, filesystem, feature_path, compress_level)


def remove_redundant_files(
    filesystem: FS,
    data_for_saving: list[tuple[type[FeatureIO], Any, str]],
    preexisting_files: FilesystemDataInfo,
    current_compress_level: int,
) -> None:
    """Removes files that should have been overwritten but were not due to different file extensions."""
    feature_paths = (path for _, ftype_dict in preexisting_files.features.items() for _, path in ftype_dict.items())
    meta_paths = (preexisting_files.bbox, preexisting_files.meta_info, preexisting_files.timestamps)
    split_paths = (path.split(".", maxsplit=1) for path in filter(None, itertools.chain(meta_paths, feature_paths)))
    old_path_extension = dict(split_paths)  # maps {path_base: file_extension}

    files_to_remove = []
    for feature_io, _, base in data_for_saving:
        extension = feature_io.get_file_extension()
        if issubclass(feature_io, FeatureIOGZip) and current_compress_level > 0:
            extension += f".{MimeType.GZIP.extension}"
        if base in old_path_extension and old_path_extension[base] != extension:
            files_to_remove.append(f"{base}.{old_path_extension[base]}")

    def _remover(path: str):  # Zarr path can also be path to a folder
        if not path.endswith("zarr") or filesystem.isfile(path):
            return filesystem.remove(path)
        return filesystem.removetree(path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(_remover, files_to_remove))  # Wrapped in a list to get better exceptions


def load_eopatch_content(filesystem: FS, patch_location: str, features: FeaturesSpecification) -> PatchContentType:
    """A utility function used by `EOPatch.load` method."""
    file_information = get_filesystem_data_info(filesystem, patch_location, features)
    bbox, timestamps, meta_info = _load_meta_features(filesystem, file_information, features)

    features_dict: dict[tuple[FeatureType, str], FeatureIO] = {}
    for ftype, fname in FeatureParser(features).get_feature_specifications():
        if ftype.is_meta():
            continue

        if fname is ...:
            for fname, path in file_information.features.get(ftype, {}).items():
                use_zarr = path.endswith(FeatureIOZarr.get_file_extension())
                features_dict[(ftype, fname)] = _get_feature_io_constructor(ftype, use_zarr)(path, filesystem)
        else:
            if ftype not in file_information.features or fname not in file_information.features[ftype]:
                raise IOError(f"Feature {(ftype, fname)} does not exist in eopatch at {patch_location}.")
            path = file_information.features[ftype][fname]
            use_zarr = path.endswith(FeatureIOZarr.get_file_extension())
            features_dict[(ftype, fname)] = _get_feature_io_constructor(ftype, use_zarr)(path, filesystem)

    return bbox, timestamps, meta_info, features_dict


def _load_meta_features(
    filesystem: FS, file_information: FilesystemDataInfo, features: FeaturesSpecification
) -> tuple[FeatureIOBBox | None, FeatureIOTimestamps | None, FeatureIOJson | None]:
    requested = {ftype for ftype, _ in FeatureParser(features).get_feature_specifications() if ftype.is_meta()}

    err_msg = "Feature {} is specified to be loaded but does not exist in EOPatch."

    bbox = None
    if file_information.bbox is not None:
        bbox = FeatureIOBBox(file_information.bbox, filesystem)
    elif FeatureType.BBOX in requested and features is not Ellipsis:
        raise IOError(err_msg.format(FeatureType.BBOX))

    timestamps = None
    if FeatureType.TIMESTAMPS in requested:
        if file_information.timestamps is not None:
            timestamps = FeatureIOTimestamps(file_information.timestamps, filesystem)
        elif features is not Ellipsis:
            raise IOError(err_msg.format(FeatureType.TIMESTAMPS))

    meta_info = None
    if FeatureType.META_INFO in requested:
        if file_information.meta_info is not None:
            meta_info = FeatureIOJson(file_information.meta_info, filesystem)
        elif any(
            ftype == FeatureType.META_INFO and isinstance(fname, str)
            for ftype, fname in FeatureParser(features).get_feature_specifications()
        ):
            raise IOError(err_msg.format(FeatureType.META_INFO))

    return bbox, timestamps, meta_info


def get_filesystem_data_info(
    filesystem: FS, patch_location: str, features: FeaturesSpecification = ...
) -> FilesystemDataInfo:
    """Returns information on all eopatch files in the storage. Filters with `features` to reduce IO calls."""
    relevant_features = FeatureParser(features).get_feature_specifications()
    relevant_feature_types = {ftype for ftype, _ in relevant_features}

    result = FilesystemDataInfo()

    for path in filesystem.listdir(patch_location):
        object_name = _remove_file_extension(path).strip("/")
        object_path = fs.path.combine(patch_location, path)

        if object_name == "timestamp":
            warnings.warn(
                (
                    f"EOPatch at {patch_location} contains the deprecated naming `timestamp` for the `timestamps`"
                    " feature. The old name will no longer be valid in the future. You can re-save the `EOPatch` to"
                    " update it."
                ),
                category=EODeprecationWarning,
                stacklevel=2,
            )
            object_name = TIMESTAMPS_FILENAME

        if "/" in object_name:  # For cases where S3 does not have a regular folder structure
            ftype_str, fname = fs.path.split(object_name)
            result.features[FeatureType(ftype_str)][fname] = path

        elif object_name == BBOX_FILENAME:
            result.bbox = object_path

        elif object_name == TIMESTAMPS_FILENAME:
            result.timestamps = object_path

        elif object_name == FeatureType.META_INFO.value:
            result.meta_info = object_path

        elif FeatureType.has_value(object_name) and FeatureType(object_name) in relevant_feature_types:
            result.features[FeatureType(object_name)] = dict(walk_feature_type_folder(filesystem, object_path))

    # Note: might simplify a few things if we filtered according to features here, especially loading stuff.
    return result


@deprecated_function(category=EODeprecationWarning)
def walk_filesystem(
    filesystem: FS, patch_location: str, features: FeaturesSpecification = ...
) -> Iterator[tuple[FeatureType, str | EllipsisType, str]]:
    """Interface to the old walk_filesystem function which yields tuples of (feature_type, feature_name, file_path)."""
    file_information = get_filesystem_data_info(filesystem, patch_location, features)

    if file_information.bbox is not None:  # remove after BBox is never None
        yield (FeatureType.BBOX, ..., file_information.bbox)

    if file_information.timestamps is not None:
        yield (FeatureType.TIMESTAMPS, ..., file_information.timestamps)

    if file_information.meta_info is not None:
        yield (FeatureType.META_INFO, ..., file_information.meta_info)

    for feature, path in file_information.iterate_features():
        yield (*feature, path)


def walk_feature_type_folder(filesystem: FS, folder_path: str) -> Iterator[tuple[str, str]]:
    """Walks a feature type subfolder of EOPatch and yields tuples (feature name, path in filesystem).
    Skips folders and files in subfolders.
    """
    for path in filesystem.listdir(folder_path):
        if "/" not in path and "." in path:
            yield _remove_file_extension(path), fs.path.combine(folder_path, path)


def _check_collisions(
    overwrite_permission: OverwritePermission, eopatch_features: list[FeatureSpec], existing_files: FilesystemDataInfo
) -> None:
    """Checks for possible name collisions to avoid unintentional overwriting."""
    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_letter_case_collisions(eopatch_features, existing_files)
        _check_add_only_permission(eopatch_features, existing_files)

    elif platform.system() == "Windows" and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES:
        _check_letter_case_collisions(eopatch_features, existing_files)

    else:
        _check_letter_case_collisions(eopatch_features, FilesystemDataInfo())


def _check_add_only_permission(eopatch_features: list[FeatureSpec], filesystem_features: FilesystemDataInfo) -> None:
    """Checks that no existing feature will be overwritten."""
    unique_filesystem_features = {_to_lowercase(*feature) for feature, _ in filesystem_features.iterate_features()}
    unique_eopatch_features = {_to_lowercase(*feature) for feature in eopatch_features}

    intersection = unique_filesystem_features.intersection(unique_eopatch_features)
    if intersection:
        raise ValueError(f"Cannot save features {intersection} with overwrite_permission=OverwritePermission.ADD_ONLY")


def _check_letter_case_collisions(eopatch_features: list[FeatureSpec], filesystem_features: FilesystemDataInfo) -> None:
    """Check that features have no name clashes (ignoring case) with other EOPatch features and saved features."""
    lowercase_features = {_to_lowercase(*feature) for feature in eopatch_features}

    if len(lowercase_features) != len(eopatch_features):
        raise IOError("Some features differ only in casing and cannot be saved in separate files.")

    for feature, _ in filesystem_features.iterate_features():
        if feature not in eopatch_features and _to_lowercase(*feature) in lowercase_features:
            raise IOError(
                f"There already exists a feature {feature} in the filesystem that only differs in "
                "casing from a feature that should be saved."
            )


def _to_lowercase(ftype: FeatureType, fname: str | None, *_: Any) -> tuple[FeatureType, str | None]:
    """Transforms a feature to it's lowercase representation."""
    return ftype, fname if fname is None else fname.lower()


def _remove_file_extension(path: str) -> str:
    """This also removes file extensions of form `.geojson.gz` unlike `fs.path.splitext`."""
    return path.split(".")[0]


class FeatureIO(Generic[T], metaclass=ABCMeta):
    """A class that handles the saving and loading process of a single feature at a given location."""

    def __init__(self, path: str, filesystem: FS):
        """
        :param path: A path in the filesystem
        :param filesystem: A filesystem object
        """
        self._check_path_extension(path)

        self.path = path
        self.filesystem = filesystem

        self.loaded_value: T | None = None

    def _check_path_extension(self, path: str) -> None:
        filename = fs.path.basename(path)
        expected_extension = f".{self.get_file_extension()}"
        if not filename.endswith(expected_extension):
            raise ValueError(f"FeatureIO expects a filepath with the {expected_extension} file extension, got {path}")

    @classmethod
    @abstractmethod
    def get_file_extension(cls) -> str:
        """The extension of files handled by the FeatureIO."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"

    def load(self) -> T:
        """Method for loading a feature. The loaded value is stored into an attribute in case a second load request is
        triggered inside a shallow-copied EOPatch.
        """
        if self.loaded_value is None:
            self.loaded_value = self._load_value()

        return self.loaded_value

    @abstractmethod
    def _load_value(self) -> T:
        """Loads the value from the storage."""

    @classmethod
    @abstractmethod
    def save(cls, data: T, filesystem: FS, feature_path: str, compress_level: int = 0) -> None:
        """Method for saving a feature. The path is assumed to be filesystem path but without file extensions.

        Example of path is `eopatch/data/NDVI`, which is then used to save `eopatch/data/NDVI.npy.gz`.
        """


class FeatureIOGZip(FeatureIO[T], metaclass=ABCMeta):
    """A class that handles the saving and loading process of a single feature at a given location.

    Uses GZip to compress files when required.
    """

    def _check_path_extension(self, path: str) -> None:
        filename = fs.path.basename(path)
        expected_extension = f".{self.get_file_extension()}"
        compressed_extension = expected_extension + f".{MimeType.GZIP.extension}"
        if not filename.endswith((expected_extension, compressed_extension)):
            raise ValueError(f"FeatureIO expects a filepath with the {expected_extension} file extension, got {path}")

    def _load_value(self) -> T:
        with self.filesystem.openbin(self.path, "r") as file_handle:
            if MimeType.GZIP.matches_extension(self.path):
                with gzip.open(file_handle, "rb") as gzip_fp:
                    return self._read_from_file(gzip_fp)
            else:
                return self._read_from_file(file_handle)

    @abstractmethod
    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> T:
        """Loads from a file and decodes content."""

    @classmethod
    def save(cls, data: T, filesystem: FS, feature_path: str, compress_level: int = 0) -> None:
        """Method for saving a feature. The path is assumed to be filesystem path but without file extensions.

        Example of path is `eopatch/data/NDVI`, which is then used to save `eopatch/data/NDVI.npy.gz`.

        To minimize the chance of corrupted files (in case of OSFS and TempFS) the file is first written and then moved
        to correct location. If any exceptions happen during the writing process the file is not moved (and old one not
        overwritten).
        """
        gz_extension = ("." + MimeType.GZIP.extension) if compress_level else ""
        path = f"{feature_path}.{cls.get_file_extension()}{gz_extension}"

        if isinstance(filesystem, (OSFS, TempFS)):
            with TempFS(temp_dir=filesystem.root_path) as tempfs:
                cls._save(data, tempfs, "tmp_feature", compress_level)
                if fs.__version__ == "2.4.16" and filesystem.exists(path):  # An issue in the fs version
                    filesystem.remove(path)
                fs.move.move_file(tempfs, "tmp_feature", filesystem, path)
            return
        cls._save(data, filesystem, path, compress_level)

    @classmethod
    def _save(cls, data: T, filesystem: FS, path: str, compress_level: int) -> None:
        """Given a filesystem it saves and compresses the data."""
        with filesystem.openbin(path, "w") as file:
            if compress_level == 0:
                cls._write_to_file(data, file, path)
            else:
                with gzip.GzipFile(fileobj=file, compresslevel=compress_level, mode="wb") as gzip_file:
                    cls._write_to_file(data, gzip_file, path)

    @classmethod
    @abstractmethod
    def _write_to_file(cls, data: T, file: BinaryIO | gzip.GzipFile, path: str) -> None:
        """Writes data to a file in the appropriate way."""


class FeatureIONumpy(FeatureIOGZip[np.ndarray]):
    """FeatureIO object specialized for Numpy arrays."""

    @classmethod
    def get_file_extension(cls) -> str:
        return MimeType.NPY.extension

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> np.ndarray:
        return np.load(file, allow_pickle=True)

    @classmethod
    def _write_to_file(cls, data: np.ndarray, file: BinaryIO | gzip.GzipFile, _: str) -> None:
        return np.save(file, data)


class FeatureIOGeoDf(FeatureIOGZip[gpd.GeoDataFrame]):
    """FeatureIO object specialized for GeoDataFrames."""

    @classmethod
    def get_file_extension(cls) -> str:
        return MimeType.GPKG.extension

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> gpd.GeoDataFrame:
        dataframe = gpd.read_file(file)

        if dataframe.crs is not None:
            # Trying to preserve a standard CRS and passing otherwise
            with contextlib.suppress(ValueError), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SHUserWarning)
                dataframe.crs = CRS(dataframe.crs).pyproj_crs()

        if TIMESTAMP_COLUMN in dataframe:
            dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(dataframe[TIMESTAMP_COLUMN])

        return dataframe

    @classmethod
    def _write_to_file(cls, data: gpd.GeoDataFrame, file: BinaryIO | gzip.GzipFile, path: str) -> None:
        layer = fs.path.basename(path)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are attempting to write an empty DataFrame to file*",
                category=UserWarning,
            )
            return data.to_file(file, driver="GPKG", encoding="utf-8", layer=layer, index=False)


class FeatureIOJson(FeatureIOGZip[T]):
    """FeatureIO object specialized for JSON-like objects."""

    @classmethod
    def get_file_extension(cls) -> str:
        return MimeType.JSON.extension

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> T:
        return json.load(file)

    @classmethod
    def _write_to_file(cls, data: T, file: BinaryIO | gzip.GzipFile, path: str) -> None:
        try:
            json_data = json.dumps(data, indent=2, default=_better_jsonify)
        except TypeError as exception:
            raise TypeError(
                f"Failed to serialize when saving JSON file to {path}. Make sure that this feature type "
                "contains only JSON serializable Python types before attempting to serialize it."
            ) from exception

        file.write(json_data.encode())


class FeatureIOTimestamps(FeatureIOJson[List[datetime.datetime]]):
    """FeatureIOJson object specialized for List[dt.datetime]."""

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> list[datetime.datetime]:
        data = json.load(file)
        return [dateutil.parser.parse(timestamp) for timestamp in data]


class FeatureIOBBox(FeatureIOGZip[BBox]):
    """FeatureIO object specialized for BBox objects."""

    @classmethod
    def get_file_extension(cls) -> str:
        return MimeType.GEOJSON.extension

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> BBox:
        json_data = json.load(file)
        return Geometry.from_geojson(json_data).bbox

    @classmethod
    def _write_to_file(cls, data: BBox, file: BinaryIO | gzip.GzipFile, _: str) -> None:
        json_data = json.dumps(data.geojson, indent=2)
        file.write(json_data.encode())


class FeatureIOZarr(FeatureIO[np.ndarray]):
    """FeatureIO object specialized for Zarr arrays."""

    @classmethod
    def get_file_extension(cls) -> str:
        return "zarr"

    def _load_value(self) -> np.ndarray:
        self._check_dependencies_imported(self.path)

        store = self._get_mapping(self.path, self.filesystem)
        zarray = zarr.open_array(store=store, mode="r+")
        return zarray[...]

    @staticmethod
    def _get_mapping(path: str, filesystem: FS) -> MutableMapping:
        if isinstance(filesystem, OSFS):
            abs_path = filesystem.getsyspath(path)
            return zarr.DirectoryStore(abs_path)
        if isinstance(filesystem, TempFS):
            abs_path = filesystem.getsyspath(path)
            return zarr.TempStore(abs_path)
        if isinstance(filesystem, fs_s3fs.S3FS):
            fsspec_filesystem = s3fs.S3FileSystem(
                key=filesystem.aws_access_key_id,
                secret=filesystem.aws_secret_access_key,
                token=filesystem.aws_session_token,
            )
            abs_path = f"{filesystem._bucket_name}/{path}"  # noqa: SLF001  #pylint: disable=protected-access
            return s3fs.S3Map(abs_path, s3=fsspec_filesystem)
        raise ValueError(f"Cannot handle filesystem {filesystem}")

    @classmethod
    def save(cls, data: np.ndarray, filesystem: FS, feature_path: str, compress_level: int = 0) -> None:  # noqa: ARG003
        cls._check_dependencies_imported(feature_path)

        path = f"{feature_path}.{cls.get_file_extension()}"
        store = cls._get_mapping(path, filesystem)
        zarr.save(store, data)

    @staticmethod
    def _check_dependencies_imported(path: str) -> None:
        if not all(dep in sys.modules for dep in ["zarr", "s3fs"]):
            msg = f"Encountered use of Zarr for {path}, but missing dependencies. Please install `zarr` and `s3fs`."
            raise ImportError(msg)


def _better_jsonify(param: object) -> Any:
    """Adds the option to serialize datetime.date and FeatureDict objects via isoformat."""
    if isinstance(param, datetime.date):
        return param.isoformat()
    if isinstance(param, Mapping):
        return dict(param.items())
    raise TypeError(f"Object of type {type(param)} is not yet supported in jsonify utility function")


def _get_feature_io_constructor(ftype: FeatureType, use_zarr: bool) -> type[FeatureIO]:
    """Creates the correct FeatureIO, corresponding to the FeatureType."""
    if ftype is FeatureType.BBOX:
        return FeatureIOBBox
    if ftype is FeatureType.META_INFO:
        return FeatureIOJson
    if ftype is FeatureType.TIMESTAMPS:
        return FeatureIOTimestamps
    if ftype.is_vector():
        return FeatureIOGeoDf
    return FeatureIOZarr if use_zarr else FeatureIONumpy
