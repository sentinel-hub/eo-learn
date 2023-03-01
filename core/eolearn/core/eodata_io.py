"""
A module implementing utilities for working with saving and loading EOPatch data

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import gzip
import json
import platform
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import dateutil.parser
import fs
import fs.move
import geopandas as gpd
import numpy as np
import pandas as pd
from fs.base import FS
from fs.osfs import OSFS
from fs.tempfs import TempFS

from sentinelhub import CRS, BBox, Geometry, MimeType
from sentinelhub.exceptions import SHUserWarning

from .constants import TIMESTAMP_COLUMN, FeatureType, FeatureTypeSet, OverwritePermission
from .exceptions import EODeprecationWarning
from .types import EllipsisType, FeaturesSpecification
from .utils.parsing import FeatureParser
from .utils.vector_io import infer_schema

if TYPE_CHECKING:
    from .eodata import EOPatch

T = TypeVar("T")
Self = TypeVar("Self", bound="FeatureIO")

FeatureInfo = Tuple[FeatureType, Union[str, EllipsisType], str]

BBOX_FILENAME = "bbox"


@dataclass
class FeatureFileInfo:
    """Information container for data on where a feature is (to be) saved."""

    feature: Tuple[FeatureType, str]
    relative_file_path: str

    def to_old(self) -> FeatureInfo:
        """Reverts data to old representation."""
        return (*self.feature, self.relative_file_path)


@dataclass
class MetaFeatureFileInfo:
    """Information container for data on where a feature is (to be) saved."""

    feature_type: FeatureType
    relative_file_path: str

    def to_old(self) -> FeatureInfo:
        """Reverts data to old representation."""
        return (self.feature_type, ..., self.relative_file_path)


@dataclass
class FilesystemDataInfo:
    """Information about data that is present on the filesystem. Fields represent paths to relevant file."""

    timestamps: Optional[str]
    bbox: Optional[str]
    meta_info: Optional[str]
    features: Dict[FeatureType, Dict[str, str]]


def save_eopatch(
    eopatch: EOPatch,
    filesystem: FS,
    patch_location: str,
    features: FeaturesSpecification = ...,
    overwrite_permission: OverwritePermission = OverwritePermission.ADD_ONLY,
    compress_level: int = 0,
) -> None:
    """A utility function used by `EOPatch.save` method."""
    patch_exists = filesystem.exists(patch_location)

    files_to_save_new = list(walk_eopatch(eopatch, patch_location, features))
    files_to_save = [x.to_old() for x in files_to_save_new]
    existing_files = list(walk_filesystem(filesystem, patch_location)) if patch_exists else []

    _check_collisions(overwrite_permission, files_to_save, existing_files)

    # Data must be collected before any tinkering with files due to lazy-loading
    data_for_saving = _prepare_features_to_save(eopatch, patch_location, files_to_save)

    if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
        _remove_old_eopatch(filesystem, patch_location)

    ftype_folders = {fs.path.dirname(path) for _, _, path in files_to_save}
    for folder in ftype_folders:
        filesystem.makedirs(folder, recreate=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        save_function = partial(_save_single_feature, filesystem=filesystem, compress_level=compress_level)
        list(executor.map(save_function, data_for_saving))  # Wrapped in a list to get better exceptions

    if overwrite_permission is not OverwritePermission.OVERWRITE_PATCH:
        remove_redundant_files(filesystem, files_to_save, existing_files, compress_level)


def _check_collisions(
    overwrite_permission: OverwritePermission,
    files_to_save: Sequence[FeatureInfo],
    existing_files: Sequence[FeatureInfo],
):
    """Checks for possible name collisions to avoid unintentional overwriting."""
    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_letter_case_collisions(files_to_save, existing_files)
        _check_add_only_permission(files_to_save, existing_files)

    elif platform.system() == "Windows" and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES:
        _check_letter_case_collisions(files_to_save, existing_files)

    else:
        _check_letter_case_collisions(files_to_save, [])


def _remove_old_eopatch(filesystem: FS, patch_location: str) -> None:
    filesystem.removetree(patch_location)
    filesystem.makedirs(patch_location, recreate=True)


def _prepare_features_to_save(
    eopatch: EOPatch, patch_location: str, files_to_save: Sequence[FeatureInfo]
) -> List[Tuple[Type[FeatureIO], Any, str]]:
    """Prepares a triple `(featureIO, data, path)` so that the `featureIO` can save `data` to `path`."""
    features_to_save: List[Tuple[Type[FeatureIO], Any, str]] = [
        (FeatureIOBBox, eopatch.bbox, fs.path.combine(patch_location, BBOX_FILENAME))
    ]

    if eopatch.bbox is None:  # remove after BBox is never None
        features_to_save = []

    for ftype, fname, feature_path in files_to_save:
        if ftype == FeatureType.BBOX:  # remove after BBOX is no longer a FeatureType
            continue
        feature_io = _get_feature_io_constructor(ftype)
        data = eopatch[(ftype, fname)]

        features_to_save.append((feature_io, data, feature_path))
    return features_to_save


def _save_single_feature(save_spec: Tuple[Type[FeatureIO[T]], T, str], *, filesystem: FS, compress_level: int) -> None:
    feature_io, data, feature_path = save_spec
    feature_io.save(data, filesystem, feature_path, compress_level)


def remove_redundant_files(
    filesystem: FS,
    eopatch_features: Sequence[FeatureInfo],
    filesystem_features: Sequence[FeatureInfo],
    current_compress_level: int,
) -> None:
    """Removes files that should have been overwritten but were not due to different compression levels."""
    files_to_remove = []
    saved_features = {(ftype, fname) for ftype, fname, _ in eopatch_features}
    for ftype, fname, path in filesystem_features:
        different_compression = MimeType.GZIP.matches_extension(path) != (current_compress_level > 0)
        if (ftype, fname) in saved_features and different_compression:
            files_to_remove.append(path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(filesystem.remove, files_to_remove))  # Wrapped in a list to get better exceptions


def load_eopatch(
    eopatch: EOPatch,
    filesystem: FS,
    patch_location: str,
    features: FeaturesSpecification = ...,
    lazy_loading: bool = False,
) -> EOPatch:
    """A utility function used by `EOPatch.load` method."""
    existing_features = list(walk_filesystem(filesystem, patch_location, features))

    loading_data: Iterable[Any] = [
        _get_feature_io_constructor(ftype)(path, filesystem) for ftype, _, path in existing_features
    ]

    if not lazy_loading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loading_data = executor.map(lambda loader: loader.load(), loading_data)

    for (ftype, fname, _), value in zip(existing_features, loading_data):
        eopatch[(ftype, fname)] = value

    return eopatch


def _get_filesystem_data_info(filesystem: FS, folder_path: str, relevant_feature_types: Set[FeatureType]):
    """Gets information about files that are part of the EOPatch. Skips features that are not relevant to reduce IO."""
    result = FilesystemDataInfo(bbox=None, timestamps=None, meta_info=None, features=defaultdict(dict))

    fname: Union[str, EllipsisType]
    for path in filesystem.listdir(folder_path):
        object_name = path.split(".")[0].strip("/")
        object_path = fs.path.combine(folder_path, path)

        if "/" in object_name:  # For cases where S3 does not have a regular folder structure
            ftype_str, fname = fs.path.split(object_name)
            result.features[FeatureType(ftype_str)][fname] = path
            continue

        if object_name == "timestamp":
            warnings.warn(
                (
                    f"EOPatch at {filesystem.getsyspath(folder_path)} contains the deprecated naming `timestamp` for"
                    " the `timestamps` feature. The old name will no longer be valid in the future. You can re-save"
                    " the `EOPatch` to update it."
                ),
                category=EODeprecationWarning,
                stacklevel=2,
            )
            object_name = FeatureType.TIMESTAMPS.value

        if object_name == BBOX_FILENAME:
            result.bbox = object_path

        elif object_name == FeatureType.TIMESTAMPS.value:
            result.timestamps = object_path

        elif object_name == FeatureType.META_INFO.value:
            result.meta_info = object_path

        elif FeatureType.has_value(object_name) and FeatureType(object_name) in relevant_feature_types:
            result.features[FeatureType(object_name)] = dict(walk_feature_type_folder(filesystem, object_path))

    return result


def _filesystem_info_to_feature_info(
    fsinfo: FilesystemDataInfo, relevant_feature_types: Set[FeatureType]
) -> Iterator[FeatureInfo]:
    if fsinfo.bbox and FeatureType.BBOX in relevant_feature_types:
        yield (FeatureType.BBOX, ..., fsinfo.bbox)
    if fsinfo.timestamps and FeatureType.TIMESTAMPS in relevant_feature_types:
        yield (FeatureType.TIMESTAMPS, ..., fsinfo.timestamps)
    if fsinfo.meta_info and FeatureType.META_INFO in relevant_feature_types:
        yield (FeatureType.META_INFO, ..., fsinfo.meta_info)

    for ftype, fnames in fsinfo.features.items():
        yield from ((ftype, fname, fpath) for fname, fpath in fnames.items())


def walk_filesystem(
    filesystem: FS, patch_location: str, features: FeaturesSpecification = ...
) -> Iterator[FeatureInfo]:
    relevant_features = FeatureParser(features).get_feature_specifications()
    relevant_feature_types = {ftype for ftype, _ in relevant_features}
    fsinfo = _get_filesystem_data_info(filesystem, patch_location, relevant_feature_types)

    for ftype, fname in FeatureParser(features).get_feature_specifications():
        if fname is not ...:
            if ftype == FeatureType.META_INFO and fsinfo.meta_info is None:
                raise IOError("There are no META_INFO features in saved EOPatch")
            if not ftype.is_meta() and fname not in fsinfo.features.get(ftype, {}):
                raise IOError(f"Feature {(ftype, fname)} does not exist in saved EOPatch")

    return _filesystem_info_to_feature_info(fsinfo, relevant_feature_types)


def old_walk_filesystem(
    filesystem: FS, patch_location: str, features: FeaturesSpecification = ...
) -> Iterator[FeatureInfo]:
    """Recursively reads a patch_location and yields tuples of (feature_type, feature_name, file_path)."""
    existing_features: DefaultDict[FeatureType, Dict[Union[str, EllipsisType], str]] = defaultdict(dict)
    for ftype, fname, path in walk_main_folder(filesystem, patch_location):
        existing_features[ftype][fname] = path

    returned_meta_features = set()
    queried_features = set()
    feature_name: Union[str, EllipsisType]
    for ftype, fname in FeatureParser(features).get_feature_specifications():
        if fname is ... and not existing_features[ftype]:
            continue

        if ftype.is_meta():
            if ftype in returned_meta_features:
                # Resolves META_INFO that is yielded multiple times by FeatureParser but is saved in one file
                continue
            fname = ...
            returned_meta_features.add(ftype)

        elif ftype not in queried_features and (fname is ... or fname not in existing_features[ftype]):
            # Either need to collect all features for ftype or there is a not-yet seen feature that could be collected
            queried_features.add(ftype)
            if ... not in existing_features[ftype]:
                raise IOError(f"There are no features of type {ftype} in saved EOPatch")

            for feature_name, path in walk_feature_type_folder(filesystem, existing_features[ftype][...]):
                existing_features[ftype][feature_name] = path

        if fname not in existing_features[ftype]:
            # ftype has already been fully collected, but the feature not found
            raise IOError(f"Feature {(ftype, fname)} does not exist in saved EOPatch")

        if fname is ... and not ftype.is_meta():
            for feature_name, path in existing_features[ftype].items():
                if feature_name is not ...:
                    yield ftype, feature_name, path
        else:
            yield ftype, fname, existing_features[ftype][fname]


def walk_main_folder(filesystem: FS, folder_path: str) -> Iterator[FeatureInfo]:
    """Walks the main EOPatch folders and yields tuples (feature type, feature name, path in filesystem).

    The results depend on the implementation of `filesystem.listdir`. For each folder that coincides with a feature
    type it returns (feature type, ..., path). If files in subfolders are also listed by `listdir` it returns
    them as well, which allows `walk_filesystem` to skip such subfolders from further searches.
    """
    fname: Union[str, EllipsisType]
    for path in filesystem.listdir(folder_path):
        raw_path = path.split(".")[0].strip("/")

        if "/" in raw_path:  # For cases where S3 does not have a regular folder structure
            ftype_str, fname = fs.path.split(raw_path)
        else:
            ftype_str, fname = raw_path, ...

        if ftype_str == "timestamp":
            warnings.warn(
                (
                    f"EOPatch at {filesystem.getsyspath(folder_path)} contains the deprecated `timestamp` feature."
                    " The old name will no longer be valid in the future. You can re-save the `EOPatch` to update it."
                ),
                category=EODeprecationWarning,
                stacklevel=2,
            )
            ftype_str = FeatureType.TIMESTAMPS.value

        if FeatureType.has_value(ftype_str):
            yield FeatureType(ftype_str), fname, fs.path.combine(folder_path, path)


def walk_feature_type_folder(filesystem: FS, folder_path: str) -> Iterator[Tuple[str, str]]:
    """Walks a feature type subfolder of EOPatch and yields tuples (feature name, path in filesystem).
    Skips folders and files in subfolders.
    """
    for path in filesystem.listdir(folder_path):
        if "/" not in path and "." in path:
            yield path.split(".")[0], fs.path.combine(folder_path, path)


def walk_eopatch(
    eopatch: EOPatch, patch_location: str, features: FeaturesSpecification
) -> Iterator[Union[FeatureFileInfo, MetaFeatureFileInfo]]:
    """Yields tuples of (feature_type, feature_name, file_path), with file_path being the expected file path."""
    returned_meta_features = set()
    for ftype, fname in FeatureParser(features).get_features(eopatch):
        ftype_path = fs.path.combine(patch_location, ftype.value)
        if ftype.is_meta():
            # META_INFO features are yielded separately by FeatureParser. We only yield them once with `...`,
            # because all META_INFO is saved together
            if eopatch[ftype] and ftype not in returned_meta_features:
                yield MetaFeatureFileInfo(ftype, ftype_path)
                returned_meta_features.add(ftype)
        else:
            fname = cast(str, fname)  # name is not None for non-meta features
            yield FeatureFileInfo((ftype, fname), fs.path.combine(ftype_path, fname))


def _check_add_only_permission(
    eopatch_features: Sequence[FeatureInfo], filesystem_features: Sequence[FeatureInfo]
) -> None:
    """Checks that no existing feature will be overwritten."""
    unique_filesystem_features = {_to_lowercase(*feature) for feature in filesystem_features}
    unique_eopatch_features = {_to_lowercase(*feature) for feature in eopatch_features}

    intersection = unique_filesystem_features.intersection(unique_eopatch_features)
    if intersection:
        raise ValueError(f"Cannot save features {intersection} with overwrite_permission=OverwritePermission.ADD_ONLY")


def _check_letter_case_collisions(
    eopatch_features: Sequence[FeatureInfo], filesystem_features: Sequence[FeatureInfo]
) -> None:
    """Check that features have no name clashes (ignoring case) with other EOPatch features and saved features."""
    lowercase_features = {_to_lowercase(*feature) for feature in eopatch_features}

    if len(lowercase_features) != len(eopatch_features):
        raise IOError("Some features differ only in casing and cannot be saved in separate files.")

    original_features = {(ftype, fname) for ftype, fname, _ in eopatch_features}

    for ftype, fname, _ in filesystem_features:
        if (ftype, fname) not in original_features and _to_lowercase(ftype, fname) in lowercase_features:
            raise IOError(
                f"There already exists a feature {(ftype, fname)} in the filesystem that only differs in "
                "casing from a feature that should be saved."
            )


def _to_lowercase(
    ftype: FeatureType, fname: Union[str, EllipsisType], *_: Any
) -> Tuple[FeatureType, Union[str, EllipsisType]]:
    """Transforms a feature to it's lowercase representation."""
    return ftype, fname if fname is ... else fname.lower()


class FeatureIO(Generic[T], metaclass=ABCMeta):
    """A class that handles the saving and loading process of a single feature at a given location."""

    def __init__(self, path: str, filesystem: FS):
        """
        :param path: A path in the filesystem
        :param filesystem: A filesystem object
        :compress_level: The compression level to be used when saving, inferred from path if not provided
        """
        filename = fs.path.basename(path)
        expected_extension = f".{self.get_file_format().extension}"
        compressed_extension = expected_extension + f".{MimeType.GZIP.extension}"
        if not filename.endswith((expected_extension, compressed_extension)):
            raise ValueError(f"FeatureIO expects a filepath with the {expected_extension} file extension, got {path}")

        self.path = path
        self.filesystem = filesystem

        self.loaded_value: Optional[T] = None

    @classmethod
    @abstractmethod
    def get_file_format(cls) -> MimeType:
        """The type of files handled by the FeatureIO."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"

    def load(self) -> T:
        """Method for loading a feature. The loaded value is stored into an attribute in case a second load request is
        triggered inside a shallow-copied EOPatch.
        """
        if self.loaded_value is not None:
            return self.loaded_value

        with self.filesystem.openbin(self.path, "r") as file_handle:
            if MimeType.GZIP.matches_extension(self.path):
                with gzip.open(file_handle, "rb") as gzip_fp:
                    self.loaded_value = self._read_from_file(gzip_fp)
            else:
                self.loaded_value = self._read_from_file(file_handle)

        return self.loaded_value

    @abstractmethod
    def _read_from_file(self, file: Union[BinaryIO, gzip.GzipFile]) -> T:
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
        path = f"{feature_path}.{cls.get_file_format().extension}{gz_extension}"

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
    def _write_to_file(cls, data: T, file: Union[BinaryIO, gzip.GzipFile], path: str) -> None:
        """Writes data to a file in the appropriate way."""


class FeatureIONumpy(FeatureIO[np.ndarray]):
    """FeatureIO object specialized for Numpy arrays."""

    @classmethod
    def get_file_format(cls) -> MimeType:
        return MimeType.NPY

    def _read_from_file(self, file: Union[BinaryIO, gzip.GzipFile]) -> np.ndarray:
        return np.load(file, allow_pickle=True)

    @classmethod
    def _write_to_file(cls, data: np.ndarray, file: Union[BinaryIO, gzip.GzipFile], _: str) -> None:
        return np.save(file, data)


class FeatureIOGeoDf(FeatureIO[gpd.GeoDataFrame]):
    """FeatureIO object specialized for GeoDataFrames."""

    @classmethod
    def get_file_format(cls) -> MimeType:
        return MimeType.GPKG

    def _read_from_file(self, file: Union[BinaryIO, gzip.GzipFile]) -> gpd.GeoDataFrame:
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
    def _write_to_file(cls, data: gpd.GeoDataFrame, file: Union[BinaryIO, gzip.GzipFile], path: str) -> None:
        layer = fs.path.basename(path)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are attempting to write an empty DataFrame to file*",
                    category=UserWarning,
                )
                return data.to_file(file, driver="GPKG", encoding="utf-8", layer=layer, index=False)
        except ValueError as err:
            # This workaround is only required for geopandas<0.11.0 and will be removed in the future.
            if data.empty:
                schema = infer_schema(data)
                return data.to_file(file, driver="GPKG", encoding="utf-8", layer=layer, schema=schema)
            raise err


class FeatureIOJson(FeatureIO[T]):
    """FeatureIO object specialized for JSON-like objects."""

    @classmethod
    def get_file_format(cls) -> MimeType:
        return MimeType.JSON

    def _read_from_file(self, file: Union[BinaryIO, gzip.GzipFile]) -> T:
        return json.load(file)

    @classmethod
    def _write_to_file(cls, data: T, file: Union[BinaryIO, gzip.GzipFile], path: str) -> None:
        try:
            json_data = json.dumps(data, indent=2, default=_jsonify_timestamp)
        except TypeError as exception:
            raise TypeError(
                f"Failed to serialize when saving JSON file to {path}. Make sure that this feature type "
                "contains only JSON serializable Python types before attempting to serialize it."
            ) from exception

        file.write(json_data.encode())


class FeatureIOTimestamps(FeatureIOJson[List[datetime.datetime]]):
    """FeatureIOJson object specialized for List[dt.datetime]."""

    def _read_from_file(self, file: Union[BinaryIO, gzip.GzipFile]) -> List[datetime.datetime]:
        data = json.load(file)
        return [dateutil.parser.parse(timestamp) for timestamp in data]


class FeatureIOBBox(FeatureIO[BBox]):
    """FeatureIO object specialized for BBox objects."""

    @classmethod
    def get_file_format(cls) -> MimeType:
        return MimeType.GEOJSON

    def _read_from_file(self, file: Union[BinaryIO, gzip.GzipFile]) -> BBox:
        json_data = json.load(file)
        return Geometry.from_geojson(json_data).bbox

    @classmethod
    def _write_to_file(cls, data: BBox, file: Union[BinaryIO, gzip.GzipFile], _: str) -> None:
        json_data = json.dumps(data.geojson, indent=2)
        file.write(json_data.encode())


def _jsonify_timestamp(param: object) -> str:
    """Adds the option to serialize datetime.date objects via isoformat."""
    if isinstance(param, datetime.date):
        return param.isoformat()
    raise TypeError(f"Object of type {type(param)} is not yet supported in jsonify utility function")


def _get_feature_io_constructor(ftype: FeatureType) -> Type[FeatureIO]:
    """Creates the correct FeatureIO, corresponding to the FeatureType."""
    if ftype is FeatureType.BBOX:
        return FeatureIOBBox
    if ftype is FeatureType.META_INFO:
        return FeatureIOJson
    if ftype is FeatureType.TIMESTAMPS:
        return FeatureIOTimestamps
    if ftype in FeatureTypeSet.VECTOR_TYPES:
        return FeatureIOGeoDf
    return FeatureIONumpy
