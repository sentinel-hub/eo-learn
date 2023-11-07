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
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
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
from sentinelhub.exceptions import SHUserWarning

from .constants import TIMESTAMP_COLUMN, FeatureType, OverwritePermission
from .exceptions import EODeprecationWarning
from .types import EllipsisType, Feature, FeaturesSpecification
from .utils.fs import get_full_path, split_all_extensions
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
    Optional[BBox],
    Optional[List[datetime.datetime]],
    Dict[Tuple[FeatureType, str], "FeatureIO"],
]
Features: TypeAlias = List[Tuple[FeatureType, str]]


BBOX_FILENAME = "bbox"
TIMESTAMPS_FILENAME = "timestamps"


@dataclass
class FilesystemDataInfo:
    """Information about data that is present on the filesystem. Fields represent paths to relevant file."""

    timestamps: str | None = None
    bbox: str | None = None
    old_meta_info: str | None = None
    features: dict[FeatureType, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))

    def iterate_features(self) -> Iterator[tuple[Feature, str]]:
        """Yields `(ftype, fname), path` tuples from `features`."""
        for ftype, ftype_dict in self.features.items():
            for fname, path in ftype_dict.items():
                yield (ftype, fname), path


@dataclass
class TemporalSelection:
    """Defines which temporal slices a temporally-partial EOPatch represents. Also carries info on how to initialize
    Zarr arrays if inferred."""

    selection: None | slice | list[int]
    full_size: int | None  # None means unknown


def save_eopatch(
    eopatch: EOPatch,
    filesystem: FS,
    patch_location: str,
    *,
    features: FeaturesSpecification,
    save_timestamps: bool | Literal["auto"],
    overwrite_permission: OverwritePermission,
    use_zarr: bool,
    temporal_selection: None | slice | list[int] | Literal["infer"],
) -> None:
    """A utility function used by `EOPatch.save` method."""
    patch_exists = filesystem.exists(patch_location)

    if temporal_selection is not None and not use_zarr:
        raise ValueError("Cannot perform partial saving via `temporal_selection` without zarr arrays.")

    eopatch_features = FeatureParser(features).get_features(eopatch)
    file_information = get_filesystem_data_info(filesystem, patch_location) if patch_exists else FilesystemDataInfo()

    _check_collisions(overwrite_permission, eopatch_features, file_information)

    if save_timestamps == "auto":
        save_timestamps = (
            features is ... or any(ftype.is_temporal() for ftype, _ in eopatch_features)
        ) and temporal_selection is None

    # Data must be collected before any tinkering with files due to lazy-loading
    data_for_saving = list(
        _yield_savers(
            eopatch=eopatch,
            features=eopatch_features,
            patch_location=patch_location,
            filesystem=filesystem,
            use_zarr=use_zarr,
            save_timestamps=save_timestamps,
            temporal_selection=_infer_temporal_selection(temporal_selection, filesystem, file_information, eopatch),
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=EODeprecationWarning)
        if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
            _remove_old_eopatch(filesystem, patch_location)

    filesystem.makedirs(patch_location, recreate=True)
    for ftype in {ftype for ftype, _ in eopatch_features}:
        filesystem.makedirs(fs.path.join(patch_location, ftype.value), recreate=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_files = list(executor.map(lambda saver: saver(), data_for_saving))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=EODeprecationWarning)
        if overwrite_permission is not OverwritePermission.OVERWRITE_PATCH:
            _remove_redundant_files(filesystem, new_files, file_information)

    _remove_old_style_metainfo(filesystem, patch_location, new_files, file_information)


def _infer_temporal_selection(
    temporal_selection: None | slice | list[int] | Literal["infer"],
    filesystem: FS,
    file_information: FilesystemDataInfo,
    eopatch: EOPatch,
) -> TemporalSelection:
    if temporal_selection != "infer":
        return TemporalSelection(temporal_selection, None)

    patch_timestamps = eopatch.get_timestamps("Cannot infer temporal selection. EOPatch to be saved has no timestamps.")
    if file_information.timestamps is None:
        raise OSError("Cannot infer temporal selection. Saved EOPatch does not have timestamps.")
    full_timestamps = FeatureIOTimestamps(file_information.timestamps, filesystem).load()
    timestamp_indices = {timestamp: idx for idx, timestamp in enumerate(full_timestamps)}
    if not all(timestamp in timestamp_indices for timestamp in patch_timestamps):
        raise ValueError(
            f"Cannot infer temporal selection. EOPatch timestamps {patch_timestamps} are not a subset of the stored"
            f" EOPatch timestamps {full_timestamps}."
        )
    return TemporalSelection([timestamp_indices[timestamp] for timestamp in patch_timestamps], len(full_timestamps))


def _remove_old_eopatch(filesystem: FS, patch_location: str) -> None:
    filesystem.removetree(patch_location)
    filesystem.makedirs(patch_location, recreate=True)


def _yield_savers(
    *,
    eopatch: EOPatch,
    features: Features,
    patch_location: str,
    filesystem: FS,
    save_timestamps: bool,
    use_zarr: bool,
    temporal_selection: TemporalSelection,
) -> Iterator[Callable[[], str]]:
    """Prepares callables that save the data and return the path to where the data was saved."""
    get_file_path = partial(fs.path.join, patch_location)

    if eopatch.bbox is not None:  # remove after BBox is never None
        bbox: BBox = eopatch.bbox  # mypy has problems
        yield partial(
            FeatureIOBBox.save,
            data=bbox,
            filesystem=filesystem,
            feature_path=get_file_path(BBOX_FILENAME),
            compress_level=0,
        )

    if eopatch.timestamps is not None and save_timestamps:
        path = get_file_path(TIMESTAMPS_FILENAME)
        yield partial(
            FeatureIOTimestamps.save,
            data=eopatch.timestamps,
            filesystem=filesystem,
            feature_path=path,
            compress_level=0,
        )

    for ftype, fname in features:
        io_constructor = _get_feature_io_constructor(ftype, use_zarr)
        feature_saver = partial(
            io_constructor.save,
            data=eopatch[ftype, fname],
            filesystem=filesystem,
            feature_path=get_file_path(ftype.value, fname),
            compress_level=1,
        )

        if ftype.is_temporal() and issubclass(io_constructor, FeatureIOZarr):
            feature_saver = partial(feature_saver, temporal_selection=temporal_selection)

        yield feature_saver


def _get_feature_io_constructor(ftype: FeatureType, use_zarr: bool) -> type[FeatureIO]:
    """Creates the correct FeatureIO, corresponding to the FeatureType."""
    if ftype is FeatureType.META_INFO:
        return FeatureIOJson
    if ftype.is_vector():
        return FeatureIOGeoDf
    return FeatureIOZarr if use_zarr else FeatureIONumpy  # type: ignore[return-value] # not sure why


def _remove_redundant_files(
    filesystem: FS,
    new_files: list[str],
    preexisting_files: FilesystemDataInfo,
) -> None:
    """Removes files that should have been overwritten but were not due to different file extensions."""
    feature_paths = (path for _, ftype_dict in preexisting_files.features.items() for _, path in ftype_dict.items())
    meta_paths = (preexisting_files.bbox, preexisting_files.old_meta_info, preexisting_files.timestamps)
    split_paths = map(split_all_extensions, filter(None, itertools.chain(meta_paths, feature_paths)))
    old_path_extension = dict(split_paths)  # maps {path_base: file_extension}

    files_to_remove = []
    for path in new_files:
        base, extension = split_all_extensions(path)
        if base in old_path_extension and old_path_extension[base] != extension:
            files_to_remove.append(f"{base}{old_path_extension[base]}")

    def _remover(path: str) -> None:  # Zarr path can also be path to a folder
        if not path.endswith("zarr") or filesystem.isfile(path):
            return filesystem.remove(path)
        filesystem.makedirs(path, recreate=True)  # solves issue where zarr root folder is not an actual folder on S3
        return filesystem.removetree(path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(_remover, files_to_remove))  # Wrapped in a list to get better exceptions


def _remove_old_style_metainfo(
    filesystem: FS,
    patch_location: str,
    new_files: list[str],
    preexisting_files: FilesystemDataInfo,
) -> None:
    """If any new meta-info was saved, this removes the old-style meta-info file."""
    if preexisting_files.old_meta_info is None:
        return

    if any(fs.path.relativefrom(patch_location, path).startswith("meta_info") for path in new_files):
        filesystem.remove(preexisting_files.old_meta_info)


def load_eopatch_content(
    filesystem: FS,
    patch_location: str,
    *,
    features: FeaturesSpecification,
    load_timestamps: bool | Literal["auto"],
    temporal_selection: None | slice | list[int] | Callable[[list[datetime.datetime]], list[bool]],
) -> PatchContentType:
    """A utility function used by `EOPatch.load` method."""
    err_msg = "Feature {} is specified to be loaded but does not exist in EOPatch at " + patch_location
    file_information = get_filesystem_data_info(filesystem, patch_location, features)
    feature_specs = FeatureParser(features).get_feature_specifications()

    simple_temporal_selection, maybe_timestamps = _extract_temporal_selection(
        filesystem, patch_location, file_information.timestamps, temporal_selection
    )

    features_dict = _load_features(filesystem, feature_specs, simple_temporal_selection, err_msg, file_information)

    bbox = None
    if file_information.bbox is not None:
        bbox = FeatureIOBBox(file_information.bbox, filesystem).load()

    auto_load = load_timestamps == "auto" and (
        features is ... or any(ftype.is_temporal() for ftype, _ in features_dict)
    )

    timestamps = None
    if load_timestamps is True or auto_load:
        if maybe_timestamps is not None:
            timestamps = maybe_timestamps
        elif file_information.timestamps is not None:
            timestamps = FeatureIOTimestamps(file_information.timestamps, filesystem, simple_temporal_selection).load()
        elif load_timestamps is True:  # this means that timestamps were requested but dont exist
            raise OSError(f"No timestamps found when loading EOPatch at {patch_location} with `load_timestamps=True`.")

    return bbox, timestamps, features_dict


def _extract_temporal_selection(
    filesystem: FS,
    patch_location: str,
    timestamps_path: str | None,
    temporal_selection: None | slice | list[int] | Callable[[list[datetime.datetime]], list[bool]],
) -> tuple[None | slice | list[int] | list[bool], list[datetime.datetime] | None]:
    """Extracts the temporal selection if available."""
    if not callable(temporal_selection):
        return temporal_selection, None
    if timestamps_path is not None:
        full_timestamps = FeatureIOTimestamps(timestamps_path, filesystem).load()
        simple_selection = temporal_selection(full_timestamps)
        return simple_selection, [timestamp for timestamp, include in zip(full_timestamps, simple_selection) if include]
    raise OSError(f"Cannot perform loading temporal selection, EOPatch at {patch_location} has no timestamps.")


def _load_features(
    filesystem: FS,
    feature_specs: list[tuple[FeatureType, str | EllipsisType]],
    temporal_selection: None | slice | list[int] | list[bool],
    err_msg: str,
    file_information: FilesystemDataInfo,
) -> dict[Feature, FeatureIO]:
    features_dict = {}
    if file_information.old_meta_info is not None and any(ftype is FeatureType.META_INFO for ftype, _ in feature_specs):
        msg = (
            "Stored EOPatch contains old-style meta-info file, which will no longer be supported in the future. Please"
            " re-save the EOPatch to correct this issue."
        )
        warnings.warn(msg, EODeprecationWarning, stacklevel=2)
        old_meta: dict[str, Any] = FeatureIOJson(file_information.old_meta_info, filesystem).load()
        features_dict = {(FeatureType.META_INFO, name): value for name, value in old_meta.items()}

    for ftype, fname in feature_specs:
        if ftype is FeatureType.META_INFO and (ftype, fname) in features_dict:  # data provided in old-style file
            continue

        if fname is ...:
            for fname, path in file_information.features.get(ftype, {}).items():
                features_dict[(ftype, fname)] = _get_feature_io(ftype, path, filesystem, temporal_selection)
        else:
            if ftype not in file_information.features or fname not in file_information.features[ftype]:
                raise OSError(err_msg.format((ftype, fname)))
            path = file_information.features[ftype][fname]
            features_dict[(ftype, fname)] = _get_feature_io(ftype, path, filesystem, temporal_selection)
    return features_dict


def _get_feature_io(
    ftype: FeatureType, path: str, filesystem: FS, temporal_selection: None | slice | list[int] | list[bool]
) -> FeatureIO:
    if ftype is FeatureType.META_INFO:
        return FeatureIOJson(path, filesystem)
    if ftype.is_vector():
        return FeatureIOGeoDf(path, filesystem)

    use_zarr = path.endswith(FeatureIOZarr.get_file_extension())

    if ftype.is_temporal():
        if use_zarr:
            return FeatureIOZarr(path, filesystem, temporal_selection)
        if temporal_selection is None:
            return FeatureIONumpy(path, filesystem)
        raise OSError(
            f"Cannot perform loading with temporal selection for numpy data at {path}. Resave feature with"
            " `use_zarr=True` to enable loading with temporal selections."
        )
    return (FeatureIOZarr if use_zarr else FeatureIONumpy)(path, filesystem)


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
                f"EOPatch at {patch_location} contains the deprecated naming `timestamp` for the `timestamps`"
                " feature. The old name will no longer be valid in the future. You can re-save the `EOPatch` to"
                " update it.",
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

        elif object_name == FeatureType.META_INFO.value and filesystem.isfile(object_path):
            result.old_meta_info = object_path

        elif FeatureType.has_value(object_name) and FeatureType(object_name) in relevant_feature_types:
            result.features[FeatureType(object_name)] = dict(walk_feature_type_folder(filesystem, object_path))

    return result


def walk_feature_type_folder(filesystem: FS, folder_path: str) -> Iterator[tuple[str, str]]:
    """Walks a feature type subfolder of EOPatch and yields tuples (feature name, path in filesystem).
    Skips folders and files in subfolders.
    """
    for path in filesystem.listdir(folder_path):
        if "/" not in path and "." in path:
            yield _remove_file_extension(path), fs.path.combine(folder_path, path)


def _check_collisions(
    overwrite_permission: OverwritePermission, eopatch_features: Features, existing_files: FilesystemDataInfo
) -> None:
    """Checks for possible name collisions to avoid unintentional overwriting."""
    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_letter_case_collisions(eopatch_features, existing_files)
        _check_add_only_permission(eopatch_features, existing_files)

    elif platform.system() == "Windows" and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES:
        _check_letter_case_collisions(eopatch_features, existing_files)

    else:
        _check_letter_case_collisions(eopatch_features, FilesystemDataInfo())


def _check_add_only_permission(eopatch_features: Features, filesystem_features: FilesystemDataInfo) -> None:
    """Checks that no existing feature will be overwritten."""
    unique_filesystem_features = {_to_lowercase(*feature) for feature, _ in filesystem_features.iterate_features()}
    unique_eopatch_features = {_to_lowercase(*feature) for feature in eopatch_features}

    intersection = unique_filesystem_features.intersection(unique_eopatch_features)
    if intersection:
        raise ValueError(f"Cannot save features {intersection} with overwrite_permission=OverwritePermission.ADD_ONLY")


def _check_letter_case_collisions(eopatch_features: Features, filesystem_features: FilesystemDataInfo) -> None:
    """Check that features have no name clashes (ignoring case) with other EOPatch features and saved features."""
    lowercase_features = {_to_lowercase(*feature) for feature in eopatch_features}

    if len(lowercase_features) != len(eopatch_features):
        raise OSError("Some features differ only in casing and cannot be saved in separate files.")

    for feature, _ in filesystem_features.iterate_features():
        if feature not in eopatch_features and _to_lowercase(*feature) in lowercase_features:
            raise OSError(
                f"There already exists a feature {feature} in the filesystem that only differs in "
                "casing from a feature that should be saved."
            )


def _to_lowercase(ftype: FeatureType, fname: str | None, *_: Any) -> tuple[FeatureType, str | None]:
    """Transforms a feature to it's lowercase representation."""
    return ftype, fname if fname is None else fname.lower()


def _remove_file_extension(path: str) -> str:
    """This also removes file extensions of form `.geojson.gz` unlike `fs.path.splitext`."""
    return split_all_extensions(path)[0]


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
        if not filename.endswith(self.get_file_extension()):
            raise ValueError(f"FeatureIO expects a filepath ending with {self.get_file_extension()}, got {path}")

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
    def save(cls, data: T, filesystem: FS, feature_path: str, compress_level: int = 0) -> str:
        """Method for saving a feature. The path is assumed to be filesystem path but without file extensions.

        Returns the full path to which the feature is saved.
        Example of path is `eopatch/data/NDVI`, which is then used to save `eopatch/data/NDVI.npy.gz`.
        """


class FeatureIOGZip(FeatureIO[T], metaclass=ABCMeta):
    """A class that handles the saving and loading process of a single feature at a given location.

    Uses GZip to compress files when required.
    """

    @classmethod
    def get_file_extension(cls, *, compress_level: int = 0) -> str:
        extension = cls._get_uncompressed_file_extension()
        return f"{extension}.gz" if compress_level > 0 else extension

    @classmethod
    @abstractmethod
    def _get_uncompressed_file_extension(cls) -> str:
        """The extension of files handled by the FeatureIO."""

    def _check_path_extension(self, path: str) -> None:
        filename = fs.path.basename(path)
        expected_extensions = (self.get_file_extension(compress_level=0), self.get_file_extension(compress_level=1))
        if not filename.endswith(expected_extensions):
            raise ValueError(f"FeatureIO expects a filepath ending with either of {expected_extensions}, got {path}")

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
    def save(cls, data: T, filesystem: FS, feature_path: str, compress_level: int = 0) -> str:
        """Method for saving a feature. The path is assumed to be filesystem path but without file extensions.

        Example of path is `eopatch/data/NDVI`, which is then used to save `eopatch/data/NDVI.npy.gz`.

        To minimize the chance of corrupted files (in case of OSFS and TempFS) the file is first written and then moved
        to correct location. If any exceptions happen during the writing process the file is not moved (and old one not
        overwritten).
        """
        path = feature_path + cls.get_file_extension(compress_level=compress_level)

        if isinstance(filesystem, (OSFS, TempFS)):
            with TempFS(temp_dir=filesystem.root_path) as tempfs:
                cls._save(data, tempfs, "tmp_feature", compress_level)
                if fs.__version__ == "2.4.16" and filesystem.exists(path):  # An issue in the fs version
                    filesystem.remove(path)
                fs.move.move_file(tempfs, "tmp_feature", filesystem, path)
        else:
            cls._save(data, filesystem, path, compress_level)
        return path

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
    def _get_uncompressed_file_extension(cls) -> str:
        return ".npy"

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> np.ndarray:
        return np.load(file, allow_pickle=True)

    @classmethod
    def _write_to_file(cls, data: np.ndarray, file: BinaryIO | gzip.GzipFile, _: str) -> None:
        return np.save(file, data)


class FeatureIOGeoDf(FeatureIOGZip[gpd.GeoDataFrame]):
    """FeatureIO object specialized for GeoDataFrames."""

    @classmethod
    def _get_uncompressed_file_extension(cls) -> str:
        return ".gpkg"

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
    def _get_uncompressed_file_extension(cls) -> str:
        return ".json"

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

    def __init__(self, path: str, filesystem: FS, temporal_selection: None | slice | list[int] | list[bool] = None):
        self.temporal_selection = slice(None) if temporal_selection is None else temporal_selection
        super().__init__(path, filesystem)

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> list[datetime.datetime]:
        data = json.load(file)
        string_timestamps = np.array(data)[self.temporal_selection]
        return [dateutil.parser.parse(raw_timestamp) for raw_timestamp in string_timestamps]


class FeatureIOBBox(FeatureIOGZip[BBox]):
    """FeatureIO object specialized for BBox objects."""

    @classmethod
    def _get_uncompressed_file_extension(cls) -> str:
        return ".geojson"

    def _read_from_file(self, file: BinaryIO | gzip.GzipFile) -> BBox:
        json_data = json.load(file)
        return Geometry.from_geojson(json_data).bbox

    @classmethod
    def _write_to_file(cls, data: BBox, file: BinaryIO | gzip.GzipFile, _: str) -> None:
        json_data = json.dumps(data.geojson, indent=2)
        file.write(json_data.encode())


class FeatureIOZarr(FeatureIO[np.ndarray]):
    """FeatureIO object specialized for Zarr arrays."""

    def __init__(self, path: str, filesystem: FS, temporal_selection: None | slice | list[int] | list[bool] = None):
        self.temporal_selection = slice(None) if temporal_selection is None else temporal_selection
        super().__init__(path, filesystem)

    @classmethod
    def get_file_extension(cls) -> str:
        return ".zarr"

    def _load_value(self) -> np.ndarray:
        self._check_dependencies_imported(self.path)

        store = self._get_mapping(self.path, self.filesystem)
        zarray = zarr.open_array(store=store, mode="r+")
        return zarray.oindex[self.temporal_selection]

    @staticmethod
    def _get_mapping(path: str, filesystem: FS) -> MutableMapping:
        abs_path = get_full_path(filesystem, path)
        if isinstance(filesystem, OSFS):
            return zarr.DirectoryStore(abs_path)
        if isinstance(filesystem, TempFS):
            return zarr.TempStore(abs_path)
        if isinstance(filesystem, fs_s3fs.S3FS):
            fsspec_filesystem = s3fs.S3FileSystem(
                key=filesystem.aws_access_key_id,
                secret=filesystem.aws_secret_access_key,
                token=filesystem.aws_session_token,
            )
            return s3fs.S3Map(abs_path, s3=fsspec_filesystem)
        raise ValueError(f"Cannot handle filesystem {filesystem} with the Zarr backend.")

    @classmethod
    def save(
        cls,
        data: np.ndarray,
        filesystem: FS,
        feature_path: str,
        compress_level: int = 0,  # noqa: ARG003
        temporal_selection: None | TemporalSelection = None,  # None means no sub-chunking (timeless)
    ) -> str:
        cls._check_dependencies_imported(feature_path)
        path = feature_path + cls.get_file_extension()
        store = cls._get_mapping(path, filesystem)
        chunk_size = data.shape if temporal_selection is None else (1, *data.shape[1:])

        if temporal_selection is None or temporal_selection.selection is None:
            zarr.save_array(store, data, chunks=chunk_size)
            return path

        selection, full_size = temporal_selection.selection, temporal_selection.full_size

        try:
            zarray = zarr.open_array(store, "r+")
        except ValueError as error:  # zarr does not expose the proper error...
            if full_size is not None:
                zarr_shape = (full_size, *data.shape[1:])
                zarray = zarr.create(zarr_shape, dtype=data.dtype, chunks=chunk_size, store=store)
            else:
                raise OSError(
                    f"Unable to open Zarr array at {path!r}. Saving with `temporal_selection` requires an initialized"
                    ' zarr array. You can also try saving with `temporal_selection="infer"`.'
                ) from error

        zarray.oindex[selection] = data
        return path

    @staticmethod
    def _check_dependencies_imported(path: str) -> None:
        if not all(dep in sys.modules for dep in ["zarr", "s3fs"]):
            msg = f"Encountered use of Zarr for {path} with missing dependencies. Please install `zarr` and `s3fs`."
            raise ImportError(msg)


def _better_jsonify(param: object) -> Any:
    """Adds the option to serialize datetime.date and FeatureDict objects via isoformat."""
    if isinstance(param, datetime.date):
        return param.isoformat()
    if isinstance(param, Mapping):
        return dict(param.items())
    raise TypeError(f"Object of type {type(param)} is not yet supported in jsonify utility function")
