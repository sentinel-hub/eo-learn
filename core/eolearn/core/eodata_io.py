"""
A module implementing utilities for working with saving and loading EOPatch data

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import concurrent.futures
import gzip
import json
import pickle
import warnings
from collections import defaultdict

import fs
import fs.move
import geopandas as gpd
import numpy as np
import pandas as pd
from fs.base import FS
from fs.osfs import OSFS
from fs.tempfs import TempFS
from geopandas import GeoDataFrame, GeoSeries

from sentinelhub import CRS, Geometry, MimeType
from sentinelhub.exceptions import SHUserWarning
from sentinelhub.os_utils import sys_is_windows

from .constants import FeatureType, OverwritePermission
from .exceptions import EODeprecationWarning
from .utils.parsing import FeatureParser
from .utils.vector_io import infer_schema


def save_eopatch(
    eopatch,
    filesystem,
    patch_location,
    features=...,
    overwrite_permission=OverwritePermission.ADD_ONLY,
    compress_level=0,
):
    """A utility function used by `EOPatch.save` method."""
    patch_exists = filesystem.exists(patch_location)

    if not patch_exists:
        filesystem.makedirs(patch_location, recreate=True)

    eopatch_features = list(walk_eopatch(eopatch, patch_location, features))
    fs_features = list(walk_filesystem(filesystem, patch_location))

    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_letter_case_collisions(eopatch_features, fs_features)
        _check_add_only_permission(eopatch_features, fs_features)

    elif sys_is_windows() and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES:
        _check_letter_case_collisions(eopatch_features, fs_features)

    else:
        _check_letter_case_collisions(eopatch_features, [])

    features_to_save = []
    for ftype, fname, path in eopatch_features:
        feature_io = FeatureIO(ftype, path, filesystem)
        data = eopatch[(ftype, fname)]

        features_to_save.append((feature_io, data, ftype.file_format(), compress_level))

    # Cannot be done before due to lazy loading (this would delete the files before the data is loaded)
    if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
        filesystem.removetree(patch_location)
        if patch_location != "/":  # avoid redundant filesystem.makedirs if the location is '/'
            filesystem.makedirs(patch_location, recreate=True)

    ftype_folders = {fs.path.dirname(path) for ftype, _, path in eopatch_features if not ftype.is_meta()}
    for folder in ftype_folders:
        if not filesystem.exists(folder):
            filesystem.makedirs(folder, recreate=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The following is intentionally wrapped in a list in order to get back potential exceptions
        list(executor.map(lambda params: params[0].save(*params[1:]), features_to_save))

    if overwrite_permission is not OverwritePermission.OVERWRITE_PATCH:
        remove_redundant_files(filesystem, eopatch_features, fs_features, compress_level)


def remove_redundant_files(filesystem, eopatch_features, filesystem_features, current_compress_level):
    """Removes files that should have been overwritten but were not due to different compression levels."""
    files_to_remove = []
    saved_features = {(ftype, fname) for ftype, fname, _ in eopatch_features}
    for ftype, fname, path in filesystem_features:
        different_compression = MimeType.GZIP.matches_extension(path) != (current_compress_level > 0)
        if (ftype, fname) in saved_features and different_compression:
            files_to_remove.append(path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The following is intentionally wrapped in a list in order to get back potential exceptions
        list(executor.map(filesystem.remove, files_to_remove))


def load_eopatch(eopatch, filesystem, patch_location, features=..., lazy_loading=False):
    """A utility function used by `EOPatch.load` method."""
    features = list(walk_filesystem(filesystem, patch_location, features))
    loading_data = [FeatureIO(ftype, path, filesystem) for ftype, _, path in features]

    if not lazy_loading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loading_data = executor.map(lambda loader: loader.load(), loading_data)

    for (ftype, fname, _), value in zip(features, loading_data):
        eopatch[(ftype, fname)] = value

    return eopatch


def walk_filesystem(filesystem, patch_location, features=...):
    """Recursively reads a patch_location and yields tuples of (feature_type, feature_name, file_path)."""
    existing_features = defaultdict(dict)
    for ftype, fname, path in walk_main_folder(filesystem, patch_location):
        existing_features[ftype][fname] = path

    returned_meta_features = set()
    queried_features = set()
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


def walk_main_folder(filesystem, folder_path):
    """Walks the main EOPatch folders and yields tuples (feature type, feature name, path in filesystem).

    The results depend on the implementation of `filesystem.listdir`. For each folder that coincides with a feature
    type it returns (feature type, ..., path). If files in subfolders are also listed by `listdir` it returns
    them as well, which allows `walk_filesystem` to skip such subfolders from further searches.
    """
    for path in filesystem.listdir(folder_path):
        raw_path = path.split(".")[0].strip("/")

        if "/" in raw_path:  # For cases where S3 does not have a regular folder structure
            ftype_str, fname = fs.path.split(raw_path)
        else:
            ftype_str, fname = raw_path, ...

        if FeatureType.has_value(ftype_str):
            yield FeatureType(ftype_str), fname, fs.path.combine(folder_path, path)


def walk_feature_type_folder(filesystem, folder_path):
    """Walks a feature type subfolder of EOPatch and yields tuples (feature name, path in filesystem).
    Skips folders and files in subfolders.
    """
    for path in filesystem.listdir(folder_path):
        if "/" not in path and "." in path:
            yield path.split(".")[0], fs.path.combine(folder_path, path)


def walk_eopatch(eopatch, patch_location, features):
    """Yields tuples of (feature_type, feature_name, file_path), with file_path being the expected file path."""
    returned_meta_features = set()
    for ftype, fname in FeatureParser(features).get_features(eopatch):
        name_basis = fs.path.combine(patch_location, ftype.value)
        if ftype.is_meta():
            # META_INFO features are yielded separately by FeatureParser. We only yield them once with `...`,
            # because all META_INFO is saved together
            if eopatch[ftype] and ftype not in returned_meta_features:
                yield ftype, ..., name_basis
                returned_meta_features.add(ftype)
        else:
            yield ftype, fname, fs.path.combine(name_basis, fname)


def _check_add_only_permission(eopatch_features, filesystem_features):
    """Checks that no existing feature will be overwritten."""
    filesystem_features = {_to_lowercase(*feature) for feature in filesystem_features}
    eopatch_features = {_to_lowercase(*feature) for feature in eopatch_features}

    intersection = filesystem_features.intersection(eopatch_features)
    if intersection:
        raise ValueError(f"Cannot save features {intersection} with overwrite_permission=OverwritePermission.ADD_ONLY")


def _check_letter_case_collisions(eopatch_features, filesystem_features):
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


def _to_lowercase(ftype, fname, *_):
    """Transforms a feature to it's lowercase representation."""
    return ftype, fname if fname is ... else fname.lower()


class FeatureIO:
    """A class that handles the saving and loading process of a single feature at a given location."""

    def __init__(self, feature_type: FeatureType, path: str, filesystem: FS):
        """
        :param feature_type: A feature type
        :param path: A path in the filesystem
        :param filesystem: A filesystem object
        """
        self.feature_type = feature_type
        self.path = path
        self.filesystem = filesystem

        self.loaded_value = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"

    def load(self):
        """Method for loading a feature. The loaded value is stored into an attribute in case a second load request is
        triggered inside a shallow-copied EOPatch.
        """
        if self.loaded_value is not None:
            return self.loaded_value

        with self.filesystem.openbin(self.path, "r") as file_handle:
            if MimeType.GZIP.matches_extension(self.path):
                path = fs.path.splitext(self.path)[0]
                with gzip.open(file_handle, "rb") as gzip_fp:
                    self.loaded_value = self._decode(gzip_fp, path)
            else:
                self.loaded_value = self._decode(file_handle, self.path)

        return self.loaded_value

    def save(self, data, file_format, compress_level=0):
        """Method for saving a feature."""
        gz_extension = ("." + MimeType.GZIP.extension) if compress_level else ""
        path = f"{self.path}.{file_format.extension}{gz_extension}"

        if isinstance(self.filesystem, (OSFS, TempFS)):
            with TempFS(temp_dir=self.filesystem.root_path) as tempfs:
                self._save(tempfs, data, "tmp_feature", file_format, compress_level)
                if fs.__version__ == "2.4.16" and self.filesystem.exists(path):  # An issue in the fs version
                    self.filesystem.remove(path)
                fs.move.move_file(tempfs, "tmp_feature", self.filesystem, path)
            return
        self._save(self.filesystem, data, path, file_format, compress_level)

    def _save(self, filesystem, data, path, file_format, compress_level=0):
        """Given a filesystem it saves and compresses the data."""
        with filesystem.openbin(path, "w") as file_handle:
            if compress_level == 0:
                self._write_to_file(data, file_handle, file_format)
                return

            with gzip.GzipFile(fileobj=file_handle, compresslevel=compress_level, mode="wb") as gzip_file_handle:
                self._write_to_file(data, gzip_file_handle, file_format)

    def _write_to_file(self, data, file, file_format):
        """Writes data to a file in the appropriate way."""
        if file_format is MimeType.NPY:
            return np.save(file, data)

        if file_format is MimeType.GPKG:
            # Temporary workaround until GeoPandas 0.11 is released
            layer = fs.path.basename(self.path)
            try:
                return data.to_file(file, driver="GPKG", encoding="utf-8", layer=layer)
            except ValueError as err:
                if data.empty:
                    schema = infer_schema(data)
                    return data.to_file(file, driver="GPKG", encoding="utf-8", layer=layer, schema=schema)
                raise err

        if file_format in [MimeType.JSON, MimeType.GEOJSON]:
            if self.feature_type is FeatureType.BBOX:
                data = data.geojson

            if self.feature_type is FeatureType.TIMESTAMP:
                data = [timestamp.isoformat() for timestamp in data]

            try:
                json_data = json.dumps(data, indent=2)
            except TypeError as exception:
                raise TypeError(
                    f"Failed to serialize {self.feature_type} into a JSON file. Make sure that this feature type "
                    "contains only JSON serializable Python types before attempting to serialize it."
                ) from exception

            return file.write(json_data.encode())

        raise ValueError(f"Unsupported file format {file_format} for feature type {self.feature_type}.")

    def _decode(self, file, path):
        """Loads from a file and decodes content."""
        file_format = MimeType(fs.path.splitext(path)[1].strip("."))

        if file_format is MimeType.NPY:
            return np.load(file)

        if file_format is MimeType.GPKG:
            dataframe = gpd.read_file(file)

            if dataframe.crs is not None:
                # Trying to preserve a standard CRS and passing otherwise
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=SHUserWarning)
                        dataframe.crs = CRS(dataframe.crs).pyproj_crs()
                except ValueError:
                    pass

            if "TIMESTAMP" in dataframe:
                dataframe.TIMESTAMP = pd.to_datetime(dataframe.TIMESTAMP)

            return dataframe

        if file_format in [MimeType.JSON, MimeType.GEOJSON]:
            json_data = json.load(file)

            if self.feature_type is FeatureType.BBOX:
                return Geometry.from_geojson(json_data).bbox

            return json_data

        if file_format is MimeType.PICKLE:
            warnings.warn(
                f"File {self.path} with data of type {self.feature_type} is in pickle format which is deprecated "
                "since eo-learn version 1.0. Please re-save this EOPatch with the new eo-learn version to "
                "update the format. In newer versions this backward compatibility will be removed.",
                EODeprecationWarning,
            )

            data = pickle.load(file)

            # There seems to be an issue in geopandas==0.8.1 where unpickling GeoDataFrames, which were saved with an
            # old geopandas version, loads geometry column into a pandas.Series instead geopandas.GeoSeries. Because
            # of that it is missing a crs attribute which is only attached to the entire GeoDataFrame
            if isinstance(data, GeoDataFrame) and not isinstance(data.geometry, GeoSeries):
                data = data.set_geometry("geometry")

            return data
        raise ValueError(f"Unsupported data type for file {path}.")
