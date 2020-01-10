"""
A module implementing utilities for working with saving and loading EOPatch data

Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pickle
import gzip
import concurrent.futures
from collections import defaultdict

import fs
from fs.tempfs import TempFS
import numpy as np
from sentinelhub.os_utils import sys_is_windows

from .constants import FeatureType, FileFormat, OverwritePermission
from .utilities import FeatureParser


def save_eopatch(eopatch, filesystem, patch_location, features=..., overwrite_permission=OverwritePermission.ADD_ONLY,
                 compress_level=0):
    """ A utility function used by EOPatch.save method
    """
    patch_exists = filesystem.exists(patch_location)

    if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
        filesystem.removetree(patch_location)
        if patch_location != '/':
            patch_exists = False

    if not patch_exists:
        filesystem.makedirs(patch_location)

    eopatch_features = list(walk_eopatch(eopatch, patch_location, features))

    if overwrite_permission is OverwritePermission.ADD_ONLY or \
            (sys_is_windows() and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES):
        fs_features = list(walk_filesystem(filesystem, patch_location, features))
    else:
        fs_features = []

    _check_case_matching(eopatch_features, fs_features)

    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_add_only_permission(eopatch_features, fs_features)

    ftype_folder_map = {(ftype, fs.path.dirname(path)) for ftype, _, path in eopatch_features if not ftype.is_meta()}
    for ftype, folder in ftype_folder_map:
        if not filesystem.exists(folder):
            filesystem.makedirs(folder)

    features_to_save = ((FeatureIO(filesystem, path),
                         eopatch[(ftype, fname)],
                         FileFormat.NPY if ftype.is_raster() else FileFormat.PICKLE,
                         compress_level) for ftype, fname, path in eopatch_features)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The following is intentionally wrapped in a list in order to get back potential exceptions
        list(executor.map(lambda params: params[0].save(*params[1:]), features_to_save))


def load_eopatch(eopatch, filesystem, patch_location, features=..., lazy_loading=False):
    """ A utility function used by EOPatch.load method
    """
    features = list(walk_filesystem(filesystem, patch_location, features))
    loading_data = [FeatureIO(filesystem, path) for _, _, path in features]

    if not lazy_loading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loading_data = executor.map(lambda loader: loader.load(), loading_data)

    for (ftype, fname, _), value in zip(features, loading_data):
        eopatch[(ftype, fname)] = value

    return eopatch


def walk_filesystem(filesystem, patch_location, features=...):
    """ Recursively reads a patch_location and returns yields tuples of (feature_type, feature_name, file_path)
    """
    existing_features = defaultdict(dict)
    for ftype, fname, path in walk_main_folder(filesystem, patch_location):
        existing_features[ftype][fname] = path

    returned_meta_features = set()
    queried_features = set()
    for ftype, fname in FeatureParser(features):
        if fname is ... and not existing_features[ftype]:
            continue

        if ftype.is_meta():
            if ftype in returned_meta_features:
                continue
            fname = ...
            returned_meta_features.add(ftype)

        elif ftype not in queried_features and (fname is ... or fname not in existing_features[ftype]):
            queried_features.add(ftype)
            if ... not in existing_features[ftype]:
                raise IOError('There is not {} in saved EOPatch'.format(ftype))

            for feature_name, path in walk_feature_type_folder(filesystem, existing_features[ftype][...]):
                existing_features[ftype][feature_name] = path

        if fname not in existing_features[ftype]:
            raise IOError('Feature {} does not exist in saved EOPatch'.format((ftype, fname)))

        if fname is ... and not ftype.is_meta():
            for feature_name, path in existing_features[ftype].items():
                if feature_name is not ...:
                    yield ftype, feature_name, path
        else:
            yield ftype, fname, existing_features[ftype][fname]


def walk_main_folder(filesystem, folder_path):
    """ Walks the main EOPatch folders and yields tuples (feature type, feature name, path in filesystem)
    """
    for path in filesystem.listdir(folder_path):
        raw_path = path.split('.')[0].strip('/')

        if '/' in raw_path:
            ftype_str, fname = fs.path.split(raw_path)
        else:
            ftype_str, fname = raw_path, ...

        if FeatureType.has_value(ftype_str):
            yield FeatureType(ftype_str), fname, fs.path.combine(folder_path, path)


def walk_feature_type_folder(filesystem, folder_path):
    """ Walks a feature type subfolder of EOPatch and yields tuples (feature name, path in filesystem)
    """
    for path in filesystem.listdir(folder_path):
        if '/' not in path and '.' in path:
            yield path.split('.')[0], fs.path.combine(folder_path, path)


def walk_eopatch(eopatch, patch_location, features=...):
    """ Recursively reads a patch_location and returns yields tuples of (feature_type, feature_name, file_path)
    """
    returned_meta_features = set()
    for ftype, fname in FeatureParser(features)(eopatch):
        name_basis = fs.path.combine(patch_location, ftype.value)
        if ftype.is_meta():
            if eopatch[ftype] and ftype not in returned_meta_features:
                yield ftype, ..., name_basis
                returned_meta_features.add(ftype)
        else:
            yield ftype, fname, fs.path.combine(name_basis, fname)


def _check_add_only_permission(eopatch_features, filesystem_features):
    """ Checks that no existing feature will be overwritten
    """
    filesystem_features = {_to_lowercase(*feature) for feature in filesystem_features}
    eopatch_features = {_to_lowercase(*feature) for feature in eopatch_features}

    intersection = filesystem_features.intersection(eopatch_features)
    if intersection:
        error_msg = "Cannot save features {} with overwrite_permission=OverwritePermission.ADD_ONLY "
        raise ValueError(error_msg.format(intersection))


def _check_case_matching(eopatch_features, filesystem_features):
    """ Checks that no two features in memory or in filesystem differ only by feature name casing
    """
    lowercase_features = {_to_lowercase(*feature) for feature in eopatch_features}

    if len(lowercase_features) != len(eopatch_features):
        raise IOError('Some features differ only in casing and cannot be saved in separate files.')

    original_features = {(ftype, fname) for ftype, fname, _ in eopatch_features}

    for ftype, fname, _ in filesystem_features:
        if (ftype, fname) not in original_features and _to_lowercase(ftype, fname) in lowercase_features:
            raise IOError('There already exists a feature {} in filesystem that only differs in casing from the one '
                          'that should be saved'.format((ftype, fname)))


def _to_lowercase(ftype, fname, *_):
    """ Tranforms a feature to it's lowercase representation
    """
    return ftype, fname if fname is ... else fname.lower()


class FeatureIO:
    """ A class handling saving and loading process of a single feature at a given location
    """
    def __init__(self, filesystem, path):
        """
        :param filesystem: A filesystem object
        :type filesystem: fs.FS
        :param path: A path in the filesystem
        :type path: str
        """
        self.filesystem = filesystem
        self.path = path

    def __repr__(self):
        """ A representation method
        """
        return '{}({})'.format(self.__class__.__name__, self.path)

    def load(self):
        """ Method for loading a feature
        """
        with self.filesystem.openbin(self.path, 'r') as file_handle:
            if self.path.endswith(FileFormat.GZIP.extension()):
                with gzip.open(file_handle, 'rb') as gzip_fp:
                    return self._decode(gzip_fp, self.path)

            return self._decode(file_handle, self.path)

    def save(self, data, file_format, compress_level=0):
        """ Method for saving a feature
        """
        gz_extension = FileFormat.GZIP.extension() if compress_level else ''
        path = self.path + file_format.extension() + gz_extension

        if isinstance(self.filesystem, (fs.osfs.OSFS, TempFS)):
            with TempFS(temp_dir=self.filesystem.root_path) as tempfs:
                self._save(tempfs, data, 'tmp_feature', file_format, compress_level)
                fs.move.move_file(tempfs, 'tmp_feature', self.filesystem, path)
            return
        self._save(self.filesystem, data, path, file_format, compress_level)

    def _save(self, filesystem, data, path, file_format, compress_level=0):
        """ Given a filesystem it saves and compresses the data
        """
        with filesystem.openbin(path, 'w') as file_handle:
            if compress_level == 0:
                self._write_to_file(data, file_handle, file_format)
                return

            with gzip.GzipFile(fileobj=file_handle, compresslevel=compress_level, mode='wb') as gzip_file_handle:
                self._write_to_file(data, gzip_file_handle, file_format)

    @staticmethod
    def _write_to_file(data, file, file_format):
        """ Writes to a file
        """
        if file_format is FileFormat.NPY:
            np.save(file, data)
        elif file_format is FileFormat.PICKLE:
            pickle.dump(data, file)

    @staticmethod
    def _decode(file, path):
        """ Loads from a file and decodes content
        """
        if FileFormat.PICKLE.extension() in path:
            return pickle.load(file)

        if FileFormat.NPY.extension() in path:
            return np.load(file)

        raise ValueError('Unsupported data type.')
