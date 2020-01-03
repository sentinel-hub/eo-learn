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

import fs
from fs.tempfs import TempFS

import numpy as np

from .constants import FeatureType, FileFormat, OverwritePermission
from .utilities import FeatureParser


def save_eopatch(eopatch, filesystem, patch_location, features=..., overwrite_permission=OverwritePermission.ADD_ONLY,
                 compress_level=0):
    """ A utility function used by EOPatch.save method
    """
    patch_exists = filesystem.exists(patch_location)

    if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
        filesystem.removetree(patch_location)
        patch_exists = False

    if not patch_exists:
        filesystem.makedir(patch_location, recreate=True)

    eopatch_features = list(walk_eopatch(eopatch, patch_location, features))

    if overwrite_permission is OverwritePermission.ADD_ONLY:
        fs_features = list(walk_filesystem(filesystem, patch_location, features))
    else:
        fs_features = []

    _check_case_matching(eopatch_features, fs_features)

    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_add_only_permission(eopatch_features, fs_features)

    itr = [(ftype, fname, path) for ftype, fname, path in eopatch_features]

    ftypes = {(ftype, fs.path.dirname(path)) for ftype, _, path in itr if not ftype.is_meta()}
    for ftype, dirname in ftypes:
        if not filesystem.exists(dirname):
            filesystem.makedirs(dirname)

    for ftype, fname, path in itr:
        file_format = FileFormat.NPY if ftype.is_raster() else FileFormat.PICKLE
        patch_io = FeatureIO(filesystem, path)
        patch_io.save(eopatch[(ftype, fname)], file_format, compress_level)


def load_eopatch(eopatch, filesystem, patch_location, features=..., lazy_loading=False):
    """ A utility function used by EOPatch.load method
    """
    for ftype, fname, path in walk_filesystem(filesystem, patch_location, features):
        patch_io = FeatureIO(filesystem, path)
        eopatch[(ftype, fname)] = patch_io if lazy_loading else patch_io.load()

    return eopatch


def walk_filesystem(filesystem, patch_location, features=...):
    """ Recursively reads a patch_location and returns yields tuples of (feature_type, feature_name, file_path)
    """
    features = list(FeatureParser(features)())
    ftype_set = set(ftype.value for ftype, _ in features)

    paths = filesystem.listdir(patch_location)
    paths = [path for path in paths if path.split('.')[0] in ftype_set]

    for path in paths:
        if '.' not in path:
            subdir = fs.path.combine(patch_location, path)
            for file in filesystem.listdir(subdir):
                yield FeatureType(path), file.split('.')[0], fs.path.combine(subdir, file)
        else:
            yield FeatureType(path.split('.')[0]), Ellipsis, fs.path.combine(patch_location, path)


def walk_eopatch(eopatch, patch_location, features=...):
    """ Recursively reads a patch_location and returns yields tuples of (feature_type, feature_name, file_path)
    """
    returned_meta_features = set()
    for ftype, fname in FeatureParser(features)(eopatch):
        name_basis = fs.path.combine(patch_location, ftype.value)
        if ftype.is_meta() and ftype not in returned_meta_features:
            if eopatch[ftype]:
                yield ftype, ..., name_basis
            returned_meta_features.add(ftype)
        else:
            yield ftype, fname, fs.path.combine(name_basis, fname)


def _check_add_only_permission(eopatch_features, filesystem_features):

    filesystem_features = {_check_feature(*feature) for feature in filesystem_features}
    eopatch_features = {_check_feature(*feature) for feature in eopatch_features}

    intersection = filesystem_features.intersection(eopatch_features)
    if intersection:
        error_msg = "Cannot save features {} with overwrite_permission=OverwritePermission.ADD_ONLY "
        raise ValueError(error_msg.format(intersection))


def _check_case_matching(eopatch_features, filesystem_features):
    features = {_check_feature(*feature) for feature in eopatch_features}

    if len(features) != len(eopatch_features):
        raise IOError("Some features differ only in casing and cannot be saved in separate files.")


def _check_feature(ftype, fname, _):
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

        if isinstance(self.filesystem, fs.osfs.OSFS):
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
