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
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from sentinelhub.os_utils import sys_is_windows

from .constants import FeatureType, FileFormat, OverwritePermission
from .utilities import FeatureParser, deep_eq


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
        fs_features = list(walk_filesystem(filesystem, patch_location))
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
                raise IOError('There are no features of type {} in saved EOPatch'.format(ftype))

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


def merge_eopatch(*eopatches, features=..., time_dependent_op='concatenate', timeless_op=None):
    """ Concatenate an existing eopatch in the filesystem and a new eopatch chronologically
    """
    _check_allowed_operations(time_dependent_op, timeless_op)

    eopatch_content = {}
    for eopatch in eopatches:
        for ftype, fname in FeatureParser(features)(eopatch):
            feat = (ftype, fname)
            if ftype.is_time_dependent() and ftype not in [FeatureType.TIMESTAMP, FeatureType.VECTOR]:
                if feat not in eopatch_content.keys():
                    eopatch_content[feat] = [eopatch[feat]]
                else:
                    if eopatch[feat].shape[1:] == eopatch_content[feat][-1].shape[1:]:
                        eopatch_content[feat].append(eopatch[feat])
                    else:
                        raise ValueError(f'The arrays have mismatching n x m x b shapes for {feat}.')
            elif ftype == FeatureType.VECTOR:
                if feat not in eopatch_content.keys():
                    eopatch_content[feat] = [eopatch[feat]]
                else:
                    eopatch_content[feat].append(eopatch[feat])
            elif ftype.is_timeless() and ftype != FeatureType.VECTOR_TIMELESS:
                if feat not in eopatch_content.keys():
                    eopatch_content[feat] = [eopatch[feat]]
                else:
                    if eopatch[feat].shape == eopatch_content[feat][-1].shape:
                        if deep_eq(eopatch[feat], eopatch_content[feat][-1]) and timeless_op:
                            raise ValueError(f'Two identical timeless arrays were found for {feat}.')
                        eopatch_content[feat].append(eopatch[feat])
                    else:
                        raise ValueError(f'The arrays have mismatching n x m x b shapes for {feat}.')

    return _perform_merge_operation(eopatches, eopatch_content, time_dependent_op, timeless_op)


def _check_allowed_operations(time_dependent_op, timeless_op):
    """ Checks that no the passed operations for time-dependent and timeless feature merging are allowed.
    """
    allowed_timeless_op = [None, "mean", "max", "min", "median"]
    allowed_time_dependent_op = ["concatenate", "mean", "max", "min", "median"]
    if timeless_op not in allowed_timeless_op:
        raise ValueError('timeless_op "%s" is invalid, must be one of %s' % (timeless_op, allowed_timeless_op))
    if time_dependent_op not in allowed_time_dependent_op:
        raise ValueError('time_dependent_op "%s" is invalid, must be one of %s'
                         % (time_dependent_op, allowed_time_dependent_op))


def _perform_merge_operation(eopatches, eopatch_content, time_dependent_op, timeless_op):
    """Performs the merging of duplicate timestamps of time-dependent features and of timeless features.
    """
    ops = {'mean': np.nanmean,
           'median': np.nanmedian,
           'min': np.nanmin,
           'max': np.nanmax
           }

    all_timestamps = [tstamp for eop in eopatches for tstamp in eop.timestamp]

    timestamp_list = [eop.timestamp for eop in eopatches]
    masks = [np.isin(time_post, list(set(time_post).difference(set(time_pre))))
             for time_pre, time_post in zip(timestamp_list[:-1], timestamp_list[1:])]
    unique_timestamps = eopatches[0].timestamp + [tstamp for eop, mask in zip(eopatches[1:], masks)
                                                  for tstamp, to_keep in zip(eop.timestamp, mask) if to_keep]

    unique_indices = sorted(range(len(unique_timestamps)), key=lambda k: unique_timestamps[k])

    eopatch = eopatches[0].__copy__()
    eopatch.timestamp = sorted(unique_timestamps)

    for feat, arrays in eopatch_content.items():
        ftype, _ = feat
        if ftype.is_time_dependent() and ftype != FeatureType.VECTOR and unique_indices:
            eopatch[feat] = np.concatenate(arrays, axis=0)
            for idx, timestamp in zip(unique_indices, eopatch.timestamp):
                array = eopatch[feat][[i for i, t in enumerate(all_timestamps) if t == timestamp]]
                _check_duplicate_timestamp(array, feat)
                eopatch[feat][idx] = ops[time_dependent_op](array, axis=0) if time_dependent_op in ops else array[0]
            eopatch[feat] = eopatch[feat][unique_indices]
        elif ftype.is_timeless():
            eopatch[feat] = ops[timeless_op](arrays, axis=0) if timeless_op in ops else arrays[0]
        elif ftype == FeatureType.VECTOR:
            eopatch[feat] = pd.concat([array for array in arrays if not array.empty]).sort_values('TIMESTAMP')\
                .drop_duplicates("TIMESTAMP")

    return eopatch


def _check_duplicate_timestamp(array, feature):
    """ Checks that no duplicate timestamps with different values exist
    """
    if any(not deep_eq(array[x], array[y]) for i, x in enumerate(range(array.shape[0]))
           for j, y in enumerate(range(array.shape[0])) if i != j):
        raise ValueError(f'Two identical timestamps with different values were found for {feature}.')


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
            data = pickle.load(file)

            # There seems to be an issue in geopandas==0.8.1 where unpickling GeoDataFrames, which were saved with an
            # old geopandas version, loads geometry column into a pandas.Series instead geopandas.GeoSeries. Because
            # of that it is missing a crs attribute which is only attached to the entire GeoDataFrame
            if isinstance(data, GeoDataFrame) and not isinstance(data.geometry, GeoSeries):
                data = data.set_geometry('geometry')

            return data

        if FileFormat.NPY.extension() in path:
            return np.load(file)

        raise ValueError('Unsupported data type.')
