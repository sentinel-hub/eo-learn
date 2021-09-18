"""
A module implementing utilities for working with different filesystems

Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os
from pathlib import Path, PurePath

import fs
from fs_s3fs import S3FS

from sentinelhub import SHConfig


def get_filesystem(path, create=False, config=None, **kwargs):
    """ A utility function for initializing any type of filesystem object with PyFilesystem2 package

    :param path: A filesystem path
    :type path: str
    :param create: If the filesystem path doesn't exist this flag indicates to either create it or raise an error
    :type create: bool
    :param config: A configuration object with AWS credentials
    :type config: SHConfig
    :param kwargs: Any keyword arguments to be passed forward
    :return: A filesystem object
    :rtype: fs.FS
    """
    if isinstance(path, Path):
        path = str(path)

    if path.startswith('s3://'):
        return load_s3_filesystem(path, config=config, **kwargs)

    return fs.open_fs(path, create=create, **kwargs)


def get_base_filesystem_and_path(*path_parts, **kwargs):
    """ Parses multiple strings that define a filesystem path and returns a filesystem object with a relative path
    on the filesystem

    :param path_parts: One or more strings defining a filesystem path
    :type path_parts: str
    :param kwargs: Parameters passed to get_filesystem function
    :return: A filesystem object and a relative path
    :rtype: (fs.FS, str)
    """
    path_parts = [str(part) for part in path_parts if part is not None]
    base_path = path_parts[0]

    if '://' in base_path:
        base_path_parts = base_path.split('/', 3)
        filesystem_path = '/'.join(base_path_parts[:-1])
        relative_path = '/'.join([base_path_parts[-1], *path_parts[1:]])

        return get_filesystem(filesystem_path, **kwargs), relative_path

    entire_path = os.path.abspath(os.path.join(*path_parts))
    pure_path = PurePath(entire_path)
    posix_path = pure_path.relative_to(pure_path.anchor).as_posix()
    filesystem_path = base_path.split('\\')[0] if '\\' in base_path else '/'

    return get_filesystem(filesystem_path, **kwargs), posix_path


def load_s3_filesystem(path, strict=False, config=None):
    """ Loads AWS s3 filesystem from a path

    :param path: A path to a folder on s3 bucket that will be the base folder in this filesystem
    :type path: str
    :param strict: If `True` the filesystem will be making additional checks to the s3. Default is `False`.
    :type strict: bool
    :param config: A configuration object with AWS credentials. By default is set to None and in this case the default
        configuration will be taken.
    :type config: SHConfig or None
    :return: A S3 filesystem object
    :rtype: fs_s3fs.S3FS
    """
    if not path.startswith('s3://'):
        raise ValueError(f"AWS path has to start with s3:// but found '{path}'")

    if config is None:
        config = SHConfig()

    path_chunks = path.split('/', 3)[2:]
    bucket_name = path_chunks[0]
    dir_path = path_chunks[1] if len(path_chunks) > 1 else '/'

    return S3FS(
        bucket_name=bucket_name,
        dir_path=dir_path,
        aws_access_key_id=config.aws_access_key_id if config.aws_access_key_id else None,
        aws_secret_access_key=config.aws_secret_access_key if config.aws_secret_access_key else None,
        strict=strict
    )
