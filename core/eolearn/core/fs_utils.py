"""
A module implementing utilities for working with different filesystems

Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import fs
from fs_s3fs import S3FS

from sentinelhub import SHConfig


def get_filesystem(path, create=False, **kwargs):
    """ A utility function for initializing any type of filesystem object with PyFilesystem2 package

    :param path: A filesystem path
    :type path: str
    :param create: If the filesystem path doesn't exist this flag indicates to either create it or raise an error
    :type create: bool
    :param kwargs: Any keyword arguments to be passed forward
    :return: A filesystem object
    :rtype: fs.FS
    """
    if path.startswith('s3://'):
        return load_s3_filesystem(path, *kwargs)

    return fs.open_fs(path, create=create, **kwargs)


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
        raise ValueError("AWS path has to start with s3:// but found '{}'".format(path))

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
