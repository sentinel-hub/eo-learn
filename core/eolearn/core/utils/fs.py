"""
A module implementing utilities for working with different filesystems

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os
from pathlib import Path, PurePath
from typing import Optional, Tuple

import fs
from boto3 import Session
from fs.base import FS
from fs_s3fs import S3FS

from sentinelhub import SHConfig


def get_filesystem(path: str, create: bool = False, config: Optional[SHConfig] = None, **kwargs) -> FS:
    """A utility function for initializing any type of filesystem object with PyFilesystem2 package.

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

    if is_s3_path(path):
        return load_s3_filesystem(path, config=config, **kwargs)

    return fs.open_fs(path, create=create, **kwargs)


def get_base_filesystem_and_path(*path_parts: str, **kwargs) -> Tuple[FS, str]:
    """Parses multiple strings that define a filesystem path and returns a filesystem object with a relative path
    on the filesystem.

    :param path_parts: One or more strings defining a filesystem path
    :param kwargs: Parameters passed to get_filesystem function
    :return: A filesystem object and a relative path
    """
    path_parts = tuple(str(part).rstrip("/") for part in path_parts if part is not None)
    base_path = path_parts[0]

    if "://" in base_path:
        base_path_parts = base_path.split("/", 3)
        filesystem_path = "/".join(base_path_parts[:-1])
        relative_path = "/".join([base_path_parts[-1], *path_parts[1:]])

        return get_filesystem(filesystem_path, **kwargs), relative_path

    entire_path = os.path.abspath(os.path.join(*path_parts))
    pure_path = PurePath(entire_path)
    posix_path = pure_path.relative_to(pure_path.anchor).as_posix()
    filesystem_path = base_path.split("\\")[0] if "\\" in base_path else "/"

    return get_filesystem(filesystem_path, **kwargs), posix_path


def load_s3_filesystem(
    path: str, strict: bool = False, config: Optional[SHConfig] = None, aws_profile: Optional[str] = None
) -> S3FS:
    """Loads AWS s3 filesystem from a path.

    :param path: A path to a folder on s3 bucket that will be the base folder in this filesystem
    :type path: str
    :param strict: If `True` the filesystem will be making additional checks to the s3. Default is `False`.
    :type strict: bool
    :param config: A configuration object with AWS credentials. By default, is set to None and in this case the default
        configuration will be taken.
    :type config: SHConfig or None
    :param aws_profile: A name of AWS profile. If given, AWS credentials will be taken from there.
    :return: A S3 filesystem object
    :rtype: fs_s3fs.S3FS
    """
    if not is_s3_path(path):
        raise ValueError(f"AWS path has to start with s3:// but found '{path}'.")

    config = config or SHConfig()
    if aws_profile:
        config = get_aws_credentials(aws_profile, config=config)

    path_chunks = path.split("/", 3)[2:]
    bucket_name = path_chunks[0]
    dir_path = path_chunks[1] if len(path_chunks) > 1 else "/"

    return S3FS(
        bucket_name=bucket_name,
        dir_path=dir_path,
        aws_access_key_id=config.aws_access_key_id or None,
        aws_secret_access_key=config.aws_secret_access_key or None,
        aws_session_token=config.aws_session_token or None,
        strict=strict,
    )


def get_aws_credentials(aws_profile: str, config: Optional[SHConfig] = None) -> SHConfig:
    """Collects credentials from AWS profile and adds them to an instance of SHConfig.

    :param aws_profile: A name of AWS profile
    :param config: If existing config object is given credentials will be added to its copy, otherwise a new config
        object will be created.
    :return: A config object with AWS credentials that have been loaded from AWS profile.
    """
    config = config.copy() if config else SHConfig()

    aws_session = Session(profile_name=aws_profile)
    aws_credentials = aws_session.get_credentials()

    config.aws_access_key_id = aws_credentials.access_key or ""
    config.aws_secret_access_key = aws_credentials.secret_key or ""
    config.aws_session_token = aws_credentials.token or ""
    return config


def get_full_path(filesystem: FS, relative_path: str) -> str:
    """Given a filesystem object and a path, relative to the filesystem it provides a full path."""
    if isinstance(filesystem, S3FS):
        # pylint: disable=protected-access
        return join_path(f"s3://{filesystem._bucket_name}", filesystem.dir_path, relative_path)

    return os.path.normpath(filesystem.getsyspath(relative_path))


def join_path(*path_parts: str) -> str:
    """A utility function for joining a path that is either local or S3.

    :param path_parts: Partial paths where the first part will be used to decide if it is an S3 path or a local path
    :return: Joined path that is also normalized and absolute.
    """
    if is_s3_path(path_parts[0]):
        path_parts = tuple((part[5:] if index == 0 else part) for index, part in enumerate(path_parts))
        path = "/".join(part.strip("/") for part in path_parts)
        path = fs.path.normpath(path)
        return f"s3://{path}"

    # pylint: disable=no-value-for-parameter
    return os.path.abspath(os.path.join(*path_parts))


def is_s3_path(path: str) -> bool:
    """Returns True if the path points to a S3 bucket, False otherwise."""
    return path.startswith("s3://")
