"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import json
import os
import unittest.mock as mock
from _thread import RLock
from pathlib import Path
from typing import List

import pytest
from botocore.credentials import Credentials
from fs.base import FS
from fs.errors import CreateFailed
from fs.memoryfs import MemoryFS
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from moto import mock_s3

from sentinelhub import SHConfig

from eolearn.core import get_filesystem, load_s3_filesystem, pickle_fs, unpickle_fs
from eolearn.core.utils.fs import get_aws_credentials, get_full_path, join_path


def test_get_local_filesystem(tmp_path):
    filesystem = get_filesystem(tmp_path)
    assert isinstance(filesystem, OSFS)

    subfolder_path = os.path.join(tmp_path, "subfolder")

    with pytest.raises(CreateFailed):
        assert get_filesystem(subfolder_path, create=False)

    filesystem = get_filesystem(subfolder_path, create=True)
    assert isinstance(filesystem, OSFS)


def test_pathlib_support(tmp_path):
    path = Path(tmp_path)
    filesystem = get_filesystem(path)
    assert isinstance(filesystem, OSFS)


@mock_s3
@pytest.mark.parametrize("aws_session_token", [None, "fake-session-token"])
def test_s3_filesystem(aws_session_token):
    folder_name = "my_folder"
    s3_url = f"s3://test-eo-bucket/{folder_name}"

    filesystem = get_filesystem(s3_url)
    assert isinstance(filesystem, S3FS)
    assert filesystem.dir_path == folder_name
    assert not filesystem.strict, "Other use-cases assume that the returned S3FS is non-strict by default."

    custom_config = SHConfig()
    custom_config.aws_access_key_id = "fake-key"
    custom_config.aws_secret_access_key = "fake-secret"
    custom_config.aws_session_token = aws_session_token
    filesystem1 = load_s3_filesystem(s3_url, strict=False, config=custom_config)
    filesystem2 = get_filesystem(s3_url, config=custom_config)

    for filesystem in [filesystem1, filesystem2]:
        assert isinstance(filesystem, S3FS)
        assert filesystem.aws_access_key_id == custom_config.aws_access_key_id
        assert filesystem.aws_secret_access_key == custom_config.aws_secret_access_key
        assert filesystem.aws_session_token == aws_session_token


@pytest.mark.parametrize("s3fs_function", [get_filesystem, load_s3_filesystem])
def test_s3fs_keyword_arguments(s3fs_function):
    filesystem = s3fs_function("s3://dummy-bucket/", acl="bucket-owner-full-control")
    assert isinstance(filesystem, S3FS)
    assert filesystem.upload_args == {"ACL": "bucket-owner-full-control"}

    upload_args = {"test": "upload"}
    download_args = {"test": "download"}
    filesystem = s3fs_function("s3://dummy-bucket/", upload_args=upload_args, download_args=download_args)
    assert isinstance(filesystem, S3FS)
    assert filesystem.upload_args == upload_args
    assert filesystem.download_args == download_args


@mock.patch("eolearn.core.utils.fs.Session")
def test_get_aws_credentials(mocked_copy):
    fake_credentials = Credentials(access_key="my-aws-access-key", secret_key="my-aws-secret-key")

    mocked_copy.return_value.get_credentials.return_value = fake_credentials

    config = get_aws_credentials("xyz")
    assert config.aws_access_key_id == fake_credentials.access_key
    assert config.aws_secret_access_key == fake_credentials.secret_key

    default_config = SHConfig()
    config = get_aws_credentials("default", config=default_config)
    assert config.aws_access_key_id != default_config.aws_access_key_id
    assert config.aws_secret_access_key != default_config.aws_secret_access_key


@pytest.mark.parametrize(
    ("filesystem", "compare_params"),
    [
        (OSFS("."), ["root_path"]),
        (TempFS(identifier="test"), ["identifier", "_temp_dir"]),
        (MemoryFS(), []),
        (
            S3FS("s3://fake-bucket/", strict=False, acl="public-read"),
            ["_bucket_name", "dir_path", "strict", "upload_args"],
        ),
    ],
)
def test_filesystem_serialization(filesystem: FS, compare_params: List[str]):
    pickled_filesystem = pickle_fs(filesystem)
    assert isinstance(pickled_filesystem, bytes)

    unpickled_filesystem = unpickle_fs(pickled_filesystem)
    assert filesystem is not unpickled_filesystem
    assert isinstance(unpickled_filesystem._lock, RLock)  # noqa[SLF001]
    for param in compare_params:
        assert getattr(filesystem, param) == getattr(unpickled_filesystem, param)


def test_tempfs_serialization():
    with TempFS() as filesystem:
        pickled_filesystem = pickle_fs(filesystem)
        assert filesystem.exists("/")

        unpickled_filesystem = unpickle_fs(pickled_filesystem)
        assert filesystem.exists("/")

    assert not unpickled_filesystem.exists("/")


@mock_s3
def test_s3fs_serialization(create_mocked_s3fs):
    """Makes sure that after serialization and deserialization filesystem object can still be used for reading,
    writing, and listing objects."""
    filesystem = create_mocked_s3fs()
    filename = "file.json"
    file_content = {"test": 42}

    with filesystem.open(filename, "w") as fp:
        json.dump(file_content, fp)

    filesystem = unpickle_fs(pickle_fs(filesystem))

    assert filesystem.listdir("/") == [filename]
    with filesystem.openbin(filename, "r") as fp:
        read_content = json.load(fp)
    assert read_content == file_content

    filename2 = "file2.json"
    with filesystem.open(filename2, "w") as fp:
        json.dump({}, fp)
    assert filesystem.listdir("/") == [filename, filename2]


@pytest.mark.parametrize(
    argnames="path_parts, expected_path",
    ids=["local", "s3"],
    argvalues=[
        (["/tmp", "folder", "xyz", "..", "file.json"], os.path.join("/tmp", "folder", "file.json")),
        (["s3://xx/", "/y/z", "a", "..", "b.json"], "s3://xx/y/z/b.json"),
    ],
)
def test_join_path(path_parts, expected_path):
    assert join_path(*path_parts) == expected_path


@pytest.mark.parametrize(
    ("filesystem", "path", "expected_full_path"),
    [
        (OSFS("/tmp"), "my/folder", "/tmp/my/folder"),
        (S3FS(bucket_name="data", dir_path="/folder"), "/sub/folder", "s3://data/folder/sub/folder"),
        (S3FS(bucket_name="data"), "/sub/folder", "s3://data/sub/folder"),
    ],
)
def test_get_full_path(filesystem, path, expected_full_path):
    full_path = get_full_path(filesystem, path)
    assert full_path == expected_full_path
