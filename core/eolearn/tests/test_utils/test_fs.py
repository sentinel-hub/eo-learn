"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os
import unittest.mock as mock
from pathlib import Path

import pytest
from botocore.credentials import Credentials
from fs.errors import CreateFailed
from fs.osfs import OSFS
from fs_s3fs import S3FS
from moto import mock_s3

from sentinelhub import SHConfig

from eolearn.core import get_filesystem, load_s3_filesystem
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
    "filesystem, path, expected_full_path",
    [
        (OSFS("/tmp"), "my/folder", "/tmp/my/folder"),
        (S3FS(bucket_name="data", dir_path="/folder"), "/sub/folder", "s3://data/folder/sub/folder"),
        (S3FS(bucket_name="data"), "/sub/folder", "s3://data/sub/folder"),
    ],
)
def test_get_full_path(filesystem, path, expected_full_path):
    full_path = get_full_path(filesystem, path)
    assert full_path == expected_full_path
