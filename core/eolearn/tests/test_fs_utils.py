"""
Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os
from pathlib import Path

import pytest
from fs.osfs import OSFS
from fs.errors import CreateFailed
from fs_s3fs import S3FS
from moto import mock_s3

from sentinelhub import SHConfig
from eolearn.core import get_filesystem, load_s3_filesystem


def test_get_local_filesystem(tmp_path):
    filesystem = get_filesystem(tmp_path)
    assert isinstance(filesystem, OSFS)

    subfolder_path = os.path.join(tmp_path, 'subfolder')

    with pytest.raises(CreateFailed):
        assert get_filesystem(subfolder_path, create=False)

    filesystem = get_filesystem(subfolder_path, create=True)
    assert isinstance(filesystem, OSFS)


def test_pathlib_support(tmp_path):
    path = Path(tmp_path)
    filesystem = get_filesystem(path)
    assert isinstance(filesystem, OSFS)


@mock_s3
def test_s3_filesystem():
    folder_name = 'my_folder'
    s3_url = 's3://test-eo-bucket/{}'.format(folder_name)

    filesystem = get_filesystem(s3_url)
    assert isinstance(filesystem, S3FS)
    assert filesystem.dir_path == folder_name

    custom_config = SHConfig()
    custom_config.aws_access_key_id = 'fake-key'
    custom_config.aws_secret_access_key = 'fake-secret'
    filesystem1 = load_s3_filesystem(s3_url, strict=False, config=custom_config)
    filesystem2 = get_filesystem(s3_url, config=custom_config)

    for filesystem in [filesystem1, filesystem2]:
        assert isinstance(filesystem, S3FS)
        assert filesystem.aws_access_key_id == custom_config.aws_access_key_id
        assert filesystem.aws_secret_access_key == custom_config.aws_secret_access_key
