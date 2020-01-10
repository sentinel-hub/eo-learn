"""
Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import unittest
import logging
import tempfile

import fs
from fs.osfs import OSFS
from fs.errors import CreateFailed
from fs_s3fs import S3FS
from moto import mock_s3

from sentinelhub import SHConfig
from eolearn.core import get_filesystem, load_s3_filesystem

logging.basicConfig(level=logging.DEBUG)


class TestFilesystemUtils(unittest.TestCase):

    def test_get_local_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filesystem = get_filesystem(tmp_dir_name)
            self.assertTrue(isinstance(filesystem, OSFS))

            subfolder_path = fs.path.combine(tmp_dir_name, 'subfolder')

            with self.assertRaises(CreateFailed):
                get_filesystem(subfolder_path, create=False)

            filesystem = get_filesystem(subfolder_path, create=True)
            self.assertTrue(isinstance(filesystem, OSFS))

    @mock_s3
    def test_s3_filesystem(self):
        folder_name = 'my_folder'
        s3_url = 's3://test-eo-bucket/{}'.format(folder_name)

        filesystem = get_filesystem(s3_url)
        self.assertTrue(isinstance(filesystem, S3FS))
        self.assertEqual(filesystem.dir_path, folder_name)

        custom_config = SHConfig()
        custom_config.aws_access_key_id = 'fake-key'
        custom_config.aws_secret_access_key = 'fake-secret'
        filesystem = load_s3_filesystem(s3_url, strict=False, config=custom_config)
        self.assertTrue(isinstance(filesystem, S3FS))
        self.assertEqual(filesystem.aws_access_key_id, custom_config.aws_access_key_id)
        self.assertEqual(filesystem.aws_secret_access_key, custom_config.aws_secret_access_key)


if __name__ == '__main__':
    unittest.main()
