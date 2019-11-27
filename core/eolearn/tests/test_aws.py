"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import logging
import os
import pickle
import boto3
from moto import mock_s3

from eolearn.core import EOPatch

logging.basicConfig(level=logging.DEBUG)


class TestEOPatchAWS(unittest.TestCase):
    PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/TestEOPatchAWS')
    eop = EOPatch.load(PATCH_FILENAME)

    MY_BUCKET = "test-bucket"
    MY_PREFIX = "TestEOPatchAWS"

    @mock_s3
    def create_and_fill_s3_bucket(self):
        s3client = boto3.client('s3', region_name='eu-central-1')
        s3resource = boto3.resource('s3', region_name='eu-central-1')
        s3resource.create_bucket(Bucket=self.MY_BUCKET)
        self.eop.save_aws(bucket_name=self.MY_BUCKET, patch_location=self.MY_PREFIX, s3client=s3client)
        return s3client, s3resource

    def empty_and_delete_s3_bucket(self, s3resource):
        bucket = s3resource.Bucket(self.MY_BUCKET)
        for key in bucket.objects.all():
            key.delete()
        bucket.delete()

    @mock_s3
    def test_save_eopatch(self):
        stats_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stats/test_save_stats.pkl')
        saved_content = pickle.load(open(stats_filename, 'rb'))

        s3client, s3resource = self.create_and_fill_s3_bucket()

        response = s3client.list_objects(Bucket=self.MY_BUCKET, Prefix=self.MY_PREFIX)
        content = [x['Key'] for x in response['Contents']]

        update_stats = True
        if update_stats:
            pickle.dump(content, open(stats_filename, 'wb'))

        self.empty_and_delete_s3_bucket(s3resource)
        self.assertEqual(saved_content, content)

    @mock_s3
    def test_load_eopatch(self):
        s3client, s3resource = self.create_and_fill_s3_bucket()
        eop = EOPatch.load_aws(bucket_name=self.MY_BUCKET, patch_location=self.MY_PREFIX, s3client=s3client)

        self.empty_and_delete_s3_bucket(s3resource)
        self.assertEqual(self.eop, eop)


if __name__ == '__main__':
    unittest.main()
