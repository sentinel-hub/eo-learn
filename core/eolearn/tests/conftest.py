"""
Module with global fixtures

Copyright (c) 2017- Sinergise and contributors

For the full list of contributors, see the CREDITS file in the root directory of this source tree.
This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import os
from typing import Callable

import boto3
import pytest
from fs_s3fs import S3FS
from moto import mock_s3

from eolearn.core import EOPatch


@pytest.fixture(scope="session", name="test_eopatch_path")
def test_eopatch_path_fixture() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data", "TestEOPatch")


@pytest.fixture(name="test_eopatch")
def test_eopatch_fixture(test_eopatch_path) -> EOPatch:
    return EOPatch.load(test_eopatch_path)


@pytest.fixture(name="create_mocked_s3fs", scope="session")
def s3_mocking_fixture() -> Callable[[str], S3FS]:
    """Provides a function for mocking S3 buckets"""

    @mock_s3
    def create_mocked_s3fs(bucket_name: str = "mocked-test-bucket") -> S3FS:
        """Creates a new empty mocked s3 bucket. If one such bucket already exists it deletes it first."""
        s3resource = boto3.resource("s3", region_name="eu-central-1")

        bucket = s3resource.Bucket(bucket_name)

        if bucket.creation_date:  # If bucket already exists
            for key in bucket.objects.all():
                key.delete()
            bucket.delete()

        s3resource.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})

        return S3FS(bucket_name=bucket_name)

    return create_mocked_s3fs
