"""
Module with global fixtures

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import os

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

pytest.register_assert_rewrite("sentinelhub.testing_utils")  # makes asserts in helper functions work with pytest

from sentinelhub import SHConfig  # noqa[E402]

from eolearn.core import EOPatch  # noqa[E402]

EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data")
TEST_EOPATCH_PATH = os.path.join(EXAMPLE_DATA_PATH, "TestEOPatch")


@pytest.fixture(name="test_eopatch")
def test_eopatch_fixture():
    return EOPatch.load(TEST_EOPATCH_PATH, lazy_loading=True)


@pytest.fixture(name="example_data_path")
def example_data_path_fixture():
    return EXAMPLE_DATA_PATH


@pytest.fixture(name="gpkg_file")
def local_gpkg_example_file_fixture():
    """A pytest fixture to retrieve a gpkg example file"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../example_data/import-gpkg-test.gpkg")


@pytest.fixture(name="s3_gpkg_file")
def s3_gpkg_example_file_fixture():
    """A pytest fixture to retrieve a gpkg example file"""
    config = SHConfig()
    aws_config = {
        "region_name": "eu-central-1",
    }
    if config.aws_access_key_id and config.aws_secret_access_key:
        aws_config["aws_access_key_id"] = config.aws_access_key_id
        aws_config["aws_secret_access_key"] = config.aws_secret_access_key

    client = boto3.client("s3", **aws_config)

    try:
        client.head_bucket(Bucket="eolearn-io")
    except (ClientError, NoCredentialsError):
        return pytest.skip(reason="No access to the bucket.")

    return "s3://eolearn-io/import-gpkg-test.gpkg"


@pytest.fixture(name="geodb_client")
def geodb_client_fixture():
    """A geoDB client object"""
    geodb = pytest.importorskip("xcube_geodb.core.geodb")

    client_id = os.getenv("GEODB_AUTH_CLIENT_ID")
    client_secret = os.getenv("GEODB_AUTH_CLIENT_SECRET")

    if not (client_id or client_secret):
        raise ValueError("Could not initiate geoDB client, GEODB_AUTH_CLIENT_ID and GEODB_AUTH_CLIENT_SECRET missing!")

    return geodb.GeoDBClient(client_id=client_id, client_secret=client_secret)
