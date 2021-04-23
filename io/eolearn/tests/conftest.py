"""
Module with global fixtures
"""
import os

import boto3
import botocore.exceptions

import pytest
from moto import mock_s3

from sentinelhub import SHConfig


@pytest.fixture(name='config')
def config_fixture():
    config = SHConfig()
    for param in config.get_params():
        env_variable = param.upper()
        if os.environ.get(env_variable):
            setattr(config, param, os.environ.get(env_variable))
    return config


@pytest.fixture(name='gpkg_file')
def local_gpkg_example_file_fixture():
    """ A pytest fixture to retrieve a gpkg example file
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/import-gpkg-test.gpkg')
    return path


@pytest.fixture(name='s3_gpkg_file')
def s3_gpkg_example_file_fixture(config):
    """ A pytest fixture to retrieve a gpkg example file
    """
    aws_config = {
        "region_name": 'eu-central-1',
    }
    if config.aws_access_key_id and config.aws_secret_access_key:
        aws_config['aws_access_key_id'] = config.aws_access_key_id
        aws_config['aws_secret_access_key'] = config.aws_secret_access_key

    client = boto3.client('s3', **aws_config)

    try:
        client.head_bucket(Bucket='eolearn-io')
        return 's3://eolearn-io/import-gpkg-test.gpkg'
    except botocore.exceptions.ClientError:
        return pytest.skip(msg='No access to the bucket.')


@pytest.fixture(name='geodb_client')
def geodb_client_fixture():
    """ A geoDB client object
    """
    geodb = pytest.importorskip('xcube_geodb.core.geodb')

    client_id = os.getenv('GEODB_AUTH_CLIENT_ID')
    client_secret = os.getenv('GEODB_AUTH_CLIENT_SECRET')

    if not (client_id or client_secret):
        raise ValueError("Could not initiate geoDB client, GEODB_AUTH_CLIENT_ID and GEODB_AUTH_CLIENT_SECRET missing!")

    return geodb.GeoDBClient(
        client_id=client_id,
        client_secret=client_secret
    )


@mock_s3
def _create_s3_bucket(bucket_name):
    s3resource = boto3.resource('s3', region_name='eu-central-1')
    bucket = s3resource.Bucket(bucket_name)

    if bucket.creation_date:  # If bucket already exists
        for key in bucket.objects.all():
            key.delete()
        bucket.delete()

    s3resource.create_bucket(Bucket=bucket_name,
                             CreateBucketConfiguration={'LocationConstraint': 'eu-central-1'})
