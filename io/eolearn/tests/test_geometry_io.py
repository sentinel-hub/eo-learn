"""
Credits:
Copyright (c) 2021-> Matej Aleksandrov, Matej Batič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest
import logging
import os

from eolearn.core import FeatureType
from eolearn.io import GeoDBVectorImportTask, GeopediaVectorImportTask, VectorImportTask

from sentinelhub import BBox, CRS
from xcube_geodb.core.geodb import GeoDBClient

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize(
    argnames='reproject, clip, n_features, bbox',
    ids=['simple', 'bbox', 'bbox_full', 'bbox_smaller'],
    argvalues=[
        (False, False, 193, None),
        (False, False, 193, BBox([857000, 6521500, 861000, 6525500], crs='epsg:2154')),
        (True, True, 193,
         BBox([657089.4234095092, 5071037.944679743, 661093.7825844937, 5075039.833417523], crs=CRS.UTM_31N)),
        (True, True, 125,
         BBox([657690.073230332, 5071637.340226217, 660493.1246978994, 5074440.391780451], crs=CRS.UTM_31N))
    ])
class TestVectorImportTask:

    @pytest.mark.parametrize('path', [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/import-gpkg-test.gpkg'),
        's3://eolearn-io/import-gpkg-test.gpkg'
    ], ids=['local', 's3'])
    def test_import_file(self, path, reproject, clip, n_features, bbox):
        feature = FeatureType.VECTOR_TIMELESS, 'lpis_iacs'
        import_task = VectorImportTask(
            feature=feature,
            path=path,
            reproject=reproject,
            clip=clip
        )
        eop = import_task.execute(bbox=bbox)
        assert len(eop[feature]) == n_features, 'Wrong number of features!'
        assert eop[feature].crs == eop.bbox.crs.pyproj_crs()

    def test_import_from_geopedia(self, reproject, clip, n_features, bbox):
        feature = FeatureType.VECTOR_TIMELESS, 'lpis_iacs'
        import_task = GeopediaVectorImportTask(
            feature=feature,
            geopedia_table=3447,
            reproject=reproject,
            clip=clip
        )
        eop = import_task.execute(bbox=bbox)
        assert len(eop[feature]) == n_features, 'Wrong number of features!'
        assert eop[feature].crs.to_epsg() == eop.bbox.crs.epsg


@pytest.fixture(name='geodb_client')
def geodb_client_fixture():
    """ A geoDB client object
        """
    client_id = os.getenv('GEODB_AUTH_CLIENT_ID')
    client_secret = os.getenv('GEODB_AUTH_CLIENT_SECRET')

    if not (client_id or client_secret):
        raise ValueError("Could not initiate geoDB client, GEODB_AUTH_CLIENT_ID and GEODB_AUTH_CLIENT_SECRET missing!")

    return GeoDBClient(
        client_id=client_id,
        client_secret=client_secret
    )


@pytest.mark.parametrize(
    argnames='reproject, clip, n_features, bbox',
    ids=['bbox', 'bbox_smaller'],
    argvalues=[
        (False, False, 193, BBox([857000, 6521500, 861000, 6525500], crs='epsg:2154')),
        (True, True, 116, BBox([857400.0, 6521900.0, 860600.0, 6525100.0], crs='epsg:2154'))
    ])
def test_import_from_geodb(geodb_client, reproject, clip, n_features, bbox):
    feature = FeatureType.VECTOR_TIMELESS, 'lpis_iacs'
    import_task = GeoDBVectorImportTask(
        feature=feature,
        geodb_client=geodb_client,
        geodb_collection='france_metropolitan_gsaa_2015',
        geodb_db='lpis_iacs',
        reproject=reproject,
        clip=clip
    )
    eop = import_task.execute(bbox=bbox)
    assert len(eop[feature]) == n_features, 'Wrong number of features!'
    assert eop[feature].crs.to_epsg() == eop.bbox.crs.epsg
