import os
import unittest
import logging

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from eolearn.core import EOPatch
from eolearn.io import OSMInput

logging.basicConfig(level=logging.DEBUG)

class GeoPandasTestCase(unittest.TestCase):
    def assertGDFEqual(self, left, right, **kwargs):
        assert_geodataframe_equal(left, right, **kwargs)


class TestOSMInput(GeoPandasTestCase):
    """ Test if OSMInput task returns the expected data
    """
    @classmethod
    def setUpClass(cls):
        cls.eopatch = EOPatch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '../../../example_data/TestEOPatch'))

    def test_osm(self):
        """Tests Osmium response + handling against OSM fixture"""
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data/osm.geojson')
        osm_fixture = gpd.read_file(path)

        feature_name = 'road'

        patch = EOPatch.load('example_data/TestEOPatch/')
        osm_task = OSMInput(feature_name, 'way["highway"="unclassified"]', polygonize=False)
        osm_task.execute(patch)

        road_df = patch.vector_timeless['road']
        print(osm_fixture.crs, road_df.crs)
        self.assertGDFEqual(osm_fixture, road_df, check_like=True)


if __name__ == '__main__':
    unittest.main()
