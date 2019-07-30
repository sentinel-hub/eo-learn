"""
Module containing tasks used for reading vector data from OpenStreetMap
"""

import overpass
import geopandas
from shapely.geometry import shape, box, mapping

from eolearn.core import EOTask, FeatureType
from sentinelhub.constants import CRS


class OSMInput(EOTask):
    """ Use OpenStreetMap (OSM) data from an Overpass API as input to a VECTOR_TIMELESS feature.
    In case of timeouts or too many requests against the main Overpass endpoint, find additional
    endpoints at see other options https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances

    :param feature_name: EOPatch feature into which data will be imported
    :type feature_name: (FeatureType, str)
    :param query: Overpass API Querystring: https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
    :type query: str
    :param polygonize: Whether or not to treat ways as polygons, defaults to True
    :type polygonize: bool
    :param overpass_opts: Options to pass to the Overpass API constructor, see: https://github.com/mvexel/overpass-api-python-wrapper#api-constructor
    :type overpass_opts: dict
    """

    def __init__(self, feature_name, query, polygonize=True, overpass_opts={}):
        self.feature_name = feature_name
        self.query = query
        self.polygonize = polygonize
        self.api = overpass.API(overpass_opts)

    def execute(self, eopatch):
        """ Execute function which adds new VECTOR_TIMELESS layer to the EOPatch
        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :return: New EOPatch with added VECTOR_TIMELESS layer
        :rtype: EOPatch
        """

        if self.feature_name is None:
            raise ValueError('\'feature_name\' is a required parameter.')
        if self.query is None:
            raise ValueError('Please provide a \'query\', https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL')
        if not eopatch.bbox:
            raise ValueError('Each EOPatch requires a bbox to fetch data')

        # handling for various bounds variables
        ll_bounds = eopatch.bbox.transform(CRS.WGS84)
        clip_shape = box(*ll_bounds)
        osm_bbox = tuple([*ll_bounds.reverse()])

        # make the overpass request
        response = self.api.get(f'{self.query}{osm_bbox}', verbosity='geom')

        # clip geometries to bounding box
        for feat in response['features']:
            geom = shape(feat['geometry'])
            if self.polygonize:
                geom = geom.convex_hull
            clipped_geom = geom.intersection(clip_shape)
            feat['geometry'] = mapping(clipped_geom)


        # import to geopandas, transform and return
        gdf = geopandas.GeoDataFrame.from_features(response['features'])
        gdf.crs = {'init' :'epsg:4326'}
        gdf = gdf.to_crs({'init': eopatch.bbox.crs.ogc_string()})
        eopatch[FeatureType.VECTOR_TIMELESS][self.feature_name] = gdf
        return eopatch
