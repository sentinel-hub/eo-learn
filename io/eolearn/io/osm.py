"""
Module containing tasks used for reading vector data from OpenStreetMap
"""

import geopandas
import overpass

from sentinelhub.constants import CRS
from shapely.geometry import shape, box, mapping, Polygon

from eolearn.core import EOTask, FeatureType


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
    :param overpass_opts: Options to pass to the Overpass API constructor,
    see: https://github.com/mvexel/overpass-api-python-wrapper#api-constructor
    :type overpass_opts: dict
    """

    def __init__(self, feature_name, query, polygonize=False, overpass_opts=None):
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
        response = self.api.get('{}{}'.format(self.query, osm_bbox), verbosity='geom')

        # clip geometries to bounding box
        for feat in response['features']:
            geom = shape(feat['geometry'])
            if self.polygonize:
                # currently the API doesn't support relations so we use shapely to wrap geometries
                # https://github.com/mvexel/overpass-api-python-wrapper/issues/106#issuecomment-421446118
                # https://github.com/mvexel/overpass-api-python-wrapper/issues/48#issuecomment-418770376
                geom = Polygon(geom)
            clipped_geom = geom.intersection(clip_shape)
            feat['geometry'] = mapping(clipped_geom)

        # import to geopandas, transform and return
        gdf = geopandas.GeoDataFrame.from_features(response['features'])
        gdf.crs = {'init' :'epsg:4326'} # all Osmium data is returned with this CRS
        gdf = gdf.to_crs({'init': eopatch.bbox.crs.ogc_string().lower()})
        eopatch[FeatureType.VECTOR_TIMELESS][self.feature_name] = gdf
        return eopatch
