"""
Module containing tasks used for reading from non-SentinelHub tile services
"""

import math
from io import BytesIO

from mercantile import tiles, ul
import numpy as np
import requests

from eolearn.core import EOTask, FeatureType
from sentinelhub.constants import CRS
from sentinelhub.io_utils import read_image

TILE_SIZE = 256
CIRC = 40075016.686 # circumference of the earth in meters

def get_zoom(res, lat):
    """ For a given resolution and latitude, return the lowest integer zoom level
    which will give 256x256 tiles with better resolution than the input"""
    # calculation from https://wiki.openstreetmap.org/wiki/Zoom_levels#Distance_per_pixel_math
    return math.ceil((math.log(math.cos(lat / 180 * math.pi) * CIRC / res) - math.log(TILE_SIZE)) / math.log(2))

def get_resolution(zoom, lat):
    """ For a given zoom level and latitude, return pixel size in degrees"""
    # calculation from https://wiki.openstreetmap.org/wiki/Zoom_levels#Distance_per_pixel_math
    return 360 * math.cos(lat / 180 * math.pi) / (math.pow(2, zoom)) / TILE_SIZE

def parse_numeric(string):
    """return an integer from a string containing one"""
    return int(''.join(x for x in string if x.isdigit()))

def get_window(zoom, maximum_latitude, image_tile_xs, image_tile_ys, latlng_bounds):
    """return the bbox as a window in pixel bounds of a larger image tile"""
    image_tile_origin = ul(min(image_tile_xs), min(image_tile_ys), zoom)
    x_res = get_resolution(zoom, maximum_latitude)
    y_res = 180 / math.pow(2, zoom) / TILE_SIZE
    top = int((image_tile_origin.lat - latlng_bounds.max_y) / y_res)
    left = int((latlng_bounds.min_x - image_tile_origin.lng) / x_res)
    bottom = int((image_tile_origin.lat - latlng_bounds.min_y) / y_res)
    right = int((latlng_bounds.max_x - image_tile_origin.lng) / x_res)
    return (top, bottom, left, right)

class MapboxXYZInput(EOTask):
    """ Use a Mapbox raster tile service as input to a DATA_TIMELESS feature

    :param id: Feature which will be exported
    :type id: str
    :param mask_name: root directory where all Geo-Tiff images will be saved
    :type mask_name: str
    :param access_token: Mapbox Access Token https://docs.mapbox.com/help/glossary/access-token/
    :type access_token: str
    """

    def __init__(self, mapbox_id, mask_name, access_token):
        self.mapbox_id = mapbox_id
        self.mask_name = mask_name
        self.access_token = access_token

    def execute(self, eopatch, zoom=None):
        """ Execute function which adds new DATA_TIMELESS layer to the EOPatch
        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param zoom: optional parameter specifying the zoom level at which to download tiles
        :type zoom: int
        :return: New EOPatch with added raster layer
        :rtype: EOPatch
        """
        latlng_bounds = eopatch.bbox.transform(CRS.WGS84)
        maximum_latitude = max(abs(latlng_bounds.min_y), abs(latlng_bounds.max_y))

        # get ideal zoom for the smallest resolution and furthest latitude from the equator
        if not zoom:
            minimum_resolution = min(
                parse_numeric(eopatch.meta_info['size_x']),
                parse_numeric(eopatch.meta_info['size_y'])
            )
            zoom = get_zoom(minimum_resolution, maximum_latitude)

        # find all necessary tiles and download them into a single 3-band image
        covering_tiles = list(tiles(*latlng_bounds, zoom))
        image_tile_xs = {t.x for t in covering_tiles}
        image_tile_ys = {t.y for t in covering_tiles}
        image_tile = np.empty(shape=(
            len(image_tile_ys) * TILE_SIZE,
            len(image_tile_xs) * TILE_SIZE,
            3
        )).astype(np.uint8)
        for tile in covering_tiles:
            url = 'http://api.mapbox.com/v4/{}/{}/{}/{}.png?access_token={}'.format(
                self.mapbox_id, tile.z, tile.x, tile.y, self.access_token
            )
            resp = requests.get(url)
            torg = ( # tile origin
                (tile.x - min(image_tile_xs)) * TILE_SIZE,
                (tile.y - min(image_tile_ys)) * TILE_SIZE
            )
            image_tile[torg[1]:torg[1] + TILE_SIZE, torg[0]:torg[0] + TILE_SIZE] = read_image(BytesIO(resp.content))

        # read into the larger image_tile
        top, bottom, left, right = get_window(zoom, maximum_latitude, image_tile_xs, image_tile_ys, latlng_bounds)


        eopatch[FeatureType.DATA_TIMELESS][self.mask_name] = image_tile[top:bottom, left:right]
        return eopatch
