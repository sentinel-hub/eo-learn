"""
Module containing tasks used for reading from non-SentinelHub tile services
"""

import math
from io import BytesIO

import numpy as np
import requests

import mercantile
import rasterio.warp
import rasterio.transform

from eolearn.core import EOTask, FeatureType
from sentinelhub import transform_bbox
from sentinelhub.constants import CRS
from sentinelhub.io_utils import read_image


TILE_SIZE = 256
CIRC = 40075016.686  # circumference of the earth in meters


class TileMapServiceInput(EOTask):
    """
    Use a tile map service (TMS) as input to a DATA_TIMELESS feature.

    At the moment the TMS input supports only services in EPSG:3857.

    :param feature_name: EOPatch feature into which data will be imported
    :type feature: (FeatureType, str)
    :param url_template: TMS URL templace (e.g http://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token=afake_access_token_here)
    :type url_template: str
    :param no_data_value: Values where given Geo-Tiff image does not cover EOPatch. Default is 0.
    :type no_data_value: int or float
    """

    def __init__(self, url_template, no_data_value=0):
        self.url_template = url_template
        self.no_data_value = no_data_value

    @staticmethod
    def _get_zoom(res, lat):
        """ For a given resolution and latitude, return the lowest integer zoom level
        which will give 256x256 tiles with better resolution than the input"""
        # calculation from https://wiki.openstreetmap.org/wiki/Zoom_levels#Distance_per_pixel_math
        return math.ceil(
            (math.log(math.cos(lat / 180 * math.pi) * CIRC / res) - math.log(TILE_SIZE)) / math.log(2)
        )

    @staticmethod
    def _parse_numeric(string):
        """return an integer from a string containing one"""
        return int(''.join(x for x in string if x.isdigit()))

    def _reproject(self, eopatch, src_raster, src_bounds):
        """
        Reprojects the raster data from Geopedia's CRS (POP_WEB) to EOPatch's CRS.
        """
        data_arr = eopatch[FeatureType.MASK]['IS_DATA']
        _, dst_height, dst_width, _ = data_arr.shape

        nbands, height, width = src_raster.shape

        dst_transform, _, _ = rasterio.warp.calculate_default_transform(
            {'init': CRS.ogc_string(CRS.POP_WEB)},
            {'init': CRS.ogc_string(eopatch.bbox.crs)}, 
            width,
            height,
            *src_bounds
        )

        dst_raster = np.ones((nbands, dst_height, dst_width), dtype=src_raster.dtype)

        src_transform = rasterio.transform.from_bounds(
            *src_bounds, width=width, height=height
        )

        dst_bbox = eopatch.bbox
        dst_transform = rasterio.transform.from_bounds(
            *dst_bbox, width=width, height=height
        )

        rasterio.warp.reproject(
            src_raster,
            dst_raster,
            src_transform=src_transform,
            src_crs={'init': CRS.ogc_string(CRS.POP_WEB)},
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs={'init': CRS.ogc_string(eopatch.bbox.crs)},
            dst_nodata=self.no_data_value
        )

        return dst_raster

    def _fetch_tiles(self, bounds, zoom):
        """Find all necessary tiles and download them into a single 3-band image."""
        covering_tiles = list(mercantile.tiles(*bounds, zoom))
        image_tile_xs = {t.x for t in covering_tiles}
        image_tile_ys = {t.y for t in covering_tiles}

        extrema = {
            'x': {'min': min(image_tile_xs), 'max': max(image_tile_xs) + 1},
            'y': {'min': min(image_tile_ys), 'max': max(image_tile_ys) + 1}
        }

        w, n = mercantile.xy(
            *mercantile.ul(extrema["x"]["min"], extrema["y"]["min"], zoom)
        )
        e, s = mercantile.xy(
            *mercantile.ul(extrema["x"]["max"], extrema["y"]["max"], zoom)
        )
        bounds = [w, s, e, n]

        image_tile = np.empty(shape=(
            len(image_tile_ys) * TILE_SIZE,
            len(image_tile_xs) * TILE_SIZE,
            3
        )).astype(np.uint8)

        for tile in covering_tiles:
            url = self.url_template.format(z=tile.z, x=tile.x, y=tile.y)
            resp = requests.get(url)

            # tile origin
            col_off = (tile.x - min(image_tile_xs)) * TILE_SIZE
            row_off = (tile.y - min(image_tile_ys)) * TILE_SIZE

            print(read_image(BytesIO(resp.content)))

            image_tile[
                row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE, :
            ] = read_image(BytesIO(resp.content))

        return image_tile, bounds

    def execute(self, eopatch, feature_name, zoom=None):
        """ Execute function which adds new DATA_TIMELESS layer to the EOPatch
        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        :param zoom: optional parameter specifying the zoom level at which to download tiles
        :type zoom: int
        :return: New EOPatch with added DATA_TIMELESS layer
        :rtype: EOPatch
        """
        latlng_bounds = eopatch.bbox.transform(CRS.WGS84)
        maximum_latitude = max(abs(latlng_bounds.min_y), abs(latlng_bounds.max_y))

        # get ideal zoom for the smallest resolution and furthest latitude from the equator
        if not zoom:
            minimum_resolution = min(
                self._parse_numeric(eopatch.meta_info['size_x']),
                self._parse_numeric(eopatch.meta_info['size_y'])
            )
            zoom = self._get_zoom(minimum_resolution, maximum_latitude)

        image_tile, image_bounds = self._fetch_tiles(latlng_bounds, zoom)

        eopatch[FeatureType.DATA_TIMELESS][feature_name] = self._reproject(
            eopatch, image_tile, image_bounds
        )
        return eopatch
