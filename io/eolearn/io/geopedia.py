"""
Module for adding data obtained from sentinelhub package to existing EOPatches

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio.transform
import rasterio.warp

from sentinelhub import CRS, BBox, GeopediaWmsRequest, MimeType

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import FeatureSpec

LOGGER = logging.getLogger(__name__)


class AddGeopediaFeatureTask(EOTask):
    """
    Task for adding a feature from Geopedia to an existing EOPatch.

    At the moment the Geopedia supports only WMS requestes in EPSG:3857, therefore to add feature to EOPatch
    in arbitrary CRS and arbitrary service type the following steps are performed:
    * transform BBOX from EOPatch's CRS to EPSG:3857
    * get raster from Geopedia Request in EPSG:3857
    * vectorize the returned raster using rasterio
    * project vectorized raster back to EOPatch's CRS
    * rasterize back and add raster to EOPatch
    """

    def __init__(
        self,
        feature: FeatureSpec,
        layer: Union[str, int],
        theme: str,
        raster_value: Union[Dict[str, Tuple[float, List[float]]], float],
        raster_dtype: Union[np.dtype, type] = np.uint8,
        no_data_val: float = 0,
        image_format: MimeType = MimeType.PNG,
        mean_abs_difference: float = 2,
    ):
        self.feature_type, self.feature_name = self.parse_feature(feature)

        self.raster_value = raster_value
        self.raster_dtype = raster_dtype
        self.no_data_val = no_data_val
        self.mean_abs_difference = mean_abs_difference

        self.layer = layer
        self.theme = theme

        self.image_format = image_format

    def _get_wms_request(self, bbox: Optional[BBox], size_x: int, size_y: int) -> GeopediaWmsRequest:
        """
        Returns WMS request.
        """
        if bbox is None:
            raise ValueError("Bbox has to be defined!")

        bbox_3857 = bbox.transform(CRS.POP_WEB)

        return GeopediaWmsRequest(
            layer=self.layer,
            theme=self.theme,
            bbox=bbox_3857,
            width=size_x,
            height=size_y,
            image_format=self.image_format,
        )

    def _reproject(self, eopatch: EOPatch, src_raster: np.ndarray) -> np.ndarray:
        """
        Re-projects the raster data from Geopedia's CRS (POP_WEB) to EOPatch's CRS.
        """
        if not eopatch.bbox:
            raise ValueError("To reproject raster data, eopatch.bbox has to be defined!")

        height, width = src_raster.shape

        dst_raster = np.ones((height, width), dtype=self.raster_dtype)

        src_bbox = eopatch.bbox.transform(CRS.POP_WEB)
        src_transform = rasterio.transform.from_bounds(*src_bbox, width=width, height=height)

        dst_bbox = eopatch.bbox
        dst_transform = rasterio.transform.from_bounds(*dst_bbox, width=width, height=height)

        rasterio.warp.reproject(
            src_raster,
            dst_raster,
            src_transform=src_transform,
            src_crs=CRS.POP_WEB.ogc_string(),
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=eopatch.bbox.crs.ogc_string(),
            dst_nodata=self.no_data_val,
        )

        return dst_raster

    def _to_binary_mask(self, array: np.ndarray, binaries_raster_value: float) -> np.ndarray:
        """
        Returns binary mask (0 and raster_value)
        """
        # check where the transparency is not zero
        return (array[..., -1] > 0).astype(self.raster_dtype) * binaries_raster_value

    def _map_from_binaries(
        self,
        eopatch: EOPatch,
        dst_shape: Union[int, Tuple[int, int]],
        request_data: np.ndarray,
        binaries_raster_value: float,
    ) -> np.ndarray:
        """
        Each request represents a binary class which will be mapped to the scalar `raster_value`
        """
        if self.feature_name in eopatch[self.feature_type]:
            raster = eopatch[self.feature_type][self.feature_name].squeeze()
        else:
            raster = np.ones(dst_shape, dtype=self.raster_dtype) * self.no_data_val

        new_raster = self._reproject(eopatch, self._to_binary_mask(request_data, binaries_raster_value))

        # update raster
        raster[new_raster != 0] = new_raster[new_raster != 0]

        return raster

    def _map_from_multiclass(
        self,
        eopatch: EOPatch,
        dst_shape: Union[int, Tuple[int, int]],
        request_data: np.ndarray,
        multiclass_raster_value: Dict[str, Tuple[float, List[float]]],
    ) -> np.ndarray:
        """
        `raster_value` is a dictionary specifying the intensity values for each class and the corresponding label value.

        A dictionary example for GLC30 LULC mapping is:
        raster_value = {'no_data': (0,[0,0,0,0]),
                        'cultivated land': (1,[193, 243, 249, 255]),
                        'forest': (2,[73, 119, 20, 255]),
                        'grassland': (3,[95, 208, 169, 255]),
                        'shrubland': (4,[112, 179, 62, 255]),
                        'water': (5,[154, 86, 1, 255]),
                        'wetland': (6,[244, 206, 126, 255]),
                        'tundra': (7,[50, 100, 100, 255]),
                        'artificial surface': (8,[20, 47, 147, 255]),
                        'bareland': (9,[202, 202, 202, 255]),
                        'snow and ice': (10,[251, 237, 211, 255])}
        """
        raster = np.ones(dst_shape, dtype=self.raster_dtype) * self.no_data_val

        for value, intensities in multiclass_raster_value.values():
            raster[np.mean(np.abs(request_data - intensities), axis=-1) < self.mean_abs_difference] = value

        return self._reproject(eopatch, raster)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        Add requested feature to this existing EOPatch.
        """
        data_arr = eopatch[FeatureType.MASK]["IS_DATA"]
        _, height, width, _ = data_arr.shape

        request = self._get_wms_request(eopatch.bbox, width, height)

        (request_data,) = np.asarray(request.get_data())

        if isinstance(self.raster_value, dict):
            raster = self._map_from_multiclass(eopatch, (height, width), request_data, self.raster_value)
        elif isinstance(self.raster_value, (int, float)):
            raster = self._map_from_binaries(eopatch, (height, width), request_data, self.raster_value)
        else:
            raise ValueError("Unsupported raster value type")

        if self.feature_type is FeatureType.MASK_TIMELESS and raster.ndim == 2:
            raster = raster[..., np.newaxis]

        eopatch[self.feature_type][self.feature_name] = raster

        return eopatch
