"""
Module for adding data obtained from sentinelhub package to existing EOPatches
"""

import numpy as np
import logging
from rasterio import transform, warp

from sentinelhub import WmsRequest, WcsRequest, MimeType, DataSource, CustomUrlParam, CRS, GeopediaWmsRequest,\
    ServiceType, transform_bbox

from eolearn.core import EOTask, FeatureType, get_common_timestamps

LOGGER = logging.getLogger(__name__)


class AddSentinelHubOGCFeature(EOTask):
    """
    Task for adding feature to existing EOPatch using Sentinel Hub's OGC request. The following OGC request
    parameters are taken from the EOPatch's meta data (set by the request that created the EOPatch):
    * size_x: width or resx
    * size_y: height or resy
    * maxcc: max cloud coverage
    * time_difference
    * service_type: WMS or WCS
    * time_interval

    In case the available dates in feature request don't match the dates of frames in the EOPatch the
    timestamp consolidation is triggered. This means that if for example and EOPatch has data for the following dates
    (noted as integers for brevity):
        * eopatch.timestamps = [1, 2, 3, 4, 5],
    while available dates for the new feature are:
        * new_feature_request.get_dates() = [1, 3, 4, 5, 6]
        * note date 2 is missing and 6 is additional in the request,
    then this task will
        * download new feature data only for the dates [1,3,4,5]
        * and will remove all data for date [2] in the eopatch.
    Finally, the eopatch will at the end contain data for dates [1, 3, 4, 5] for old and new features.

    :param layer: the preconfigured layer to be added to EOPatch. Required.
    :type layer: str
    :param feature_name: user specified name (key) for this feature. Optional.
    :type feature_name: str or None. Default is the same as layer.
    :param data_source: Source of requested satellite data.
    :type data_source: constants.DataSource
    :param image_format: format of the returned image by the Sentinel Hub's WMS getMap service. Default is 32-bit TIFF.
    :type image_format: constants.MimeType
    :param instance_id: user's instance id. If ``None`` the instance id is taken from the ``config.json``
                        configuration file from sentinelhub-py package.
    :type instance_id: str or None
    :param custom_url_params: dictionary of CustomUrlParameters and their values supported by Sentinel Hub's WMS and WCS
                              services. All available parameters are described at
                              http://www.sentinel-hub.com/develop/documentation/api/custom-url-parameters. Note: in
                              case of constants.CustomUrlParam.EVALSCRIPT the dictionary value must be a string
                              of Javascript code that is not encoded into base64.
    :type custom_url_params: dictionary of CustomUrlParameter enum and its value, i.e.
                              ``{constants.CustomUrlParam.ATMFILTER:'ATMCOR'}``
    """

    def __init__(self, feature_type, layer, feature_name=None, data_source=None,
                 image_format=MimeType.TIFF_d32f, instance_id=None, custom_url_params=None):

        self.feature_type = feature_type
        self.layer = layer
        self.feature_name = layer if feature_name is None else feature_name
        self.data_source = data_source
        self.image_format = image_format
        self.instance_id = instance_id

        custom_params = {CustomUrlParam.SHOWLOGO: False,
                         CustomUrlParam.TRANSPARENT: False}
        if custom_url_params is None:
            self.custom_url_params = custom_params
        else:
            self.custom_url_params = {**custom_params, **custom_url_params}

    def _get_wms_request(self, bbox, time_interval, size_x, size_y, maxcc, time_difference):
        """
        Returns WMS request.
        """
        return WmsRequest(layer=self.layer,
                          bbox=bbox,
                          time=time_interval,
                          width=size_x,
                          height=size_y,
                          maxcc=maxcc,
                          image_format=self.image_format,
                          custom_url_params=self.custom_url_params,
                          time_difference=time_difference,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _get_wcs_request(self, bbox, time_interval, size_x, size_y, maxcc, time_difference):
        """
        Returns WCS request.
        """
        return WcsRequest(layer=self.layer,
                          bbox=bbox,
                          time=time_interval,
                          resx=size_x, resy=size_y,
                          maxcc=maxcc,
                          image_format=self.image_format,
                          custom_url_params=self.custom_url_params,
                          time_difference=time_difference,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _reshape_array(self, array, dims_dict):
        """ Reshape array if dimensions do not match requirements

        :param array: Input array
        :param dims_dict: Dictionary with target dimensionality for the feature types
        :return: Reshaped array with additional channel
        """
        if array.ndim == dims_dict[self.feature_type.value] - 1:
            return array.reshape(array.shape + (1,))
        return array

    def _check_dimensionality(self, array, dims_dict):
        """ Method to ensure array has the dimensionality required by the feature type

        :param array: Input array
        :param dims_dict: Dictionary with target dimensionality for the feature types
        :return: Reshaped array with additional channel
        """
        if self.feature_type in [FeatureType.DATA, FeatureType.MASK]:
            return self._reshape_array(array, dims_dict)
        if self.feature_type in [FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS]:
            array = array.squeeze(axis=0)
            return self._reshape_array(array, dims_dict)
        return array

    def execute(self, eopatch):
        """
        Add requested feature to this existing EOPatch.
        """
        size_x = eopatch.meta_info['size_x']
        size_y = eopatch.meta_info['size_y']
        maxcc = eopatch.meta_info['maxcc']
        time_difference = eopatch.meta_info['time_difference']
        service_type = eopatch.meta_info['service_type']
        time_interval = (eopatch.timestamp[0].isoformat(), eopatch.timestamp[-1].isoformat())

        request = {ServiceType.WMS: self._get_wms_request,
                   ServiceType.WCS: self._get_wcs_request}[service_type](eopatch.bbox, time_interval, size_x, size_y,
                                                                         maxcc, time_difference)

        # check timestamp consistency between request and this eopatch
        request_dates = request.get_dates()
        download_frames = get_common_timestamps(request_dates, eopatch.timestamp)

        request_return = request.get_data(raise_download_errors=False, data_filter=download_frames)
        bad_data = [idx for idx, value in enumerate(request_return) if value is None]
        for idx in reversed(sorted(bad_data)):
            LOGGER.warning('Data from %s could not be downloaded for %s!', str(request_dates[idx]), self.layer)
            del request_return[idx]
            del request_dates[idx]

        request_data = np.asarray(request_return)

        removed_frames = eopatch.consolidate_timestamps(request_dates)
        for rm_frame in removed_frames:
            LOGGER.warning('Removed data for frame %s from eopatch '
                           'due to unavailability of %s!', str(rm_frame), self.layer)

        request_data = self._check_dimensionality(request_data, eopatch.ndims)

        eopatch.add_feature(self.feature_type, self.feature_name, request_data)

        return eopatch


class AddSen2CorClassificationFeature(AddSentinelHubOGCFeature):
    """
    Adds SCL (scene classification), CLD (cloud probability) or SNW (snow probability)
    Sen2Cor classification results to EOPatch's MASK or DATA feature. The feature is added to MASK (SCL) or
    DATA (CLD, SNW) feature types of EOPatch. The feature names are set to be SCL, CLD or SNW.

    Sen2Cor's scene classification (SCL) contains 11 classes with the following values and meanings:
       * 1 - SC_SATURATED_DEFECTIVE
       * 2 - SC_DARK_FEATURE_SHADOW
       * 3 - SC_CLOUD_SHADOW
       * 4 - VEGETATION
       * 5 - NOT-VEGETATED
       * 6 - WATER
       * 7 - SC_CLOUD_LOW_PROBA / UNCLASSIFIED
       * 8 - SC_CLOUD_MEDIUM_PROBA
       * 9 - CLOUD_HIGH_PROBABILITY
       * 10 - THIN_CIRRUS
       * 11 - SNOW
    """
    def __init__(self, sen2cor_classification, layer, **kwargs):
        # definition of possible types and target features
        classification_types = {'SCL': FeatureType.MASK,
                                'CLD': FeatureType.DATA,
                                'SNW': FeatureType.DATA}

        if sen2cor_classification not in classification_types.keys():
            raise ValueError('Unsupported Sen2Cor classification type: {}.'
                             ' Possible types are: {}'.format(sen2cor_classification, classification_types))

        evalscript = 'return ['+sen2cor_classification+'];'

        super(AddSen2CorClassificationFeature, self).__init__(feature_type=classification_types[sen2cor_classification],
                                                              feature_name=sen2cor_classification,
                                                              layer=layer,
                                                              data_source=DataSource.SENTINEL2_L2A,
                                                              custom_url_params={CustomUrlParam.EVALSCRIPT: evalscript},
                                                              **kwargs)


class AddDEMFeature(AddSentinelHubOGCFeature):
    """
    Adds DEM to DATA_TIMELESS EOPatch feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddDEMFeature, self).__init__(feature_type=FeatureType.DATA_TIMELESS, layer=layer,
                                            data_source=DataSource.DEM, **kwargs)


class AddS2L1CFeature(AddSentinelHubOGCFeature):
    """
    Adds Sentinel-2 L1C feature to EOPatch's DATA feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddS2L1CFeature, self).__init__(feature_type=FeatureType.DATA, layer=layer,
                                              data_source=DataSource.SENTINEL2_L1C, **kwargs)


class AddS2L2AFeature(AddSentinelHubOGCFeature):
    """
    Adds Sentinel-2 L2A feature to EOPatch's DATA feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddS2L2AFeature, self).__init__(feature_type=FeatureType.DATA, layer=layer,
                                              data_source=DataSource.SENTINEL2_L2A, **kwargs)


class AddL8Feature(AddSentinelHubOGCFeature):
    """
    Adds Landsat 8 feature to EOPatch's DATA feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddL8Feature, self).__init__(feature_type=FeatureType.DATA, layer=layer,
                                           data_source=DataSource.LANDSAT8, **kwargs)


class AddGeopediaFeature(EOTask):
    """
    Task for adding a feature from Geopedia to an existing EOPatch.

    At the moment the Geopedia supports only WMS requestes in EPSG:3857, therefore to add feature to EOPatch
    in arbitrary CRS and arbitrary service type the following steps are performed:
    * transform BBOX from EOPatch's CRS to EPSG:3857
    * get raster from Geopedia Request in EPSG:3857
    * vectorize the returned raster using rasterio
    * project vectorised raster back to EOPatch's CRS
    * rasterize back and add raster to EOPatch
    """

    def __init__(self, feature_type, feature_name, layer, theme,
                 raster_value, raster_dtype=np.uint8, no_data_val=0,
                 image_format=MimeType.PNG, mean_abs_difference=2):

        self.feature_type = feature_type
        self.feature_name = feature_name

        self.raster_value = raster_value
        self.raster_dtype = raster_dtype
        self.no_data_val = no_data_val
        self.mean_abs_difference = mean_abs_difference

        self.layer = layer
        self.theme = theme

        self.image_format = image_format

    def _get_wms_request(self, bbox, size_x, size_y):
        """
        Returns WMS request.
        """
        bbox_3857 = transform_bbox(bbox, CRS.POP_WEB)

        return GeopediaWmsRequest(layer=self.layer,
                                  theme=self.theme,
                                  bbox=bbox_3857,
                                  width=size_x,
                                  height=size_y,
                                  image_format=self.image_format,
                                  custom_url_params={CustomUrlParam.TRANSPARENT: True})

    def _get_wcs_request(self, bbox, size_x, size_y):
        """
        Returns WMS request.
        """
        raise NotImplementedError

    def _reproject(self, eopatch, src_raster):
        """
        Reprojects the raster data from Geopedia's CRS (POP_WEB) to EOPatch's CRS.
        """
        height, width = src_raster.shape

        dst_raster = np.ones((height, width), dtype=self.raster_dtype)

        src_bbox = transform_bbox(eopatch.bbox, CRS.POP_WEB)
        src_transform = transform.from_bounds(*src_bbox, width=width, height=height)

        dst_bbox = eopatch.bbox
        dst_transform = transform.from_bounds(*dst_bbox, width=width, height=height)

        warp.reproject(src_raster, dst_raster,
                       src_transform=src_transform, src_crs={'init': CRS.ogc_string(CRS.POP_WEB)},
                       src_nodata=0,
                       dst_transform=dst_transform, dst_crs={'init': CRS.ogc_string(eopatch.bbox.crs)},
                       dst_nodata=self.no_data_val)

        return dst_raster

    def _to_binary_mask(self, array):
        """
        Returns binary mask (0 and raster_value)
        """
        # check where the transparency is not zero
        return (array[..., -1] > 0).astype(self.raster_dtype) * self.raster_value

    def _map_from_binaries(self, eopatch, dst_shape, request_data):
        """
        Each request represents a binary class which will be mapped to the scalar `raster_value`
        """
        if eopatch.feature_exists(self.feature_type, self.feature_name):
            raster = eopatch.get_feature(self.feature_type, self.feature_name).squeeze()
        else:
            raster = np.ones(dst_shape, dtype=self.raster_dtype) * self.no_data_val

        new_raster = self._reproject(eopatch, self._to_binary_mask(request_data))

        # update raster
        raster[new_raster != 0] = new_raster[new_raster != 0]

        return raster

    def _map_from_multiclass(self, eopatch, dst_shape, request_data):
        """
        `raster_value` is a dictionary specifying the intensity values for each class and the corresponding label value.

        A dictionary example for GLC30 LULC mapping is:
        raster_value = {'no_data': (0,[0,0,0,0]),
                        'cultivated land': (1,[193, 243, 249, 255]),
                        'forest': (2,[73, 119, 20, 255]),
                        'grassland': (3,[95, 208, 169, 255]),
                        'schrubland': (4,[112, 179, 62, 255]),
                        'water': (5,[154, 86, 1, 255]),
                        'wetland': (6,[244, 206, 126, 255]),
                        'thundra': (7,[50, 100, 100, 255]),
                        'artificial surface': (8,[20, 47, 147, 255]),
                        'bareland': (9,[202, 202, 202, 255]),
                        'snow and ice': (10,[251, 237, 211, 255])}
        """
        raster = np.ones(dst_shape, dtype=self.raster_dtype) * self.no_data_val

        for key in self.raster_value.keys():
            value, intensities = self.raster_value[key]
            raster[np.mean(np.abs(request_data - intensities), axis=-1) < self.mean_abs_difference] = value

        return self._reproject(eopatch, raster)

    def execute(self, eopatch):
        """
        Add requested feature to this existing EOPatch.
        """
        data_arr = eopatch.get_feature(FeatureType.MASK, 'IS_DATA')
        _, height, width, _ = data_arr.shape

        request = self._get_wms_request(eopatch.bbox, width, height)

        request_data, = np.asarray(request.get_data())

        if isinstance(self.raster_value, dict):
            raster = self._map_from_multiclass(eopatch, (height, width), request_data)
        elif isinstance(self.raster_value, (int, float)):
            raster = self._map_from_binaries(eopatch, (height, width), request_data)
        else:
            raise ValueError("Unsupported raster value type")

        if (self.feature_type in [FeatureType.MASK_TIMELESS]) and raster.ndim == 2:
            raster = raster[..., np.newaxis]

        eopatch.add_feature(self.feature_type, self.feature_name, raster)

        return eopatch
