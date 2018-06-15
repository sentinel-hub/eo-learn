"""
Module for creating new EOPatches with data obtained from sentinelhub package
"""

import datetime
import numpy as np
import logging

from sentinelhub import WmsRequest, WcsRequest, MimeType, DataSource, CustomUrlParam, ServiceType

from eolearn.core import EOPatch, EOTask

LOGGER = logging.getLogger(__name__)


class SentinelHubOGCInput(EOTask):
    """
    Task for creating EOPatch and filling it with data using Sentinel Hub's OGC request.

    :param layer: the preconfigured layer to be added to EOPatch's DATA feature. Required.
    :type layer: str
    :param feature_name: user specified name (key) for this feature. Optional.
    :type feature_name: str or None. Default is the same as layer.
    :param valid_data_mask_name: user specified name (key) for the valid data mask returned by the OGC request.
                                 Optional.
    :type valid_data_mask_name: str. Default is `'IS_DATA'`.
    :param service_type: type of OGC service (WMS or WCS)
    :type service_type: constants.ServiceType
    :param data_source: Source of requested satellite data.
    :type data_source: constants.DataSource
    :param size_x: number of pixels in x or resolution in x (i.e. ``512`` or ``10m``)
    :type size_x: int or str, depends on the service_type
    :param size_y: number of pixels in x or resolution in y (i.e. ``512`` or ``10m``)
    :type size_y: int or str, depends on the service_type
    :param maxcc: maximum accepted cloud coverage of an image. Float between 0.0 and 1.0. Default is ``1.0``.
    :type maxcc: float
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
                              ``{constants.CustomUrlParam.ATMFILTER: 'ATMCOR'}``
    :param time_difference: The time difference below which dates are deemed equal. That is, if for the given set of OGC
                            parameters the images are available at datestimes `d1<=d2<=...<=dn` then only those with
                            `dk-dj>time_difference` will be considered. The default time difference is negative (`-1s`),
                            meaning that all dates are considered by default.
    :type time_difference: datetime.timedelta
    """

    def __init__(self, layer, feature_name=None, valid_data_mask_name='IS_DATA',
                 service_type=None, data_source=None,
                 size_x=None, size_y=None, maxcc=1.0, image_format=MimeType.TIFF_d32f,
                 instance_id=None, custom_url_params=None, time_difference=datetime.timedelta(seconds=-1)):
        # pylint: disable=too-many-arguments
        self.layer = layer
        self.feature_name = layer if feature_name is None else feature_name
        self.valid_data_mask_name = valid_data_mask_name
        self.service_type = service_type
        self.data_source = data_source
        self.size_x = size_x
        self.size_y = size_y
        self.maxcc = maxcc
        self.image_format = image_format
        self.instance_id = instance_id

        custom_params = {CustomUrlParam.SHOWLOGO: False,
                         CustomUrlParam.TRANSPARENT: True}
        if custom_url_params is None:
            self.custom_url_params = custom_params
        else:
            self.custom_url_params = {**custom_params, **custom_url_params}

        self.time_difference = time_difference

    def _get_wms_request(self, bbox, time_interval):
        """
        Returns WMS request with given BBOX and time_interval.
        """
        return WmsRequest(layer=self.layer,
                          bbox=bbox,
                          time=time_interval,
                          width=self.size_x, height=self.size_y,
                          maxcc=self.maxcc,
                          image_format=self.image_format,
                          custom_url_params=self.custom_url_params,
                          time_difference=self.time_difference,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _get_wcs_request(self, bbox, time_interval):
        """
        Returns WCS request with given BBOX and time_interval.
        """
        return WcsRequest(layer=self.layer,
                          bbox=bbox,
                          time=time_interval,
                          resx=self.size_x, resy=self.size_y,
                          maxcc=self.maxcc,
                          image_format=self.image_format,
                          custom_url_params=self.custom_url_params,
                          time_difference=self.time_difference,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def execute(self, bbox, time_interval):
        """
        Creates OGC (WMS or WCS) request, downloads requested data and stores it together
        with valid data mask in newly created EOPatch. Returns the EOPatch.

        :param bbox: specifies the bounding box of the requested image. Coordinates must be in
                     the specified coordinate reference system. Required.
        :type bbox: common.BBox
        :param time_interval: time or time range for which to return the results, in ISO8601 format
                              (year-month-date, for example: ``2016-01-01``, or year-month-dateThours:minuts:seconds
                              format, i.e. ``2016-01-01T16:31:21``). When a single time is specified the request will
                              return data for that specific date, if it exists. If a time range is specified the result
                              is a list of all scenes between the specified dates conforming to the cloud coverage
                              criteria. Most recent acquisition being first in the list. For the latest acquisition use
                              ``latest``. Examples: ``latest``, ``'2016-01-01'``, or ``('2016-01-01', ' 2016-01-31')``
         :type time_interval: str, or tuple of str
        """
        request = {ServiceType.WMS: self._get_wms_request,
                   ServiceType.WCS: self._get_wcs_request}[self.service_type](bbox, time_interval)

        request_return = request.get_data(raise_download_errors=False)
        timestamps = request.get_dates()

        bad_data = [idx for idx, value in enumerate(request_return) if value is None]
        for idx in reversed(sorted(bad_data)):
            LOGGER.warning(f'Data from {timestamps[idx]} could not be downloaded for {self.layer}!')
            del request_return[idx]
            del timestamps[idx]

        request_data = np.asarray(request_return)

        data = request_data[..., :-1]
        if data.ndim == 3:
            data = data.reshape(data.shape + (1,))
        valid_data = (request_data[..., -1] > 0.5).astype(np.int).reshape(request_data[..., -1].shape + (1,))

        eop_data = {self.feature_name: data}
        eop_mask = {self.valid_data_mask_name: valid_data}

        meta_info = {'size_x': self.size_x, 'size_y': self.size_y, 'maxcc': self.maxcc,
                     'time_difference': self.time_difference,
                     'service_type': self.service_type,
                     'time_interval': time_interval}

        return EOPatch(bbox=bbox, timestamp=timestamps, data=eop_data, mask=eop_mask, meta_info=meta_info)


class SentinelHubWMSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, data_source=None, width=None, height=None, **kwargs):
        super(SentinelHubWMSInput, self).__init__(layer=layer, data_source=data_source, service_type=ServiceType.WMS,
                                                  size_x=width, size_y=height, **kwargs)


class SentinelHubWCSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, data_source=None, resx=None, resy=None, **kwargs):
        super(SentinelHubWCSInput, self).__init__(layer=layer, data_source=data_source, service_type=ServiceType.WCS,
                                                  size_x=resx, size_y=resy, **kwargs)


class S2L1CWMSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L1C data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, width=None, height=None, **kwargs):
        super(S2L1CWMSInput, self).__init__(layer=layer, data_source=DataSource.SENTINEL2_L1C,
                                            service_type=ServiceType.WMS, size_x=width, size_y=height, **kwargs)


class S2L1CWCSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L1C data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, resx=None, resy=None, **kwargs):
        super(S2L1CWCSInput, self).__init__(layer=layer, data_source=DataSource.SENTINEL2_L1C,
                                            service_type=ServiceType.WCS, size_x=resx, size_y=resy, **kwargs)


class L8L1CWMSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with Landsat-8 L1C data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, width=None, height=None, **kwargs):
        super(L8L1CWMSInput, self).__init__(layer=layer, data_source=DataSource.LANDSAT8, service_type=ServiceType.WMS,
                                            size_x=width, size_y=height, **kwargs)


class L8L1CWCSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with Landsat-8 L1C data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, resx=None, resy=None, **kwargs):
        super(L8L1CWCSInput, self).__init__(layer=layer, data_source=DataSource.LANDSAT8, service_type=ServiceType.WCS,
                                            size_x=resx, size_y=resy, **kwargs)


class S2L2AWMSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L2A data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, width=None, height=None, **kwargs):
        super(S2L2AWMSInput, self).__init__(layer=layer, data_source=DataSource.SENTINEL2_L2A,
                                            service_type=ServiceType.WMS, size_x=width, size_y=height, **kwargs)


class S2L2AWCSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L2A data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, resx=None, resy=None, **kwargs):
        super(S2L2AWCSInput, self).__init__(layer=layer, data_source=DataSource.SENTINEL2_L2A,
                                            service_type=ServiceType.WCS, size_x=resx, size_y=resy, **kwargs)
