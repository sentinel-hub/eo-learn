"""
Module for creating new EOPatches with data obtained from sentinelhub package

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)
Copyright (c) 2018-2019 William Ouellette

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
import datetime as dt

import numpy as np
from sentinelhub import WmsRequest, WcsRequest, MimeType, DataSource, CustomUrlParam, ServiceType

from eolearn.core import EOPatch, EOTask, FeatureType, get_common_timestamps

LOGGER = logging.getLogger(__name__)


class SentinelHubOGCInput(EOTask):
    """
    Task for creating EOPatch and filling it with data using Sentinel Hub's OGC request.

    :param layer: the preconfigured layer to be added to EOPatch's DATA feature.
    :type layer: str
    :param feature: Feature to which the data will be added. By default the name will be the same as the name of the
        layer
    :type feature: str or (FeatureType, str) or None
    :param valid_data_mask_feature: A feature to which valid data mask will be stored. Default is `'IS_DATA'`.
    :type valid_data_mask_feature: str or (FeatureType, str)
    :param service_type: type of OGC service (WMS or WCS)
    :type service_type: ServiceType
    :param data_source: Source of requested satellite data.
    :type data_source: DataSource
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
    :param raise_download_errors: `True` if any errors during download should stop the task and `False` if the task
        should continue and remove timestamps with frames that failed to download.
    :type raise_download_errors: bool
    """

    def __init__(self, layer, feature=None, valid_data_mask_feature='IS_DATA', service_type=None, data_source=None,
                 size_x=None, size_y=None, maxcc=None, image_format=MimeType.TIFF_d32f, instance_id=None,
                 custom_url_params=None, time_difference=None, raise_download_errors=True):
        # pylint: disable=too-many-arguments
        self.layer = layer
        self.feature_type, self.feature_name = next(self._parse_features(layer if feature is None else feature,
                                                                         default_feature_type=FeatureType.DATA)())
        self.valid_data_mask_feature = self._parse_features(valid_data_mask_feature,
                                                            default_feature_type=FeatureType.MASK)
        self.service_type = service_type
        self.data_source = data_source
        self.size_x = size_x
        self.size_y = size_y
        self.maxcc = maxcc
        self.image_format = image_format
        self.instance_id = instance_id

        if custom_url_params is None:
            custom_url_params = {}
        self.custom_url_params = {**{CustomUrlParam.SHOWLOGO: False,
                                     CustomUrlParam.TRANSPARENT: True},
                                  **custom_url_params}

        self.time_difference = time_difference
        self.raise_download_errors = raise_download_errors

    def _get_parameter(self, name, eopatch):
        """ Collects the parameter either from initialization parameters or from EOPatch
        """
        if hasattr(self, name) and getattr(self, name) is not None:
            return getattr(self, name)
        if name == 'bbox' and eopatch.bbox:
            return eopatch.bbox
        if name in eopatch.meta_info:
            return eopatch.meta_info[name]
        if name == 'maxcc':
            return 1.0
        if name == 'time_difference':
            return dt.timedelta(seconds=-1)
        if name in ('size_x', 'size_y'):
            return None

        raise ValueError('Parameter {} was neither defined in initialization of {} nor is contained in '
                         'EOPatch'.format(name, self.__class__.__name__))

    def _prepare_request_data(self, eopatch, bbox, time_interval):
        """ Collects all parameters used for DataRequest, each one is taken either from initialization parameters or
        from EOPatch
        """

        service_type = ServiceType(self._get_parameter('service_type', eopatch))
        if time_interval is None:
            time_interval = self._get_parameter('time_interval', eopatch)
        if service_type is ServiceType.WMS:
            size_x_name, size_y_name = 'width', 'height'
        else:
            size_x_name, size_y_name = 'resx', 'resy'
        return {
            'layer': self.layer,
            'bbox': bbox if bbox is not None else self._get_parameter('bbox', eopatch),
            'time': time_interval,
            'time_difference': self._get_parameter('time_difference', eopatch),
            'maxcc': self._get_parameter('maxcc', eopatch),
            'image_format': self.image_format,
            'custom_url_params': self.custom_url_params,
            'data_source': self.data_source,
            'instance_id': self.instance_id,
            size_x_name: self._get_parameter('size_x', eopatch),
            size_y_name: self._get_parameter('size_y', eopatch)
        }, service_type

    def _add_data(self, eopatch, data):
        """ Adds downloaded data to EOPatch """
        valid_mask = data[..., -1]
        data = data[..., :-1]

        if data.ndim == 3:
            data = data.reshape(data.shape + (1,))
        if not self.feature_type.is_time_dependent():
            if data.shape[0] > 1:
                raise ValueError('Cannot save time dependent data to time independent feature')
            data = data.squeeze(axis=0)
        if self.feature_type.is_discrete():
            data = data.astype(np.int32)

        eopatch[self.feature_type][self.feature_name] = data

        mask_feature_type, mask_feature_name = next(self.valid_data_mask_feature())

        max_value = self.image_format.get_expected_max_value()
        valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape + (1,))

        if mask_feature_name not in eopatch[mask_feature_type]:
            eopatch[mask_feature_type][mask_feature_name] = valid_data

    def _add_meta_info(self, eopatch, request_params, service_type):
        """ Adds any missing metadata info to EOPatch """

        for param, eoparam in zip(['time', 'time_difference', 'maxcc'], ['time_interval', 'time_difference', 'maxcc']):
            if eoparam not in eopatch.meta_info:
                eopatch.meta_info[eoparam] = request_params[param]

        if 'service_type' not in eopatch.meta_info:
            eopatch.meta_info['service_type'] = service_type.value

        for param in ['size_x', 'size_y']:
            if param not in eopatch.meta_info:
                eopatch.meta_info[param] = getattr(self, param)

        if eopatch.bbox is None:
            eopatch.bbox = request_params['bbox']

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """
        Creates OGC (WMS or WCS) request, downloads requested data and stores it together
        with valid data mask in newly created EOPatch. Returns the EOPatch.

        :param eopatch:
        :type eopatch: EOPatch or None
        :param bbox: specifies the bounding box of the requested image. Coordinates must be in
                     the specified coordinate reference system. Required.
        :type bbox: BBox
        :param time_interval: time or time range for which to return the results, in ISO8601 format
                              (year-month-date, for example: ``2016-01-01``, or year-month-dateThours:minutes:seconds
                              format, i.e. ``2016-01-01T16:31:21``). When a single time is specified the request will
                              return data for that specific date, if it exists. If a time range is specified the result
                              is a list of all scenes between the specified dates conforming to the cloud coverage
                              criteria. Most recent acquisition being first in the list. For the latest acquisition use
                              ``latest``. Examples: ``latest``, ``'2016-01-01'``, or ``('2016-01-01', ' 2016-01-31')``
         :type time_interval: datetime.datetime, str, or tuple of datetime.datetime/str
        """
        if eopatch is None:
            eopatch = EOPatch()

        request_params, service_type = self._prepare_request_data(eopatch, bbox, time_interval)
        request = {ServiceType.WMS: WmsRequest,
                   ServiceType.WCS: WcsRequest}[service_type](**request_params)

        request_dates = request.get_dates()

        if not eopatch.timestamp:
            eopatch.timestamp = request_dates

        download_frames = None
        if self.feature_type.is_time_dependent():
            download_frames = get_common_timestamps(request_dates, eopatch.timestamp)

        images = request.get_data(raise_download_errors=self.raise_download_errors, data_filter=download_frames)

        if not self.raise_download_errors:
            bad_data = [idx for idx, value in enumerate(images) if value is None]
            for idx in reversed(bad_data):
                LOGGER.warning('Data from %s could not be downloaded for %s!', str(request_dates[idx]), self.layer)
                del images[idx]
                del request_dates[idx]

            for removed_frame in eopatch.consolidate_timestamps(request_dates):
                LOGGER.warning('Removed data for frame %s from EOPatch '
                               'due to unavailability of %s!', str(removed_frame), self.layer)

        self._add_data(eopatch, np.asarray(images))
        self._add_meta_info(eopatch, request_params, service_type)
        return eopatch


class SentinelHubWMSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with data using Sentinel Hub's WMS request.
    """

    def __init__(self, layer, data_source=None, width=None, height=None, **kwargs):
        super().__init__(layer=layer, data_source=data_source, service_type=ServiceType.WMS,
                         size_x=width, size_y=height, **kwargs)


class SentinelHubWCSInput(SentinelHubOGCInput):
    """
    Task for creating EOPatches and filling them with data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, data_source=None, resx=None, resy=None, **kwargs):
        super().__init__(layer=layer, data_source=data_source, service_type=ServiceType.WCS,
                         size_x=resx, size_y=resy, **kwargs)


class S2L1CWMSInput(SentinelHubWMSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L1C data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.SENTINEL2_L1C, **kwargs)


class S2L1CWCSInput(SentinelHubWCSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L1C data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.SENTINEL2_L1C, **kwargs)


class L8L1CWMSInput(SentinelHubWMSInput):
    """
    Task for creating EOPatches and filling them with Landsat-8 L1C data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.LANDSAT8, **kwargs)


class L8L1CWCSInput(SentinelHubWCSInput):
    """
    Task for creating EOPatches and filling them with Landsat-8 L1C data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.LANDSAT8, **kwargs)


class S2L2AWMSInput(SentinelHubWMSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L2A data using Sentinel Hub's WMS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.SENTINEL2_L2A, **kwargs)


class S2L2AWCSInput(SentinelHubWCSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-2 L2A data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.SENTINEL2_L2A, **kwargs)


class S1IWWMSInput(SentinelHubWMSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-1 IW GRD data using Sentinel Hub's WMS request.

    :param orbit: String specifying the orbit hte data belongs to. Options are `'both'`, `'ascending'` and
        `'descending'`. Default is `'both'`
    :type orbit: str
    """
    def __init__(self, layer, orbit='both', **kwargs):
        data_source = {'both': DataSource.SENTINEL1_IW,
                       'ascending': DataSource.SENTINEL1_IW_ASC,
                       'descending': DataSource.SENTINEL1_IW_DES}[orbit]
        super().__init__(layer=layer, data_source=data_source, **kwargs)


class S1IWWCSInput(SentinelHubWCSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-1 IW GRD data using Sentinel Hub's WCS request.

    :param orbit: String specifying the orbit hte data belongs to. Options are `'both'`, `'ascending'` and
        `'descending'`. Default is `'both'`
    :type orbit: str
    """
    def __init__(self, layer, orbit='both', **kwargs):
        data_source = {'both': DataSource.SENTINEL1_IW,
                       'ascending': DataSource.SENTINEL1_IW_ASC,
                       'descending': DataSource.SENTINEL1_IW_DES}[orbit]
        super().__init__(layer=layer, data_source=data_source, **kwargs)


class DEMWMSInput(SentinelHubWMSInput):
    """
    Adds DEM to DATA_TIMELESS EOPatch feature.
    """
    def __init__(self, layer, feature=None, **kwargs):
        if feature is None:
            feature = (FeatureType.DATA_TIMELESS, layer)
        elif isinstance(feature, str):
            feature = (FeatureType.DATA_TIMELESS, feature)
        super().__init__(layer=layer, feature=feature, data_source=DataSource.DEM, **kwargs)


class DEMWCSInput(SentinelHubWCSInput):
    """
    Adds DEM to DATA_TIMELESS EOPatch feature.
    """
    def __init__(self, layer, feature=None, **kwargs):
        if feature is None:
            feature = (FeatureType.DATA_TIMELESS, layer)
        elif isinstance(feature, str):
            feature = (FeatureType.DATA_TIMELESS, feature)
        super().__init__(layer=layer, feature=feature, data_source=DataSource.DEM, **kwargs)


class AddSen2CorClassificationFeature(SentinelHubOGCInput):
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
        classification_types = {
            'SCL': FeatureType.MASK,
            'CLD': FeatureType.DATA,
            'SNW': FeatureType.DATA
        }

        if sen2cor_classification not in classification_types:
            raise ValueError('Unsupported Sen2Cor classification type: {}.'
                             ' Possible types are: {}'.format(sen2cor_classification, classification_types))

        evalscript = 'return [{}];'.format(sen2cor_classification)

        super().__init__(feature=(classification_types[sen2cor_classification], sen2cor_classification),
                         layer=layer, data_source=DataSource.SENTINEL2_L2A,
                         custom_url_params={CustomUrlParam.EVALSCRIPT: evalscript}, **kwargs)
