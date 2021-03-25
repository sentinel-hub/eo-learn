""" Input tasks that collect data from `Sentinel-Hub Process API
<https://docs.sentinel-hub.com/api/latest/api/process/>`__

Credits:
Copyright (c) 2019-2021 Matej Aleksandrov, Matej Batič, Matic Lubej, Jovan Višnjić (Sinergise)
Copyright (c) 2019-2021 Beno Šircelj

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import collections
import datetime as dt
import logging
from itertools import repeat

import numpy as np
from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet

from sentinelhub import DataCollection, MimeType, SHConfig, SentinelHubCatalog, SentinelHubDownloadClient, \
    SentinelHubRequest, bbox_to_dimensions, filter_times, parse_time_interval
from sentinelhub.data_collections import handle_deprecated_data_source

LOGGER = logging.getLogger(__name__)


def get_available_timestamps(bbox, config, data_collection, time_difference, time_interval=None, maxcc=None):
    """Helper function to search for all available timestamps, based on query parameters

    :param bbox: Bounding box
    :type bbox: BBox
    :param time_interval: Time interval to query available satellite data from
        type time_interval: different input formats available (e.g. (str, str), or (datetime, datetime)
    :param data_collection: Source of requested satellite data.
    :type data_collection: DataCollection
    :param maxcc: Maximum cloud coverage, in ratio [0, 1], default is None
    :type maxcc: float
    :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
    :type time_difference: datetime.timedelta
    :param config: Sentinel Hub Config
    :type config: SHConfig
    :return: list of datetimes with available observations
    """

    query = None
    if maxcc and data_collection.has_cloud_coverage:
        if isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError('Maximum cloud coverage "maxcc" parameter should be a float on an interval [0, 1]')
        query = {'eo:cloud_cover': {'lte': int(maxcc * 100)}}

    fields = {'include': ['properties.datetime'], 'exclude': []}

    catalog = SentinelHubCatalog(base_url=data_collection.service_url, config=config)
    search_iterator = catalog.search(collection=data_collection, bbox=bbox, time=time_interval,
                                     query=query, fields=fields)

    all_timestamps = search_iterator.get_timestamps()
    filtered_timestamps = filter_times(all_timestamps, time_difference)

    if len(filtered_timestamps) == 0:
        raise ValueError("No available images for requested time range: {}".format(time_interval))

    return filtered_timestamps


class SentinelHubInputBase(EOTask):
    """ Base class for Processing API input tasks
    """

    def __init__(self, data_collection, size=None, resolution=None, cache_folder=None, config=None, max_threads=None,
                 data_source=None):
        """
        :param data_collection: A collection of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param data_source: A deprecated alternative to data_collection
        :type data_source: DataCollection
        """
        if (size is None) == (resolution is None):
            raise ValueError("Exactly one of the parameters 'size' and 'resolution' should be given.")

        self.size = size
        self.resolution = resolution
        self.config = config or SHConfig()
        self.max_threads = max_threads
        self.data_collection = DataCollection(handle_deprecated_data_source(data_collection, data_source))
        self.cache_folder = cache_folder

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """ Main execute method for the Process API tasks
        """

        eopatch = eopatch or EOPatch()

        self._check_and_set_eopatch_bbox(bbox, eopatch)
        size_x, size_y = self._get_size(eopatch)

        if time_interval:
            time_interval = parse_time_interval(time_interval)
            timestamp = self._get_timestamp(time_interval, eopatch.bbox)
        else:
            timestamp = eopatch.timestamp

        if eopatch.timestamp and timestamp:
            self.check_timestamp_difference(timestamp, eopatch.timestamp)
        elif timestamp:
            eopatch.timestamp = timestamp

        requests = self._build_requests(eopatch.bbox, size_x, size_y, timestamp, time_interval)
        requests = [request.download_list[0] for request in requests]

        LOGGER.debug('Downloading %d requests of type %s', len(requests), str(self.data_collection))
        client = SentinelHubDownloadClient(config=self.config)
        responses = client.download(requests, max_threads=self.max_threads)
        LOGGER.debug('Downloads complete')

        temporal_dim = len(timestamp) if timestamp else 1
        shape = temporal_dim, size_y, size_x
        self._extract_data(eopatch, responses, shape)

        eopatch.meta_info['size_x'] = size_x
        eopatch.meta_info['size_y'] = size_y
        if timestamp:  # do not overwrite time interval in case of timeless features
            eopatch.meta_info['time_interval'] = time_interval

        self._add_meta_info(eopatch)

        return eopatch

    def _get_size(self, eopatch):
        """Get the size (width, height) for the request either from inputs, or from the (existing) eopatch"""
        if self.size is not None:
            return self.size

        if self.resolution is not None:
            return bbox_to_dimensions(eopatch.bbox, self.resolution)

        if eopatch.meta_info and eopatch.meta_info.get('size_x') and eopatch.meta_info.get('size_y'):
            return eopatch.meta_info.get('size_x'), eopatch.meta_info.get('size_y')

        raise ValueError('Size or resolution for the requests should be provided!')

    def _add_meta_info(self, eopatch):
        """Add information to eopatch metadata"""
        if self.maxcc:
            eopatch.meta_info['maxcc'] = self.maxcc
        if self.time_difference:
            eopatch.meta_info['time_difference'] = self.time_difference

    @staticmethod
    def _check_and_set_eopatch_bbox(bbox, eopatch):
        if eopatch.bbox is None:
            if bbox is None:
                raise ValueError('Either the eopatch or the task must provide valid bbox.')
            eopatch.bbox = bbox
            return

        if bbox is None or eopatch.bbox == bbox:
            return
        raise ValueError('Either the eopatch or the task must provide bbox, or they must be the same.')

    @staticmethod
    def check_timestamp_difference(timestamp1, timestamp2):
        """ Raises an error if the two timestamps are not the same
        """
        error_msg = "Trying to write data to an existing eopatch with a different timestamp."
        if len(timestamp1) != len(timestamp2):
            raise ValueError(error_msg)

        for ts1, ts2 in zip(timestamp1, timestamp2):
            if ts1 != ts2:
                raise ValueError(error_msg)

    def _extract_data(self, eopatch, images, shape):
        """ Extract data from the received images and assign them to eopatch features
        """
        raise NotImplementedError("The _extract_data method should be implemented by the subclass.")

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build requests
        """
        raise NotImplementedError("The _build_requests method should be implemented by the subclass.")

    def _get_timestamp(self, time_interval, bbox):
        """ Get the timestamp array needed as a parameter for downloading the images
        """


ProcApiType = collections.namedtuple('ProcApiType', 'id unit sample_type np_dtype feature_type')


class SentinelHubEvalscriptTask(SentinelHubInputBase):
    """ Process API task to download data using evalscript
    """

    def __init__(self, features=None, evalscript=None, data_collection=None, size=None, resolution=None,
                 maxcc=None, time_difference=None, cache_folder=None,
                 max_threads=None, config=None, mosaicking_order=None, aux_request_args=None):
        """
        :param features: Features to construct from the evalscript.
        :param evalscript: Evascript for the request. Beware that all outputs from SentinelHub services should be named
            and should have the same name as corresponding feature
        :type evalscript: str
        :param data_collection: Source of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param maxcc: Maximum cloud coverage, a float in interval [0, 1]
        :type maxcc: float
        :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
        :type time_difference: datetime.timedelta
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :type mosaicking_order: str
        :param aux_request_args: a dictionary with auxiliary information for the input_data part of the SH request
        :type aux_request_args: dict
        """
        super().__init__(data_collection=data_collection, size=size, resolution=resolution, cache_folder=cache_folder,
                         config=config, max_threads=max_threads)

        self.features = self._parse_and_validate_features(features)
        self.responses = self._create_response_objects()

        if not evalscript:
            raise ValueError('evalscript parameter must not be missing/empty')
        self.evalscript = evalscript

        if maxcc and isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError('maxcc should be a float on an interval [0, 1]')

        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.mosaicking_order = mosaicking_order
        self.aux_request_args = aux_request_args

    def _parse_and_validate_features(self, features):
        if not features:
            raise ValueError('features must be defined')

        allowed_features = FeatureTypeSet.RASTER_TYPES.union({FeatureType.META_INFO})
        _features = list(self._parse_features(features, allowed_feature_types=allowed_features, new_names=True)())

        ftr_data_types = set(ft for ft, _, _ in _features if not ft.is_meta())
        if all(ft.is_timeless() for ft in ftr_data_types) or all(ft.is_time_dependent() for ft in ftr_data_types):
            return _features

        raise ValueError('Cannot mix time dependent and timeless requests!')

    def _create_response_objects(self):
        """ Construct SentinelHubRequest output_responses from features
        """
        responses = []
        for feat_type, feat_name, _ in self.features:
            if feat_type.is_raster():
                responses.append(SentinelHubRequest.output_response(feat_name, MimeType.TIFF))
            elif feat_type.is_meta():
                responses.append(SentinelHubRequest.output_response('userdata', MimeType.JSON))
            else:
                # should not happen as features have already been validated
                raise ValueError(f'{feat_type} not supported!')

        return responses

    def _get_timestamp(self, time_interval, bbox):
        """ Get the timestamp array needed as a parameter for downloading the images
        """
        if any(feat_type.is_timeless() for feat_type, _, _ in self.features if feat_type.is_raster()):
            return None

        return get_available_timestamps(bbox=bbox, time_interval=time_interval, data_collection=self.data_collection,
                                        maxcc=self.maxcc, time_difference=self.time_difference, config=self.config)

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build requests
        """
        if timestamp:
            dates = [(date - self.time_difference, date + self.time_difference) for date in timestamp]
        else:
            dates = [parse_time_interval(time_interval, allow_undefined=True)] if time_interval else [None]

        return [self._create_sh_request(date, bbox, size_x, size_y) for date in dates]

    def _create_sh_request(self, time_interval, bbox, size_x, size_y):
        """ Create an instance of SentinelHubRequest
        """
        return SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=self.data_collection,
                mosaicking_order=self.mosaicking_order,
                time_interval=time_interval,
                maxcc=self.maxcc,
                other_args=self.aux_request_args
            )],
            responses=self.responses,
            bbox=bbox,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config
        )

    def _extract_data(self, eopatch, data_responses, shape):
        """ Extract data from the received images and assign them to eopatch features
        """
        if len(self.features) == 1:
            ftype, fname, _ = self.features[0]
            extension = 'json' if ftype.is_meta() else 'tif'
            data_responses = [{f'{fname}.{extension}': data} for data in data_responses]

        for ftype, fname, new_fname in self.features:
            if ftype.is_meta():
                data = [data['userdata.json'] for data in data_responses]

            elif ftype.is_time_dependent():
                data = np.asarray([data[f"{fname}.tif"] for data in data_responses])
                data = data[..., np.newaxis] if data.ndim == 3 else data

            else:
                data = np.asarray(data_responses[0][f"{fname}.tif"])[..., np.newaxis]

            eopatch[ftype][new_fname] = data

        return eopatch


class SentinelHubInputTask(SentinelHubInputBase):
    """ Process API input task that loads 16bit integer data and converts it to a 32bit float feature.
    """
    # pylint: disable=too-many-arguments
    PREDEFINED_BAND_TYPES = {
        ProcApiType("bool_mask", 'DN', 'UINT8', bool, FeatureType.MASK): [
            "dataMask"
        ],
        ProcApiType("mask", 'DN', 'UINT8', np.uint8, FeatureType.MASK): [
            "CLM", "SCL"
        ],
        ProcApiType("uint8_data", 'DN', 'UINT8', np.uint8, FeatureType.DATA): [
            "SNW", "CLD", "CLP"
        ],
        ProcApiType("bands", 'DN', 'UINT16', np.uint16, FeatureType.DATA): [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "B13"
        ],
        ProcApiType("other", 'REFLECTANCE', 'FLOAT32', np.float32, FeatureType.DATA): [
            "sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean"
        ]
    }

    CUSTOM_BAND_TYPE = ProcApiType("custom", 'REFLECTANCE', 'FLOAT32', np.float32, FeatureType.DATA)

    def __init__(self, data_collection=None, size=None, resolution=None, bands_feature=None, bands=None,
                 additional_data=None, evalscript=None, maxcc=None, time_difference=None, cache_folder=None,
                 max_threads=None, config=None, bands_dtype=np.float32, single_scene=False,
                 mosaicking_order=None, aux_request_args=None, data_source=None):
        """
        :param data_collection: Source of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param bands_feature: Target feature into which to save the downloaded images.
        :type bands_feature: tuple(sentinelhub.FeatureType, str)
        :param bands: An array of band names. If not specified it will download all bands specified for a given data
            collection.
        :type bands: list[str]
        :param additional_data: A list of additional data to be downloaded, such as SCL, SNW, dataMask, etc.
        :type additional_data: list[tuple(sentinelhub.FeatureType, str)]
        :param evalscript: An optional parameter to override an evascript that is generated by default
        :type evalscript: str or None
        :param maxcc: Maximum cloud coverage.
        :type maxcc: float
        :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
        :type time_difference: datetime.timedelta
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param bands_dtype: dtype of the bands array
        :type bands_dtype: type
        :param single_scene: If true, the service will compute a single image for the given time interval using
            mosaicking.
        :type single_scene: bool
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :type mosaicking_order: str
        :param aux_request_args: a dictionary with auxiliary information for the input_data part of the SH request
        :type aux_request_args: dict
        :param data_source: A deprecated alternative to data_collection
        :type data_source: DataCollection
        """
        super().__init__(data_collection=data_collection, size=size, resolution=resolution, cache_folder=cache_folder,
                         config=config, max_threads=max_threads, data_source=data_source
        )
        self.evalscript = evalscript
        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.single_scene = single_scene
        self.bands_dtype = bands_dtype
        self.mosaicking_order = mosaicking_order
        self.aux_request_args = aux_request_args

        self.bands_feature = next(self._parse_features(bands_feature, allowed_feature_types=[FeatureType.DATA])()) \
            if bands_feature else None
        self.requested_bands = {}

        if bands_feature:
            if not bands:
                bands = data_collection.bands
            self._add_request_bands(self.requested_bands, bands)

        if additional_data is not None:
            additional_data = list(self._parse_features(additional_data, new_names=True)())
            self._add_request_bands(self.requested_bands, (band for ftype, band, new_name in additional_data))

        self.additional_data = additional_data

    @staticmethod
    def _add_request_bands(request_dict, added_bands):
        predefined_types = SentinelHubInputTask.PREDEFINED_BAND_TYPES.items()

        for band in added_bands:
            found = next(((btype, band) for btype, bands in predefined_types if band in bands), None)
            api_type, band = found or (SentinelHubInputTask.CUSTOM_BAND_TYPE, band)

            if api_type not in request_dict:
                request_dict[api_type] = []

            request_dict[api_type].append(band)

    def generate_evalscript(self):
        """ Generate the evalscript to be passed with the request, based on chosen bands
        """
        evalscript = """
            //VERSION=3

            function setup() {{
                return {{
                    input: [{{
                        bands: [{bands}],
                        units: [{units}]
                    }}],
                    output: [
                        {outputs}
                    ]
                }}
            }}

            function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
                outputMetadata.userData = {{ "norm_factor":  inputMetadata.normalizationFactor }}
            }}

            function evaluatePixel(sample) {{
                return {samples}
            }}
        """

        outputs = [
            "{{ id:{id}, bands:{num_bands}, sampleType: SampleType.{sample_type} }}".format(
                id='\"{}\"'.format(btype.id), num_bands=len(bands), sample_type=btype.sample_type
            )
            for btype, bands in self.requested_bands.items()
        ]

        samples = [
            (btype.id, '[{samples}]'.format(samples=', '.join("sample.{}".format(band) for band in bands)))
            for btype, bands in self.requested_bands.items()
        ]

        # return value of the evaluatePixel has to be a list if we're returning just one output, and a dict otherwise
        # an issue has been reported to the service team and this might get fixed
        if len(samples) == 1:
            _, sample_bands = samples[0]
            samples = sample_bands
        else:
            samples = ', '.join('{band_id}: {bands}'.format(band_id=band_id, bands=bands) for band_id, bands in samples)
            samples = '{{{samples}}};'.format(samples=samples)

        bands = ["\"{}\"".format(band) for bands in self.requested_bands.values() for band in bands]

        units = (unit.unit for btype, bands in self.requested_bands.items() for unit, band in zip(repeat(btype), bands))
        units = ["\"{}\"".format(unit) for unit in units]

        evalscript = evalscript.format(
            bands=', '.join(bands), units=', '.join(units), outputs=', '.join(outputs), samples=samples
        )

        return evalscript

    def _get_timestamp(self, time_interval, bbox):
        """ Get the timestamp array needed as a parameter for downloading the images
        """
        if self.single_scene:
            return [time_interval[0]]

        return get_available_timestamps(bbox=bbox, time_interval=time_interval, data_collection=self.data_collection,
                                        maxcc=self.maxcc, time_difference=self.time_difference, config=self.config)

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build requests
        """
        if self.single_scene:
            dates = [parse_time_interval(time_interval)]
        else:
            dates = [(date - self.time_difference, date + self.time_difference) for date in timestamp]

        return [self._create_sh_request(date1, date2, bbox, size_x, size_y) for date1, date2 in dates]

    def _create_sh_request(self, date_from, date_to, bbox, size_x, size_y):
        """ Create an instance of SentinelHubRequest
        """
        responses = [SentinelHubRequest.output_response(btype.id, MimeType.TIFF) for btype in self.requested_bands]
        responses.append(SentinelHubRequest.output_response('userdata', MimeType.JSON))

        return SentinelHubRequest(
            evalscript=self.evalscript or self.generate_evalscript(),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    time_interval=(date_from, date_to),
                    mosaicking_order=self.mosaicking_order,
                    maxcc=self.maxcc,
                    other_args=self.aux_request_args
                )
            ],
            responses=responses,
            bbox=bbox,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config
        )

    def _extract_data(self, eopatch, images, shape):
        """ Extract data from the received images and assign them to eopatch features
        """
        if self.additional_data:
            self._extract_additional_features(eopatch, images, shape)

        if self.bands_feature:
            self._extract_bands_feature(eopatch, images, shape)

        return eopatch

    def _extract_additional_features(self, eopatch, images, shape):
        """ Extracts additional features from response into an EOPatch
        """
        feature = {band: (ftype, new_name) for ftype, band, new_name in self.additional_data}
        for btype, tifs, bands in self._iter_tifs(images, ['bool_mask', 'mask', 'uint8_data', 'other']):
            for band in bands:
                eopatch[feature[band]] = self._extract_array(tifs, bands.index(band), shape, btype.np_dtype)

    def _extract_bands_feature(self, eopatch, images, shape):
        """ Extract the bands feature arrays and concatenate them along the last axis
        """
        tifs = self._iter_tifs(images, ['bands', 'custom'])
        norms = [(img.get('userdata.json') or {}).get('norm_factor', 1) for img in images]

        itr = [(btype, images, bands, bands.index(band)) for btype, images, bands in tifs for band in bands]
        bands = [self._extract_array(images, idx, shape, self.bands_dtype, norms) for btype, images, band, idx in itr]

        if self.bands_dtype == np.uint16:
            norms = np.asarray(norms).reshape(shape[0], 1).astype(np.float32)
            eopatch[(FeatureType.SCALAR, 'NORM_FACTORS')] = norms

        eopatch[self.bands_feature] = np.concatenate(bands, axis=-1)

    def _iter_tifs(self, tars, band_types):
        rtypes = (btype for btype in self.requested_bands if btype.id in band_types)
        return ((btype, [tar[btype.id + '.tif'] for tar in tars], self.requested_bands[btype]) for btype in rtypes)

    @staticmethod
    def _extract_array(tifs, idx, shape, dtype, norms=None):
        """ Extract a numpy array from the received tifs and normalize it if normalization factors are provided
        """

        feature_arrays = (np.atleast_3d(img)[..., idx] for img in tifs)
        if norms and dtype == np.float32:
            feature_arrays = (np.round(array * norm, 4) for array, norm in zip(feature_arrays, norms))

        return np.asarray(list(feature_arrays), dtype=dtype).reshape(*shape, 1)


class SentinelHubDemTask(SentinelHubEvalscriptTask):
    """
    Adds DEM data (one of the `collections <https://docs.sentinel-hub.com/api/latest/data/dem/#deminstance>`__) to
        DATA_TIMELESS EOPatch feature.
    """

    def __init__(self, feature=None, data_collection=DataCollection.DEM, **kwargs):
        if feature is None:
            feature = (FeatureType.DATA_TIMELESS, 'dem')
        elif isinstance(feature, str):
            feature = (FeatureType.DATA_TIMELESS, feature)

        if feature[0].is_time_dependent():
            raise ValueError("DEM feature should be timeless!")

        ft_name = feature[1]
        evalscript = f"""
            //VERSION=3

            function setup() {{
                return {{
                    input: ["DEM"],
                    output: {{
                        id: "{ft_name}",
                        bands: 1,
                        sampleType: SampleType.UINT16
                    }}
                }}
            }}

            function evaluatePixel(sample) {{
                return {{ {ft_name}: [sample.DEM] }}
            }}
        """

        super().__init__(evalscript=evalscript, features=[feature], data_collection=data_collection, **kwargs)


class SentinelHubSen2corTask(SentinelHubInputTask):
    """
    Adds SCL (scene classification), CLD (cloud probability) or SNW (snow probability) (or any their combination)
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

    def __init__(self, sen2cor_classification, data_collection=DataCollection.SENTINEL2_L2A, **kwargs):
        """
        :param sen2cor_classification: "SCL" (scene classification), "CLD" (cloud probability) or "SNW"
            (snow probability) masks to be retrieved. Also a list of their combination (e.g. ["SCL","CLD"])
        :param sen2cor_classification: str or [str]
        :param kwargs: Additional arguments that will be passed to the `SentinelHubInputTask`
        """
        # definition of possible types and target features
        classification_types = {
            'SCL': FeatureType.MASK,
            'CLD': FeatureType.DATA,
            'SNW': FeatureType.DATA
        }

        if isinstance(sen2cor_classification, str):
            sen2cor_classification = [sen2cor_classification]

        for s2c in sen2cor_classification:
            if s2c not in classification_types:
                raise ValueError(f'Unsupported Sen2Cor classification type: {s2c}. '
                                 f'Possible types are: {classification_types}!')

        if data_collection != DataCollection.SENTINEL2_L2A:
            raise ValueError('Sen2Cor classification layers are only available on Sentinel-2 L2A data.')

        features = [(classification_types[s2c], s2c) for s2c in sen2cor_classification]
        super().__init__(additional_data=features, data_collection=data_collection, **kwargs)
