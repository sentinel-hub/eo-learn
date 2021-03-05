""" An input task for the `sentinelhub processing api <https://docs.sentinel-hub.com/api/latest/reference/>`
"""
import collections
import logging
import datetime as dt
from itertools import repeat

import numpy as np
from sentinelhub import SentinelHubRequest, WebFeatureService, MimeType, SentinelHubDownloadClient, SHConfig, \
    bbox_to_dimensions, parse_time_interval, DataCollection
from sentinelhub.data_collections import handle_deprecated_data_source

from eolearn.core import EOPatch, EOTask, FeatureType

LOGGER = logging.getLogger(__name__)


def get_available_timestamps(bbox, config, data_collection, maxcc, time_difference, time_interval):
    """Helper function to search for all available timestamps, based on query parameters

    :param bbox: Bounding box
    :type bbox: BBox
    :param time_interval: Time interval to query available satellite data from
    type time_interval: different input formats available (e.g. (str, str), or (datetime, datetime)
    :param data_collection: Source of requested satellite data.
    :type data_collection: DataCollection
    :param maxcc: Maximum cloud coverage.
    :type maxcc: float
    :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
    :type time_difference: datetime.timedelta
    :param config: Sentinel Hub Config
    :type config: SHConfig
    :return: list of datetimes with available observations
    """
    wfs = WebFeatureService(bbox=bbox, time_interval=time_interval, data_collection=data_collection, maxcc=maxcc,
                            config=config)
    dates = wfs.get_dates()

    if len(dates) == 0:
        raise ValueError("No available images for requested time range: {}".format(time_interval))

    dates = sorted(dates)
    return [dates[0]] + [d2 for d1, d2 in zip(dates[:-1], dates[1:]) if d2 - d1 > time_difference]


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
        """ Main execute method for the Processing API tasks
        """

        eopatch = eopatch or EOPatch()

        self._check_and_set_eopatch_bbox(bbox, eopatch)

        if self.size is not None:
            size_x, size_y = self.size
        elif self.resolution is not None:
            size_x, size_y = bbox_to_dimensions(eopatch.bbox, self.resolution)

        if time_interval:
            time_interval = parse_time_interval(time_interval)
            timestamp = self._get_timestamp(time_interval, eopatch.bbox)
        else:
            timestamp = eopatch.timestamp

        if eopatch.timestamp:
            self.check_timestamp_difference(timestamp, eopatch.timestamp)
        elif timestamp:
            eopatch.timestamp = timestamp

        requests = self._build_requests(eopatch.bbox, size_x, size_y, timestamp, time_interval)
        requests = [request.download_list[0] for request in requests]

        LOGGER.debug('Downloading %d requests of type %s', len(requests), str(self.data_collection))
        client = SentinelHubDownloadClient(config=self.config)
        images = client.download(requests, max_threads=self.max_threads)
        LOGGER.debug('Downloads complete')

        temporal_dim = len(timestamp) if timestamp else 1
        shape = temporal_dim, size_y, size_x
        self._extract_data(eopatch, images, shape)

        eopatch.meta_info['size_x'] = size_x
        eopatch.meta_info['size_y'] = size_y
        eopatch.meta_info['time_interval'] = time_interval
        eopatch.meta_info['service_type'] = 'processing'

        self._add_meta_info(eopatch)

        return eopatch

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

    def _add_meta_info(self, eopatch):
        """ Add any additional meta data to the eopatch
        """


ProcApiType = collections.namedtuple('ProcApiType', 'id unit sample_type np_dtype feature_type')


class SentinelHubInputTask(SentinelHubInputBase):
    """ A processing API input task that loads 16bit integer data and converts it to a 32bit float feature.
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
                 additional_data=None, evalscript=None, maxcc=1.0, time_difference=None, cache_folder=None,
                 max_threads=None, config=None, bands_dtype=np.float32, single_scene=False,
                 mosaicking_order='mostRecent', aux_request_args=None, data_source=None):
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
        super().__init__(
            data_collection=data_collection, size=size, resolution=resolution, cache_folder=cache_folder, config=config,
            max_threads=max_threads, data_source=data_source
        )
        self.evalscript = evalscript
        self.maxcc = maxcc
        self.time_difference = dt.timedelta(seconds=1) if time_difference is None else time_difference
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

    def _add_meta_info(self, eopatch):
        """ Add any additional meta data to the eopatch
        """
        eopatch.meta_info['maxcc'] = self.maxcc
        eopatch.meta_info['time_difference'] = self.time_difference


class SentinelHubDemTask(SentinelHubInputBase):
    """ A processing API input task that downloads the digital elevation model
    """

    def __init__(self, dem_feature, size=None, resolution=None, cache_folder=None, config=None,
                 max_threads=None):
        """
        :param dem_feature: Target feature into which to save the DEM array.
        :type dem_feature: tuple(sentinelhub.FeatureType, str)
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
        """

        super().__init__(
            data_collection=DataCollection.DEM, size=size, resolution=resolution, cache_folder=cache_folder,
            config=config, max_threads=max_threads
        )

        feature_parser = self._parse_features(
            dem_feature,
            default_feature_type=FeatureType.DATA_TIMELESS,
            allowed_feature_types=[FeatureType.DATA_TIMELESS]
        )

        self.dem_feature = next(feature_parser())

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build requests
        """
        evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: ["DEM"],
                    output:{
                        id: "default",
                        bands: 1,
                        sampleType: SampleType.UINT16
                    }
                }
            }

            function evaluatePixel(sample) {
                return [sample.DEM]
            }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=self.data_collection)],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config
        )

        return [request]

    def _extract_data(self, eopatch, images, shape):
        """ Extract data from the received images and assign them to eopatch features
        """
        tif = images[0]
        eopatch[self.dem_feature] = tif[..., np.newaxis].astype(np.int16)
