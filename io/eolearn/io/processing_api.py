""" An input task for the `sentinelhub processing api <https://docs.sentinel-hub.com/api/latest/reference/>`
"""
import json
import logging
import datetime as dt
import numpy as np

from sentinelhub import WebFeatureService, MimeType, SentinelHubDownloadClient, DownloadRequest, SHConfig,\
    bbox_to_dimensions, parse_time_interval, DataSource
import sentinelhub.sentinelhub_request as shr
from sentinelhub.time_utils import iso_to_datetime

from eolearn.core import EOPatch, EOTask, FeatureType

LOGGER = logging.getLogger(__name__)


class SentinelHubInputBase(EOTask):
    """ Base class for Processing API input tasks
    """
    def __init__(self, data_source, size=None, resolution=None, cache_folder=None, config=None, max_threads=None):
        """
        :param data_source: Source of requested satellite data.
        :type data_source: DataSource
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a tuple for X and Y axis.
        :type resolution: tuple(int, int)
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        """

        if (size is None) == (resolution is None):
            raise ValueError("Exactly one of the parameters 'size' and 'resolution' should be given.")

        self.size = size
        self.resolution = resolution
        self.config = config or SHConfig()
        self.max_threads = max_threads
        self.data_source = data_source

        self.request_args = dict(
            url=self.config.get_sh_processing_api_url(),
            headers={"accept": "application/tar", 'content-type': 'application/json'},
            data_folder=cache_folder,
            hash_save=bool(cache_folder),
            request_type='POST',
            data_type=MimeType.TAR
        )

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """ Main execute method for the Processing API tasks
        """

        if eopatch is not None and (bbox or time_interval):
            raise ValueError('Either an eopatch must be provided or bbox and time interval, not both.')

        if eopatch is None:
            eopatch = EOPatch()
            eopatch.bbox = bbox

        if self.size is not None:
            size_x, size_y = self.size
        elif self.resolution is not None:
            size_x, size_y = bbox_to_dimensions(eopatch.bbox, self.resolution)

        if time_interval:
            time_interval = parse_time_interval(time_interval)
            timestamp = self._get_timestamp(time_interval, bbox)
        else:
            timestamp = None

        if eopatch.timestamp:
            self.check_timestamp_difference(timestamp, eopatch.timestamp)
        elif timestamp:
            eopatch.timestamp = timestamp

        payloads = self._build_payloads(bbox, size_x, size_y, timestamp, time_interval)
        requests = [DownloadRequest(post_values=payload, **self.request_args) for payload in payloads]

        LOGGER.debug('Downloading %d requests of type %s', len(requests), str(self.data_source))
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

    def _build_payloads(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build payloads for the requests to the service
        """
        raise NotImplementedError("The _build_payloads method should be implemented by the subclass.")

    def _get_timestamp(self, time_interval, bbox):
        """ Get the timestamp array needed as a parameter for downloading the images
        """

    def _add_meta_info(self, eopatch):
        """ Add any additional meta data to the eopatch
        """


class SentinelHubInputTask(SentinelHubInputBase):
    """ A processing API input task that loads 16bit integer data and converts it to a 32bit float feature.
    """
    def __init__(self, data_source, size=None, resolution=None, bands_feature=None, bands=None, additional_data=None,
                 maxcc=1.0, time_difference=None, cache_folder=None, max_threads=None, config=None,
                 bands_dtype=np.float32, single_scene=False, mosaicking_order='mostRecent'):
        """
        :param data_source: Source of requested satellite data.
        :type data_source: DataSource
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :type resolution: Resolution in meters, passed as a tuple for X and Y axis.
        :type resolution: tuple(int, int)
        :param bands_feature: Target feature into which to save the downloaded images.
        :type bands_feature: tuple(sentinelhub.FeatureType, str)
        :param bands: An array of band names.
        :type bands: list[str]
        :param additional_data: A list of additional data to be downloaded, such as SCL, SNW, dataMask, etc.
        :type additional_data: list[tuple(sentinelhub.FeatureType, str)]
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
        :type bands_dtype: np.dtype
        :param single_scene: If true, the service will compute a single image for the given time interval using
                             mosaicking.
        :type single_scene: bool
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :type mosaicking_order: str
        """
        super().__init__(
            data_source=data_source, size=size, resolution=resolution, cache_folder=cache_folder, config=config,
            max_threads=max_threads
        )

        self.data_source = data_source
        self.maxcc = maxcc
        self.time_difference = dt.timedelta(seconds=1) if time_difference is None else time_difference
        self.single_scene = single_scene
        self.bands_dtype = bands_dtype

        mosaic_order_params = ["mostRecent", "leastRecent", "leastCC"]
        if mosaicking_order not in mosaic_order_params:
            msg = "{} is not a valid mosaickingOrder parameter, it should be one of: {}"
            raise ValueError(msg.format(mosaicking_order, mosaic_order_params))

        self.mosaicking_order = mosaicking_order

        self.bands_feature = next(self._parse_features(bands_feature)()) if bands_feature else None

        if bands is not None:
            self.bands = bands
        elif bands_feature is not None:
            self.bands = data_source.bands()
        else:
            self.bands = []

        if additional_data is None:
            self.additional_data = []
        else:
            self.additional_data = list(self._parse_features(additional_data, new_names=True)())

        self.all_bands = self.bands + [f_name for _, f_name, _ in self.additional_data]

    def generate_evalscript(self):
        """ Generate the evalscript to be passed with the request, based on chosen bands
        """
        evalscript = """
            function setup() {{
                return {{
                    input: [{{
                        bands: {bands},
                        units: "DN"
                    }}],
                    output: {{
                        id:"default",
                        bands: {num_bands},
                        sampleType: SampleType.UINT16
                    }}
                }}
            }}

            function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
                outputMetadata.userData = {{ "norm_factor":  inputMetadata.normalizationFactor }}
            }}

            function evaluatePixel(sample) {{
                return [{samples}]
            }}
        """

        samples = ', '.join(['sample.{}'.format(band) for band in self.all_bands])

        return evalscript.format(bands=json.dumps(self.all_bands), num_bands=len(self.all_bands), samples=samples)

    def _get_timestamp(self, time_interval, bbox):
        """ Get the timestamp array needed as a parameter for downloading the images
        """
        if self.single_scene:
            return [time_interval[0]]

        wfs = WebFeatureService(
            bbox=bbox, time_interval=time_interval, data_source=self.data_source, maxcc=self.maxcc
        )

        dates = wfs.get_dates()

        if len(dates) == 0:
            raise ValueError("No available images for requested time range: {}".format(time_interval))

        dates = sorted(dates)

        return [dates[0]] + [d2 for d1, d2 in zip(dates[:-1], dates[1:]) if d2 - d1 > self.time_difference]

    def _build_payloads(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build payloads for the requests to the service
        """
        if self.single_scene:
            dates = [(iso_to_datetime(time_interval[0]), iso_to_datetime(time_interval[1]))]
        else:
            dates = [(date - self.time_difference, date + self.time_difference) for date in timestamp]

        return [self._request_payload(date1, date2, bbox, size_x, size_y) for date1, date2 in dates]

    def _request_payload(self, date_from, date_to, bbox, size_x, size_y):
        """ Build the payload dictionary for the request
        """
        time_from, time_to = date_from.isoformat() + 'Z', date_to.isoformat() + 'Z'

        responses = [shr.response('default', MimeType.TIFF.get_string()), shr.response('userdata', 'application/json')]

        data = shr.data(time_from=time_from, time_to=time_to, data_type=self.data_source.api_identifier())
        data['dataFilter']['maxCloudCoverage'] = int(self.maxcc * 100)
        data['dataFilter']['mosaickingOrder'] = self.mosaicking_order

        return shr.body(
            request_bounds=shr.bounds(crs=bbox.crs.opengis_string, bbox=list(bbox)),
            request_data=[data],
            request_output=shr.output(size_x=size_x, size_y=size_y, responses=responses),
            evalscript=self.generate_evalscript()
        )

    def _extract_data(self, eopatch, images, shape):
        """ Extract data from the received images and assign them to eopatch features
        """

        images = ((img['default.tif'], img['userdata.json']) for img in images)
        images = [(img, meta.get('norm_factor', 0)) for img, meta in images]

        for f_type, f_name_src, f_name_dst in self.additional_data:
            eopatch[(f_type, f_name_dst)] = self._extract_additional_data(images, f_type, f_name_src, shape)

        if self.bands:
            self._extract_bands(eopatch, images, shape)

    def _extract_additional_data(self, images, f_type, f_name, shape):
        """ extract additional_data from the received images each as a separate feature
        """
        type_dict = {
            FeatureType.MASK: np.bool
        }

        dst_type = type_dict.get(f_type, np.uint16)

        idx = self.all_bands.index(f_name)
        feature_arrays = [np.atleast_3d(img)[..., idx] for img, norm_factor in images]

        return np.asarray(feature_arrays, dtype=dst_type).reshape(*shape, 1)

    def _extract_bands(self, eopatch, images, shape):
        num_bands = len(self.bands)

        bands = [img[..., :num_bands].astype(self.bands_dtype) for img, _ in images]
        norms = [norm_factor for _, norm_factor in images]

        if self.bands_dtype == np.int16:
            norms = np.asarray(norms).reshape(shape[0], 1).astype(np.float32)
            eopatch[(FeatureType.SCALAR, 'NORM_FACTORS')] = norms
        else:
            bands = [np.round(band * norm, 4) for band, norm in zip(bands, norms)]

        eopatch[self.bands_feature] = np.asarray(bands).reshape(*shape, num_bands)

    def _add_meta_info(self, eopatch):
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
        :type resolution: Resolution in meters, passed as a tuple for X and Y axis.
        :type resolution: tuple(int, int)
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        """

        super().__init__(
            data_source=DataSource.DEM, size=size, resolution=resolution, cache_folder=cache_folder, config=config,
            max_threads=max_threads
        )

        feature_parser = self._parse_features(
            dem_feature,
            default_feature_type=FeatureType.DATA_TIMELESS,
            allowed_feature_types=[FeatureType.DATA_TIMELESS]
        )

        self.dem_feature = next(feature_parser())

    def _build_payloads(self, bbox, size_x, size_y, timestamp, time_interval):
        """ Build payloads for the requests to the service
        """
        evalscript = """
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

        responses = [shr.response('default', 'image/tiff'), shr.response('userdata', 'application/json')]
        request_body = shr.body(
            request_bounds=shr.bounds(crs=bbox.crs.opengis_string, bbox=list(bbox)),
            request_data=[{"type": "DEM"}],
            request_output=shr.output(size_x=size_x, size_y=size_y, responses=responses),
            evalscript=evalscript
        )

        return [request_body]

    def _extract_data(self, eopatch, images, shape):
        """ Extract data from the received images and assign them to eopatch features
        """
        tif = images[0]['default.tif']

        eopatch[self.dem_feature] = tif[..., np.newaxis].astype(np.int16)
