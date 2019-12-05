""" An input task for the `sentinelhub processing api <https://docs.sentinel-hub.com/api/latest/reference/>`
"""
import json
import logging
import datetime as dt
import numpy as np

from sentinelhub import WebFeatureService, MimeType, SentinelHubDownloadClient, DownloadRequest, SHConfig,\
    bbox_to_dimensions
import sentinelhub.sentinelhub_request as shr

from eolearn.core import EOPatch, EOTask, FeatureType

LOGGER = logging.getLogger(__name__)


class SentinelHubInputTask(EOTask):
    """ A processing API input task that loads 16bit integer data and converts it to a 32bit float feature.
    """
    def __init__(self, data_source, size=None, resolution=None, bands_feature=None, bands=None, additional_data=None,
                 maxcc=1.0, time_difference=None, cache_folder=None, max_threads=None, config=None,
                 bands_dtype=np.float32):
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
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        """
        self.size = size
        self.resolution = resolution
        self.data_source = data_source
        self.maxcc = maxcc
        self.time_difference = dt.timedelta(seconds=1) if time_difference is None else time_difference
        self.cache_folder = cache_folder
        self.max_threads = max_threads
        self.config = config or SHConfig()
        self.bands_dtype = bands_dtype

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

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """ Make a WFS request to get valid dates, download an image for each valid date and store it in an EOPatch

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

        if eopatch is not None and (bbox or time_interval):
            raise ValueError('Either an eopatch must be provided or bbox and time interval, not both.')

        if eopatch is None:
            eopatch = EOPatch()
            eopatch.bbox = bbox

        if self.size is not None:
            size_x, size_y = self.size
        elif self.resolution is not None:
            size_x, size_y = bbox_to_dimensions(eopatch.bbox, self.resolution)

        eopatch.timestamp = self.get_dates(eopatch.bbox, time_interval)

        payloads = (self._request_payload(date, eopatch.bbox, size_x, size_y) for date in eopatch.timestamp)

        request_args = dict(
            url=self.config.get_sh_processing_api_url(),
            headers={"accept": "application/tar", 'content-type': 'application/json'},
            data_folder=self.cache_folder,
            hash_save=bool(self.cache_folder),
            request_type='POST',
            data_type=MimeType.TAR
        )
        requests = [DownloadRequest(post_values=payload, **request_args) for payload in payloads]

        LOGGER.debug('Downloading %d requests of type %s', len(requests), str(self.data_source))
        LOGGER.debug('Downloading bands: [%s]', ', '.join(self.all_bands))
        client = SentinelHubDownloadClient(config=self.config)
        images = client.download(requests, max_threads=self.max_threads)
        LOGGER.debug('Downloads complete')

        images = ((img['default.tif'], img['userdata.json']) for img in images)
        images = [(img, meta.get('norm_factor', 0)) for img, meta in images]

        shape = len(eopatch.timestamp), size_y, size_x

        for f_type, f_name_src, f_name_dst in self.additional_data:
            eopatch[(f_type, f_name_dst)] = self._extract_additional_data(images, f_type, f_name_src, shape)

        if self.bands:
            self._extract_bands(eopatch, images, shape)

        eopatch.meta_info['service_type'] = 'processing'
        eopatch.meta_info['size_x'] = size_x
        eopatch.meta_info['size_y'] = size_y
        eopatch.meta_info['maxcc'] = self.maxcc
        eopatch.meta_info['time_interval'] = time_interval
        eopatch.meta_info['time_difference'] = self.time_difference

        return eopatch

    def _request_payload(self, date, bbox, size_x, size_y):
        """ Build the payload dictionary for the request
        """
        date_from, date_to = date - self.time_difference, date + self.time_difference
        time_from, time_to = date_from.isoformat() + 'Z', date_to.isoformat() + 'Z'

        responses = [shr.response('default', MimeType.TIFF.get_string()), shr.response('userdata', 'application/json')]

        data = shr.data(time_from=time_from, time_to=time_to, data_type=self.data_source.api_identifier())
        data['dataFilter']['maxCloudCoverage'] = int(self.maxcc * 100)

        return shr.body(
            request_bounds=shr.bounds(crs=bbox.crs.opengis_string, bbox=list(bbox)),
            request_data=[data],
            request_output=shr.output(size_x=size_x, size_y=size_y, responses=responses),
            evalscript=self.generate_evalscript()
        )

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

    def get_dates(self, bbox, time_interval):
        """ Make a WebFeatureService request to get dates and clean them according to self.time_difference
        """
        wfs = WebFeatureService(
            bbox=bbox, time_interval=time_interval, data_source=self.data_source, maxcc=self.maxcc
        )

        dates = wfs.get_dates()

        if len(dates) == 0:
            raise ValueError("No available images for requested time range: {}".format(time_interval))

        dates = sorted(dates)
        dates = [dates[0]] + [d2 for d1, d2 in zip(dates[:-1], dates[1:]) if d2 - d1 > self.time_difference]
        return dates

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
