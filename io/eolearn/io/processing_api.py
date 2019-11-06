''' An input task for the `sentinelhub processing api <https://docs.sentinel-hub.com/api/latest/reference/>`
'''
import json
import logging
from copy import deepcopy
import datetime as dt
import numpy as np

from sentinelhub import WebFeatureService, MimeType, SentinelHubDownloadClient, parse_time_interval, DownloadRequest,\
     SHConfig
import sentinelhub.sentinelhub_request as shr

from eolearn.core import EOPatch, EOTask

LOGGER = logging.getLogger(__name__)


class SentinelHubProcessingInput(EOTask):
    ''' A processing API input task that loads 16bit integer data and converts it to a 32bit float feature.
    '''
    def __init__(self, size_x, size_y, bbox, time_range, data_source, bands_feature=None, bands=None,
                 additional_data=None, maxcc=1.0, time_difference=-1, cache_folder=None, max_threads=5):
        """
        :param size_x: Number of pixels in x dimension.
        :type size_x: int
        :param size_y: Number of pixels in y dimension.
        :type size_y: int
        :param bbox: Bounding box.
        :type bbox: sentinelhhub.BBox
        :param time_range: A range tuple of (date_from, date_to), defining a time range from which to acquire the data.
        :type time_range: tuple(str, str)
        :param bands_feature: Target feature into which to save the downloaded images.
        :type bands_feature: tuple(sentinelhub.FeatureType, str)
        :param bands: An array of band names.
        :type bands: list[str]
        :param additional_data: A list of additional data to be downloaded, such as SCL, SNW, dataMask, etc.
        :type additional_data: list[tuple(sentinelhub.FeatureType, str)]
        :param maxcc: Maximum cloud coverage.
        :type maxcc: float
        :param time_difference: Minimum allowed time difference in seconds, used when filtering dates.
        :type time_difference: int
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        """
        self.size_x = size_x
        self.size_y = size_y
        self.bbox = bbox
        self.time_range = parse_time_interval(time_range)
        self.data_source = data_source
        self.maxcc = maxcc
        self.time_difference = dt.timedelta(seconds=time_difference)
        self.cache_folder = cache_folder
        self.max_threads = max_threads

        self.bands_feature = bands_feature
        self.bands = bands or data_source.bands() if bands_feature else []
        self.additional_data = additional_data or []

        self.all_bands = self.bands + [f_name for _, f_name in self.additional_data]

    @staticmethod
    def request_from_date(request, date):
        ''' Make a deep copy of a request and sets it's (from, to) range according to the provided 'date' argument

        :param request: Path to cache_folder. If set to None (default) requests will not be cached.
        :type request: str
        '''
        date_from, date_to = date, date + dt.timedelta(seconds=1)
        time_from, time_to = date_from.isoformat() + 'Z', date_to.isoformat() + 'Z'

        request = deepcopy(request)
        for data in request['input']['data']:
            time_range = data['dataFilter']['timeRange']
            time_range['from'] = time_from
            time_range['to'] = time_to

        return request

    def generate_evalscript(self):
        ''' Generate the evalscript to be passed with the request, based on chosen bands
        '''
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
                return {samples}
            }}
        """

        samples = ', '.join(['sample.{}'.format(band) for band in self.all_bands])
        samples = '[{}]'.format(samples)

        return evalscript.format(bands=json.dumps(self.all_bands), num_bands=len(self.all_bands), samples=samples)

    def get_dates(self):
        ''' Make a WebFeatureService request to get dates and clean them according to self.time_difference
        '''
        wfs = WebFeatureService(
            bbox=self.bbox, time_interval=self.time_range, data_source=self.data_source, maxcc=self.maxcc
        )

        dates = wfs.get_dates()

        if len(dates) == 0:
            raise ValueError("No available images for requested time range: {}".format(self.time_range))

        dates = sorted(dates)
        dates = [dates[0]] + [d2 for d1, d2 in zip(dates[:-1], dates[1:]) if d2 - d1 > self.time_difference]
        return dates

    def execute(self, eopatch=None):
        ''' Make a WFS request to get valid dates, download an image for each valid date and store it in an EOPatch

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        '''
        dates = self.get_dates()

        responses = [
            shr.response('default', 'image/tiff'),
            shr.response('userdata', 'application/json')
        ]

        request = shr.body(
            request_bounds=shr.bounds(crs=self.bbox.crs.opengis_string, bbox=list(self.bbox)),
            request_data=[shr.data(data_type=self.data_source.api_identifier())],
            request_output=shr.output(size_x=self.size_x, size_y=self.size_y, responses=responses),
            evalscript=self.generate_evalscript()
        )

        url = SHConfig().get_sh_processing_api_url()
        headers = {"accept": "application/tar", 'content-type': 'application/json'}
        payloads = [self.request_from_date(request, date) for date in dates]
        args = dict(url=url, headers=headers, data_folder=self.cache_folder, hash_save=bool(self.cache_folder),
                    request_type='POST', data_type=MimeType.TAR)
        request_list = [DownloadRequest(post_values=payload, **args) for payload in payloads]

        LOGGER.debug('Downloading %d requests of type %s', len(request_list), str(self.data_source))
        LOGGER.debug('Downloading bands: [%s]', ', '.join(self.all_bands))
        client = SentinelHubDownloadClient()
        images = client.download_data(request_list)
        LOGGER.debug('Downloads complete')

        images = ((img['default.tif'], img['userdata.json']) for img in images)
        images = [(img, meta.get('norm_factor', 0) if meta else 0) for img, meta in images]

        eopatch = EOPatch() if eopatch is None else eopatch

        shape = len(dates), self.size_y, self.size_x

        # exctract additional_data from the received images each as a separate feature
        for f_type, f_name in self.additional_data:
            idx = self.all_bands.index(f_name)
            feature_arrays = [np.atleast_3d(img)[..., idx] for img, norm_factor in images]
            eopatch[(f_type, f_name)] = np.asarray(feature_arrays).reshape(*shape, 1)

        # exctract bands from the received and save them as self.bands_feature
        if self.bands:
            img_bands = len(self.bands)
            img_arrays = [img[..., slice(img_bands)].astype(np.float32) * norm_factor for img, norm_factor in images]
            eopatch[self.bands_feature] = np.asarray(img_arrays).reshape(*shape, img_bands)

        return eopatch
