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

from eolearn.core import EOPatch, EOTask, FeatureType

LOGGER = logging.getLogger(__name__)

EVALSCRIPT = """
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


class SentinelHubProcessingInput(EOTask):
    ''' A processing API input task that loads 16bit integer data and converts it to a 32bit float feature.
    '''
    def __init__(self, feature_name, size_x, size_y, bbox, time_range, data_source, maxcc=1.0, time_difference=-1,
                 bands=None, data_folder=None, max_threads=5):
        """
        :param feature_name: Target feature into which to save the downloaded images.
        """
        self.feature_name = feature_name
        self.size_x = size_x
        self.size_y = size_y
        self.bbox = bbox
        self.time_range = parse_time_interval(time_range)
        self.data_source = data_source
        self.maxcc = maxcc
        self.time_difference = dt.timedelta(seconds=time_difference)
        self.data_folder = data_folder
        self.bands = data_source.bands() if bands is None else bands
        self.max_threads = max_threads

    @staticmethod
    def request_from_date(request, date):
        ''' Make a deep copy of a request and sets it's (from, to) range according to the provided 'date' argument
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
        samples = ', '.join(['sample.{}'.format(band) for band in self.bands])
        samples = '[{}]'.format(samples)
        script = EVALSCRIPT.format(bands=json.dumps(self.bands), num_bands=len(self.bands), samples=samples)
        return script

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
        args = dict(url=url, headers=headers, data_folder=self.data_folder, hash_save=True, request_type='POST',
                    data_type=MimeType.TAR)
        request_list = [DownloadRequest(post_values=payload, **args) for payload in payloads]

        LOGGER.debug('Starting %d processing requests.', len(request_list))
        client = SentinelHubDownloadClient()
        images = client.download_data(request_list)
        LOGGER.debug('Downloads complete')

        images = ((img['default.tif'], img['userdata.json']) for img in images)
        images = [(img.astype(np.float32), meta.get('norm_factor', 0) if meta else 0) for img, meta in images]

        eopatch = EOPatch() if eopatch is None else eopatch

        shape = len(dates), self.size_y, self.size_x

        if 'dataMask' in self.bands:
            is_data_arrays = [img[..., -1:] for img, norm_factor in images]
            eopatch[(FeatureType.MASK, 'IS_DATA')] = np.asarray(is_data_arrays).reshape(*shape, 1).astype(np.bool)

        img_bands = len(self.bands) - 1 if 'dataMask' in self.bands else len(self.bands)
        img_arrays = [img[..., slice(img_bands)] * norm_factor for img, norm_factor in images]
        eopatch[(FeatureType.DATA, self.feature_name)] = np.asarray(img_arrays).reshape(*shape, img_bands)

        return eopatch
