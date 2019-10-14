''' An input task for the `sentinelhub processing api <https://docs.sentinel-hub.com/api/latest/reference/>`
'''
import io
import json
import tarfile
from copy import deepcopy
import datetime as dt
import numpy as np

from sentinelhub import SentinelHubRequest, SentinelHubOutput, SentinelHubBounds, SentinelHubData, \
    SentinelHubOutputResponse, WebFeatureService, decoding, MimeType, DataSource, SentinelhubClient, \
    parse_time_interval

from eolearn.core import EOPatch, EOTask, FeatureType

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


def tar_to_numpy(data):
    ''' A decoder to convert response bytes into a (image: np.ndarray, nomr_factor: int) tuple
    '''
    tar = tarfile.open(fileobj=io.BytesIO(data))

    img_member = tar.getmember('default.tif')
    file = tar.extractfile(img_member)
    image = decoding.decode_image(file.read(), MimeType.TIFF_d16)
    image = image.astype(np.int16)

    json_member = tar.getmember('userdata.json')
    file = tar.extractfile(json_member)
    meta_obj = decoding.decode_data(file.read(), MimeType.JSON)

    if meta_obj is None:
        return image, 0

    return image, meta_obj['norm_factor']

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


class SentinelHubProcessingInput(EOTask):
    ''' A processing API input task that loads 16bit integer data and converts it to a 32bit float feature.
    '''
    def __init__(self, feature_name, time_range, size_x, size_y, bbox, maxcc=1.0, time_difference=-1,
                 data_source=DataSource.SENTINEL2_L1C, bands=None, cache_dir=None):
        """
        :param feature_name: Target feature into which to save the downloaded images.
        """

        self.time_range = parse_time_interval(time_range)
        self.size_x = size_x
        self.size_y = size_y
        self.feature_name = feature_name
        self.bbox = bbox
        self.maxcc = maxcc
        self.data_source = data_source
        self.time_difference = dt.timedelta(seconds=time_difference)
        self.bands = data_source.bands() if bands is None else bands
        self.cache_dir = cache_dir

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

        if len(dates) <= 1:
            return dates

        dates = sorted(dates)

        cleaned_dates = [dates[0]]
        for curr_date in dates[1:]:
            if curr_date - cleaned_dates[-1] > self.time_difference:
                cleaned_dates.append(curr_date)

        return cleaned_dates

    def execute(self, eopatch=None):
        ''' Make a WFS request to get valid dates, download an image for each valid date and store it in an EOPatch

        :param eopatch: input EOPatch
        :type eopatch: EOPatch
        '''
        dates = self.get_dates()

        responses = [
            SentinelHubOutputResponse('default', 'image/tiff'),
            SentinelHubOutputResponse('userdata', 'application/json')
        ]

        request = SentinelHubRequest(
            bounds=SentinelHubBounds(crs=self.bbox.crs.opengis_string, bbox=list(self.bbox)),
            data=[SentinelHubData(data_type=self.data_source.api_identifier())],
            output=SentinelHubOutput(size_x=self.size_x, size_y=self.size_y, responses=responses),
            evalscript=self.generate_evalscript()
        )

        headers = {"accept": "application/tar", 'content-type': 'application/json'}

        requests = [(request_from_date(request, date), date) for date in dates]

        client = SentinelhubClient(cache_dir=self.cache_dir)
        images = [(date, client.get(req, decoder=tar_to_numpy, headers=headers)) for req, date in requests]

        images = [(date, img) for date, img in images if img is not None]
        dates = [date for date, img in images]

        if eopatch is None:
            eopatch = EOPatch()

        arrays = [(img * norm_factor).astype(np.float32) for date, (img, norm_factor) in images]
        shape = len(dates), self.size_y, self.size_x, len(self.bands)
        eopatch[(FeatureType.DATA, self.feature_name)] = np.asarray(arrays).reshape(*shape)

        return eopatch
