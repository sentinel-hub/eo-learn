import tarfile
import numpy as np
import io
from copy import deepcopy
import datetime as dt

from sentinelhub import MimeType, CRS, SentinelHubRequest, SentinelHubOutput, SentinelHubBounds, SentinelHubData,\
    SentinelHubOutputResponse, SentinelHubWrapper, WebFeatureService, decoding, MimeType, DataSource, \
    SentinelhubClient, parse_time_interval

from eolearn.core import EOPatch, EOTask, FeatureType


EVALSCRIPT_L1C = """
    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
                units: "DN"
            }],
            output: {
                id:"default",
                bands: 13,
                sampleType: SampleType.UINT16
            }
        }
    }

    function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
        outputMetadata.userData = { "norm_factor":  inputMetadata.normalizationFactor }
    }

    function evaluatePixel(sample) {
        return [ sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
                 sample.B07, sample.B08, sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12]
    }
"""

EVALSCRIPT_L2A = """
    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
                units: "DN"
            }],
            output: {
                id:"default",
                bands: 12,
                sampleType: SampleType.UINT16
            }
        }
    }

    function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
        outputMetadata.userData = { "norm_factor":  inputMetadata.normalizationFactor }
    }

    function evaluatePixel(sample) {
        return [ sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
                 sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12]
    }
"""


def tar_to_numpy(data):
    tar = tarfile.open(fileobj=io.BytesIO(data))

    img_member = tar.getmember('default.tif')
    file = tar.extractfile(img_member)
    image = decoding.decode_image(file.read(), MimeType.TIFF_d16)
    image = image.astype(np.int16)

    try:
        json_member = tar.getmember('userdata.json')
        file = tar.extractfile(json_member)
        meta_obj = decoding.decode_data(file.read(), MimeType.JSON)
        norm_factor = meta_obj['norm_factor']
    except TypeError:
        norm_factor = 0

    return image, norm_factor

def copy_format_request(request, date):
    date_from, date_to = date, date + dt.timedelta(seconds=1)
    time_from, time_to = date_from.isoformat() + 'Z', date_to.isoformat() + 'Z'

    request = deepcopy(request)
    for data in request.body['input']['data']:
        time_range = data['dataFilter']['timeRange']
        time_range['from'] = time_from
        time_range['to'] = time_to
    return request


class SentinelHubProcessingInput(EOTask):
    def __init__(self, feature_name, time_range, size_x, size_y, bbox, maxcc=1.0, time_difference=-1, store='16bit',
                 data_source=DataSource.SENTINEL2_L1C):

        self.time_range = parse_time_interval(time_range)
        self.size_x = size_x
        self.size_y = size_y
        self.feature_name = feature_name
        self.bbox = bbox
        self.maxcc = maxcc
        self.store = store
        self.data_source = data_source
        self.time_difference = dt.timedelta(seconds=time_difference)

    def execute(self, eopatch=None):
        # ------------------- get dates -------------------

        wfs = WebFeatureService(
            bbox=self.bbox, time_interval=self.time_range, data_source=self.data_source, maxcc=self.maxcc
        )

        dates = wfs.get_dates()

        if len(dates) <= 1:
            return dates

        sorted_dates = sorted(dates)

        separate_dates = [sorted_dates[0]]
        for curr_date in sorted_dates[1:]:
            if curr_date - separate_dates[-1] > self.time_difference:
                separate_dates.append(curr_date)

        dates = separate_dates

        # iter_pairs = ((d1, d2) for d1, d2 in zip(sorted_dates[:-1], sorted_dates[1:]))

        # ------------------- build request -------------------

        responses = [
            SentinelHubOutputResponse('default', 'image/tiff'),
            SentinelHubOutputResponse('userdata', 'application/json')
        ]

        # TODO: temporary solution, DataSource itself should support such mapping
        data_type = {
            DataSource.SENTINEL2_L1C: 'S2L1C',
            DataSource.SENTINEL2_L2A: 'S2L2A'
        }[self.data_source]

        evalscript = {
            DataSource.SENTINEL2_L1C: EVALSCRIPT_L1C,
            DataSource.SENTINEL2_L2A: EVALSCRIPT_L2A
        }[self.data_source]

        body = SentinelHubRequest(
            bounds=SentinelHubBounds(crs=self.bbox.crs.opengis_string, bbox=list(self.bbox)),
            data=[SentinelHubData(data_type=data_type)],
            output=SentinelHubOutput(size_x=self.size_x, size_y=self.size_y, responses=responses),
            evalscript=evalscript
        )

        request = SentinelHubWrapper(
            body=body, headers={"accept": "application/tar", 'content-type': 'application/json'}
        )

        # ------------------- map dates to the built request -------------------

        requests = [(copy_format_request(request, date), date) for date in dates]

        # ------------------- map dates to the built request -------------------

        client = SentinelhubClient(cache_dir='cache_dir')
        images = [(date, client.get(req, decoder=tar_to_numpy)) for req, date in requests]

        images = [(date, img) for date, img in images if img is not None]
        dates = [date for date, img in images]

        if eopatch is None:
            eopatch = EOPatch()

        if self.store == '16bit':
            norm_factor = [img[1][1] for img in images]
            arrays = [img[1][0] for img in images]

            eopatch.timestamp = dates
            eopatch[(FeatureType.DATA, self.feature_name)] = np.asarray(arrays)
            eopatch[(FeatureType.SCALAR, 'norm_factor')] = np.asarray(norm_factor)[:, np.newaxis]
        elif self.store == '32bit':
            arrays = [(img * norm_factor).astype(np.float32) for date, (img, norm_factor) in images]
            eopatch[(FeatureType.DATA, self.feature_name)] = np.asarray(arrays)

        return eopatch
