from sentinelhub import MimeType, CRS, SentinelHubRequest, SentinelHubOutput, SentinelHubBounds,\
    SentinelHubData, SentinelHubOutputResponse, SentinelHubWrapper, SHConfig, SentinelhubSession, WebFeatureService, \
    decoding, MimeType, DataSource, SentinelhubClient, BBox

from eolearn.core import EOPatch, EOTask, FeatureType
from sentinelhub import parse_time_interval
import tarfile
import json
import tifffile as tiff
import numpy as np
import io
from copy import deepcopy
import datetime as dt


EVALSCRIPT = """
    function setup() {
        return {
            input: ["B02", "B03", "B04"],
            output: { id:"default", bands: 3}
        }
    }

    function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
        outputMetadata.userData = { "metadata":  JSON.stringify(inputMetadata) }
    }

    function evaluatePixel(sample) {
        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
"""

def tar_to_numpy(data):
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data))

        img_member = tar.getmember('default.tif')
        file = tar.extractfile(img_member)
        image = decoding.decode_image(file.read(), MimeType.TIFF_d16)
        image = image.astype(np.int16)

        json_member = tar.getmember('userdata.json')
        file = tar.extractfile(json_member)
        meta_obj = decoding.decode_data(file.read(), MimeType.JSON)

        # decode_data only accepts encoded bytes, not str
        meta_dict = json.loads(meta_obj['metadata'])
        norm_factor = meta_dict['normalizationFactor']

        return image, norm_factor
    except:
        return None

def copy_format_request(request, date):
    date_from, date_to = date, date + dt.timedelta(seconds=1)
    time_from, time_to = date_from.isoformat() + 'Z', date_to.isoformat() + 'Z'

    request = deepcopy(request)
    for data in request.body['input']['data']:
        time_range = data['dataFilter']['timeRange']
        time_range['from'] = time_from
        time_range['to'] = time_to
    return request

class Sentinelhub16bitInput(EOTask):
    def __init__(self, feature_name, time_range, size_x, size_y, bbox):
        self.time_range = parse_time_interval(time_range)
        self.size_x = size_x
        self.size_y = size_y
        self.feature_name = feature_name
        self.bbox = bbox

    def execute(self, eopatch=None):
        # ------------------- get dates -------------------

        wfs = WebFeatureService(
            bbox=self.bbox, time_interval=self.time_range, data_source=DataSource.SENTINEL2_L1C, maxcc=1.0
        )

        dates = wfs.get_dates()

        # ------------------- build request -------------------

        responses = [
            SentinelHubOutputResponse('default', 'image/tiff'),
            SentinelHubOutputResponse('userdata', 'application/json')
        ]

        body = SentinelHubRequest(
            bounds=SentinelHubBounds(crs=self.bbox.crs.opengis_string, bbox=list(self.bbox)),
            data=[SentinelHubData(data_type='S2L1C')],
            output=SentinelHubOutput(size_x=self.size_x, size_y=self.size_y, responses=responses),
            evalscript=EVALSCRIPT
        )

        request = SentinelHubWrapper(
            body=body, headers={"accept": "application/tar", 'content-type': 'application/json'}
        )

        # ------------------- map dates to the built request -------------------

        requests = [(copy_format_request(request, date), date) for date in dates]

        # ------------------- map dates to the built request -------------------

        client = SentinelhubClient()
        images = [(date, client.get(req, decoder=tar_to_numpy)) for req, date in requests]
        images = [(date, img) for date, img in images if img is not None]
        images = sorted(images, key=lambda x: x[0])
        dates = [date for date, img in images]

        norm_factor = [img[1][1] for img in images]
        arrays = [img[1][0] for img in images]

        if eopatch is None:
            eopatch = EOPatch()

        eopatch.timestamp = dates
        eopatch[(FeatureType.DATA, self.feature_name)] = np.asarray(arrays)
        eopatch[(FeatureType.SCALAR, 'norm_factor')] = np.asarray(norm_factor)[:,np.newaxis]

        return eopatch

