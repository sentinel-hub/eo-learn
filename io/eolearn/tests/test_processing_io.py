""" Testing SentinelHubProcessingInput
"""

import unittest
import datetime as dt
from sentinelhub import CRS, BBox, DataSource

from eolearn.io import SentinelHubProcessingInput
from eolearn.core import FeatureType

# import sys
# import logging
# logging.basicConfig(stream=sys.stdout)
# logging.getLogger("eolearn.io.processing_api").setLevel(logging.DEBUG)
# logging.getLogger("sentinelhub.sentinelhub_client").setLevel(logging.DEBUG)
# logging.getLogger("sentinelhub.sentinelhub_rate_limit").setLevel(logging.DEBUG)

class TestProcessingIO(unittest.TestCase):
    """ Test cases for SentinelHubProcessingInput
    """
    size = (99, 101)
    bbox = BBox(bbox=[268892, 4624365, 268892+size[0]*10, 4624365+size[1]*10], crs=CRS.UTM_33N)
    time_interval = ('2017-12-15', '2017-12-30')
    maxcc = 0.8
    time_difference = dt.timedelta(minutes=60)
    max_threads = 3

    def test_S2L1C(self):
        """ Download S2L1C bands and dataMask
        """
        task = SentinelHubProcessingInput(
            bands_feature=(FeatureType.DATA, 'BANDS'),
            additional_data=[(FeatureType.MASK, 'dataMask')],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_source=DataSource.SENTINEL2_L1C,
            max_threads=self.max_threads,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, 'BANDS')]
        is_data = eopatch[(FeatureType.MASK, 'dataMask')]

        width, height = self.size
        self.assertTrue(bands.shape == (4, height, width, 13))
        self.assertTrue(is_data.shape == (4, height, width, 1))

    def test_specific_bands(self):
        """ Download S2L1C bands and dataMask
        """
        task = SentinelHubProcessingInput(
            bands_feature=(FeatureType.DATA, 'BANDS'),
            bands=["B01", "B02", "B03"],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_source=DataSource.SENTINEL2_L1C,
            max_threads=self.max_threads
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, 'BANDS')]

        width, height = self.size
        self.assertTrue(bands.shape == (4, height, width, 3))

    def test_S2L2A(self):
        """ Download just SCL, without other bands
        """
        task = SentinelHubProcessingInput(
            bands_feature=(FeatureType.DATA, 'BANDS'),
            additional_data=[(FeatureType.MASK, 'dataMask')],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_source=DataSource.SENTINEL2_L2A,
            max_threads=self.max_threads
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, 'BANDS')]
        is_data = eopatch[(FeatureType.MASK, 'dataMask')]

        width, height = self.size
        self.assertTrue(bands.shape == (4, height, width, 12))
        self.assertTrue(is_data.shape == (4, height, width, 1))

    def test_scl_only(self):
        """ Download just SCL, without any other bands
        """
        task = SentinelHubProcessingInput(
            bands_feature=None,
            additional_data=[(FeatureType.DATA, 'SCL')],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_source=DataSource.SENTINEL2_L2A,
            max_threads=self.max_threads
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        scl = eopatch[(FeatureType.DATA, 'SCL')]

        width, height = self.size
        self.assertTrue(scl.shape == (4, height, width, 1))


if __name__ == "__main__":
    unittest.main()
