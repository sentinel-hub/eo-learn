import unittest
import datetime as dt
from sentinelhub import CRS, BBox, DataSource

from eolearn.io import SentinelHubProcessingDEM
from eolearn.core import FeatureType

# import sys
# import logging
# logging.basicConfig(stream=sys.stdout)
# logging.getLogger("eolearn.io.processing_api").setLevel(logging.DEBUG)
# logging.getLogger("sentinelhub.sentinelhub_client").setLevel(logging.DEBUG)
# logging.getLogger("sentinelhub.sentinelhub_rate_limit").setLevel(logging.DEBUG)

class TestProcessingIO(unittest.TestCase):
    size = (100, 100)
    bbox = BBox(bbox=[268892, 4624365, 268892+size[0]*10, 4624365+size[1]*10], crs=CRS.UTM_33N)
    time_interval = ('2017-12-15', '2017-12-30')
    maxcc = 0.8
    time_difference = dt.timedelta(minutes=60)
    max_threads = 3

    def test_dem(self):
        """ Download S2L1C bands and dataMask
        """
        task = SentinelHubProcessingDEM(
            bands_feature=(FeatureType.MASK_TIMELESS, 'DEM'),
            # additional_data=[(FeatureType.MASK, 'dataMask')],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_source=DataSource.DEM,
            max_threads=self.max_threads,
            cache_folder='test_dem'
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        dem = eopatch[(FeatureType.MASK_TIMELESS, 'DEM')]

        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()
