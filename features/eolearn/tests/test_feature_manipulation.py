import unittest
import datetime

from eolearn.features import FilterTimeSeries
from eolearn.core import EOPatch

import numpy as np


class TestFeatureManipulation(unittest.TestCase):

    def test_content_after_timefilter(self):
        timestamps = [datetime.datetime(2017, 1, 1, 10, 4, 7),
                      datetime.datetime(2017, 1, 4, 10, 14, 5),
                      datetime.datetime(2017, 1, 11, 10, 3, 51),
                      datetime.datetime(2017, 1, 14, 10, 13, 46),
                      datetime.datetime(2017, 1, 24, 10, 14, 7),
                      datetime.datetime(2017, 2, 10, 10, 1, 32),
                      datetime.datetime(2017, 2, 20, 10, 6, 35),
                      datetime.datetime(2017, 3, 2, 10, 0, 20),
                      datetime.datetime(2017, 3, 12, 10, 7, 6),
                      datetime.datetime(2017, 3, 15, 10, 12, 14)]
        data = np.random.rand(10, 100, 100, 3)

        new_start = 4
        new_end = -3

        old_interval = (timestamps[0], timestamps[-1])
        new_interval = (timestamps[new_start], timestamps[new_end])

        new_timestamps = [ts for ts in timestamps[new_start:new_end+1]]

        eop = EOPatch(timestamp=timestamps,
                      data={'data': data},
                      meta_info={'time_interval': old_interval})

        filter_task = FilterTimeSeries(start_date=new_interval[0], end_date=new_interval[1])
        filter_task.execute(eop)

        updated_interval = eop.meta_info['time_interval']
        updated_timestamps = eop.timestamp

        self.assertEqual(new_interval, updated_interval)
        self.assertEqual(new_timestamps, updated_timestamps)


if __name__ == '__main__':
    unittest.main()
