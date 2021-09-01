"""
Credits:
Copyright (c) 2018-2020 William Ouellette
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import unittest
import logging
import datetime as dt

import numpy as np
from geopandas import GeoDataFrame

from sentinelhub import BBox, CRS
from eolearn.core import EOPatch

logging.basicConfig(level=logging.INFO)


class TestEOPatchMerge(unittest.TestCase):

    def test_time_dependent_merge(self):
        all_timestamps = [dt.datetime(2020, month, 1) for month in range(1, 7)]
        eop1 = EOPatch(
            data={
                'bands': np.ones((3, 4, 5, 2))
            },
            timestamp=[all_timestamps[0], all_timestamps[5], all_timestamps[4]]
        )
        eop2 = EOPatch(
            data={
                'bands': np.ones((5, 4, 5, 2))
            },
            timestamp=[all_timestamps[3], all_timestamps[1], all_timestamps[2], all_timestamps[4], all_timestamps[3]]
        )

        eop = eop1.merge(eop2)
        expected_eop = EOPatch(
            data={
                'bands': np.ones((6, 4, 5, 2))
            },
            timestamp=all_timestamps
        )
        self.assertEqual(eop, expected_eop)

        eop = eop1.merge(eop2, time_dependent_op='concatenate')
        expected_eop = EOPatch(
            data={
                'bands': np.ones((8, 4, 5, 2))
            },
            timestamp=all_timestamps[:4] + [all_timestamps[3], all_timestamps[4]] + all_timestamps[4:]
        )
        self.assertEqual(eop, expected_eop)

        eop1.data['bands'][1, ...] = 6
        eop1.data['bands'][-1, ...] = 3
        eop2.data['bands'][0, ...] = 5
        eop2.data['bands'][1, ...] = 4

        with self.assertRaises(ValueError):
            eop1.merge(eop2)

        eop = eop1.merge(eop2, time_dependent_op='mean')
        expected_eop = EOPatch(
            data={
                'bands': np.ones((6, 4, 5, 2))
            },
            timestamp=all_timestamps
        )
        expected_eop.data['bands'][1, ...] = 4
        expected_eop.data['bands'][3, ...] = 3
        expected_eop.data['bands'][4, ...] = 2
        expected_eop.data['bands'][5, ...] = 6
        self.assertEqual(eop, expected_eop)

    def test_time_dependent_merge_with_missing_features(self):
        timestamps = [dt.datetime(2020, month, 1) for month in range(1, 7)]
        eop1 = EOPatch(
            data={'bands': np.ones((6, 4, 5, 2))},
            label={'label': np.ones((6, 7), dtype=np.uint8)},
            timestamp=timestamps
        )
        eop2 = EOPatch(timestamp=timestamps[:4])

        eop = eop1.merge(eop2)
        assert eop == eop1

        eop = eop2.merge(eop1, eop1, eop2, time_dependent_op='min')
        assert eop == eop1

        eop = eop1.merge()
        assert eop == eop1

    def test_failed_time_dependent_merge(self):
        eop1 = EOPatch(
            data={'bands': np.ones((6, 4, 5, 2))}
        )
        with self.assertRaises(ValueError):
            eop1.merge()

        eop2 = EOPatch(
            data={'bands': np.ones((1, 4, 5, 2))},
            timestamp=[dt.datetime(2020, 1, 1)]
        )
        with self.assertRaises(ValueError):
            eop2.merge(eop1)

    def test_timeless_merge(self):
        eop1 = EOPatch(
            mask_timeless={
                'mask': np.ones((3, 4, 5), dtype=np.int16),
                'mask1': np.ones((5, 4, 3), dtype=np.int16)
            }
        )
        eop2 = EOPatch(
            mask_timeless={
                'mask': 4 * np.ones((3, 4, 5), dtype=np.int16),
                'mask2': np.ones((4, 5, 3), dtype=np.int16)
            }
        )

        with self.assertRaises(ValueError):
            eop1.merge(eop2)

        eop = eop1.merge(eop2, timeless_op='concatenate')
        expected_eop = EOPatch(
            mask_timeless={
                'mask': np.ones((3, 4, 10), dtype=np.int16),
                'mask1': eop1.mask_timeless['mask1'],
                'mask2': eop2.mask_timeless['mask2']
            }
        )
        expected_eop.mask_timeless['mask'][..., 5:] = 4
        self.assertEqual(eop, expected_eop)

        eop = eop1.merge(eop2, eop2, timeless_op='min')
        expected_eop = EOPatch(
            mask_timeless={
                'mask': eop1.mask_timeless['mask'],
                'mask1': eop1.mask_timeless['mask1'],
                'mask2': eop2.mask_timeless['mask2']
            }
        )
        self.assertEqual(eop, expected_eop)

    def test_vector_merge(self):
        bbox = BBox((1, 2, 3, 4), CRS.WGS84)
        df = GeoDataFrame({
            'values': [1, 2],
            'TIMESTAMP': [dt.datetime(2017, 1, 1, 10, 4, 7), dt.datetime(2017, 1, 4, 10, 14, 5)],
            'geometry': [bbox.geometry, bbox.geometry]
        }, crs=bbox.crs.pyproj_crs())

        eop1 = EOPatch(
            vector_timeless={
                'vectors': df
            }
        )

        for eop in [eop1.merge(eop1), eop1 + eop1]:
            self.assertEqual(eop, eop1)

        eop2 = eop1.__deepcopy__()
        eop2.vector_timeless['vectors'].crs = CRS.POP_WEB.pyproj_crs()
        with self.assertRaises(ValueError):
            eop1.merge(eop2)

    def test_meta_info_merge(self):
        eop1 = EOPatch(
            meta_info={
                'a': 1,
                'b': 2
            }
        )
        eop2 = EOPatch(
            meta_info={
                'a': 1,
                'c': 5
            }
        )

        eop = eop1.merge(eop2)
        expected_eop = EOPatch(
            meta_info={
                'a': 1,
                'b': 2,
                'c': 5
            }
        )
        self.assertEqual(eop, expected_eop)

        eop2.meta_info['a'] = 3
        with self.assertWarns(UserWarning):
            eop = eop1.merge(eop2)
        self.assertEqual(eop, expected_eop)

    def test_bbox_merge(self):
        eop1 = EOPatch(bbox=BBox((1, 2, 3, 4), CRS.WGS84))
        eop2 = EOPatch(bbox=BBox((1, 2, 3, 4), CRS.POP_WEB))

        eop = eop1.merge(eop1)
        self.assertEqual(eop, eop1)

        with self.assertRaises(ValueError):
            eop1.merge(eop2)


if __name__ == '__main__':
    unittest.main()
