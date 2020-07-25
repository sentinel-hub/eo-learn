"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import numpy as np

from datetime import date, timedelta

from eolearn.core import EOPatch, FeatureType
from eolearn.features import AddMaxMinNDVISlopeIndicesTask, AddMaxMinTemporalIndicesTask, \
    AddSpatioTemporalFeaturesTask, TemporalRollingWindowTask, \
    SurfaceExtractionTask, MaxMeanLenTask


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


class TestTemporalFeaturesTasks(unittest.TestCase):

    def test_temporal_indices(self):
        """ Test case for computation of argmax/argmin of NDVI and another band

        Cases with and without data masking are tested
        """
        # EOPatch
        eopatch = EOPatch()
        t, h, w, c = 5, 3, 3, 2
        # NDVI
        ndvi_shape = (t, h, w, 1)
        # VAlid data mask
        valid_data = np.ones(ndvi_shape, np.bool)
        valid_data[0] = 0
        valid_data[-1] = 0
        # Fill in eopatch
        eopatch.add_feature(FeatureType.DATA, 'NDVI', np.arange(np.prod(ndvi_shape)).reshape(ndvi_shape))
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', np.ones(ndvi_shape, dtype=np.int16))
        eopatch.add_feature(FeatureType.MASK, 'VALID_DATA', valid_data)
        # Task
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
        # Run task
        new_eopatch = add_ndvi(eopatch)
        # Asserts
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], np.zeros((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t-1)*np.ones((h, w, 1))))
        del add_ndvi, new_eopatch
        # Repeat with valid dat amask
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=True)
        new_eopatch = add_ndvi(eopatch)
        # Asserts
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t-2)*np.ones((h, w, 1))))
        del add_ndvi, new_eopatch, valid_data
        # BANDS
        bands_shape = (t, h, w, c)
        eopatch.add_feature(FeatureType.DATA, 'BANDS', np.arange(np.prod(bands_shape)).reshape(bands_shape))
        add_bands = AddMaxMinTemporalIndicesTask(data_feature='BANDS',
                                                 data_index=1,
                                                 amax_data_feature='ARGMAX_B1',
                                                 amin_data_feature='ARGMIN_B1',
                                                 mask_data=False)
        new_eopatch = add_bands(eopatch)
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_B1'], np.zeros((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_B1'], (t-1)*np.ones((h, w, 1))))

    def test_ndvi_slope_indices(self):
        """ Test case for computation of argmax/argmin of NDVI slope

            The NDVI is a sinusoid over 0-pi over the temporal dimension

            Cases with and without data masking are tested
        """
        # Slope needs timestamps
        timestamp = perdelta(date(2018, 3, 1), date(2018, 3, 11), timedelta(days=1))
        # EOPatch
        eopatch = EOPatch(timestamp=list(timestamp))
        t, h, w, = 10, 3, 3
        # NDVI is a sinusoid where max slope is at index 1 and min slope at index 8
        ndvi_shape = (t, h, w, 1)
        xx = np.zeros(ndvi_shape, np.float32)
        x = np.linspace(0, np.pi, t)
        xx[:, :, :, :] = x[:, None, None, None]
        # Valid data mask
        valid_data = np.ones(ndvi_shape, np.uint8)
        valid_data[1] = 0
        valid_data[-1] = 0
        valid_data[4] = 0
        # Fill EOPatch
        eopatch.add_feature(FeatureType.DATA, 'NDVI', np.sin(xx))
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', np.ones(ndvi_shape, np.bool))
        eopatch.add_feature(FeatureType.MASK, 'VALID_DATA', valid_data)
        # Tasks
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
        add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=False)
        # Run
        new_eopatch = add_ndvi_slope(add_ndvi(eopatch))
        # Assert
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], (t-1)*np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t//2-1)*np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI_SLOPE'], (t-2)*np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI_SLOPE'], np.ones((h, w, 1))))
        del add_ndvi_slope, add_ndvi, new_eopatch
        # Run on valid data only now
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=True)
        add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=True)
        # Run
        new_eopatch = add_ndvi_slope(add_ndvi(eopatch))
        # Assert
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], 0 * np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t // 2) * np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI_SLOPE'], (t - 3) * np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI_SLOPE'], 2 * np.ones((h, w, 1))))

    def test_stf_task(self):
        """ Test case for computation of spatio-temporal features

            The NDVI is a sinusoid over 0-pi over the temporal dimension, while bands is an array with values equal to
            the temporal index
        """
        # Timestamps
        timestamp = perdelta(date(2018, 3, 1), date(2018, 3, 11), timedelta(days=1))
        # EOPatch
        eopatch = EOPatch(timestamp=list(timestamp))
        # Shape of arrays
        t, h, w, c = 10, 3, 3, 2
        # NDVI is a sinusoid where max slope is at index 1 and min slope at index 8
        ndvi_shape = (t, h, w, 1)
        bands_shape = (t, h, w, c)
        xx = np.zeros(ndvi_shape, np.float32)
        x = np.linspace(0, np.pi, t)
        xx[:, :, :, :] = x[:, None, None, None]
        # Bands are arrays with values equal to the temporal index
        bands = np.ones(bands_shape)*np.arange(t)[:, None, None, None]
        # Add features to eopatch
        eopatch.add_feature(FeatureType.DATA, 'NDVI', np.sin(xx))
        eopatch.add_feature(FeatureType.DATA, 'BANDS', bands)
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', np.ones(ndvi_shape, np.bool))
        # Tasks
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
        add_bands = AddMaxMinTemporalIndicesTask(data_feature='BANDS',
                                                 data_index=1,
                                                 amax_data_feature='ARGMAX_B1',
                                                 amin_data_feature='ARGMIN_B1',
                                                 mask_data=False)
        add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=False)
        add_stf = AddSpatioTemporalFeaturesTask(argmax_red='ARGMAX_B1', data_feature='BANDS', indices=[0, 1])
        # Run tasks
        new_eopatch = add_stf(add_ndvi_slope(add_bands(add_ndvi(eopatch))))
        # Asserts
        self.assertTrue(new_eopatch.data_timeless['STF'].shape == (h, w, c*5))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 0:c], 4*np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, c:2*c], 9*np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 2*c:3*c], np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 3*c:4*c], 8*np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 4*c:5*c], 9*np.ones((h, w, c))))

    def test_surface_extraction_task(self):
        """Test case for computation of surface extraction task

        Fictional ndvi is so that corner cases are considered. Both positive and negative derivative cases are tested.
        The case with and without valid pixel mask is tested.
        """
        eopatch = EOPatch(timestamp=list(perdelta(date(2020, 3, 1), date(2020, 3, 23), timedelta(days=6))))
        # Fill EOPatch
        ndvi = np.array([
            [
                [[1], [4], [5], [4]],
                [[2], [3], [16], [11]],
            ],
            [
                [[200], [3], [22], [5]],
                [[6], [19], [8], [9]],
            ],
            [
                [[-1], [4], [17], [4]],
                [[20], [-3], [6], [11]],
            ],
            [
                [[1], [4], [5], [4]],
                [[2], [3], [16], [12]],
            ],
        ])
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi)

        mask_feature = (FeatureType.MASK, 'derivative_mask')
        in_feature = (FeatureType.DATA, 'NDVI')

        task = SurfaceExtractionTask(in_feature, 'derivative', in_feature, mask_feature)
        grad = np.gradient(eopatch[in_feature], np.asarray([x.toordinal() for x in eopatch.timestamp]), axis=0)
        eopatch[mask_feature] = grad >= 0

        pos_transition = [
            [[1, 0], [1, 0], [0, 1], [0, 1]],
            [[0, 1], [1, 0], [1, 0], [1, 0]]
        ]
        pos_der = [
            [[-1, 0, 0], [57, 12, 0.08333333], [87, 6, 2.8333333], [33, 6, 0.16666666]],
            [[30, 6, 0.6666666], [-1, 0, 0], [72, 6, 1.666666], [141, 12, 0.25]]
        ]

        out_patch = task(eopatch)
        self.assertTrue(np.array_equal(out_patch.mask_timeless['derivative_transition'], np.array(pos_transition)))
        self.assertTrue(np.allclose(out_patch.data_timeless['derivative'], np.array(pos_der)))

        # Negative
        eopatch[mask_feature] = grad <= 0
        out_patch = task(eopatch)

        neg_transition = [
            [[1, 1], [0, 1], [1, 0], [1, 0]],
            [[1, 0], [1, 1], [0, 1], [0, 1]]
        ]
        neg_der = [
            [[603, 6, -33.5], [27, 6, -0.1666666], [72, 6, -2], [63, 12, -0.08333333]],
            [[72, 6, -3], [54, 6, -3.6666666], [78, 6, -1.3333333], [66, 6, -0.333333]],
        ]

        self.assertTrue(np.array_equal(out_patch.mask_timeless['derivative_transition'], np.array(neg_transition)))
        self.assertTrue(np.allclose(out_patch.data_timeless['derivative'], np.array(neg_der)))

        fill_value = -2
        # Valid mask feature
        valid_mask = np.array([
            [[1], [0], [0], [1]],
            [[1], [0], [1], [1]],
        ])
        # Set the full mask to 0 on pixel
        eopatch[mask_feature][:, -1, -1] = 0
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'valid_pixels', valid_mask)
        task = SurfaceExtractionTask(in_feature, 'derivative', in_feature, mask_feature,
                                     valid_mask_feature=(FeatureType.MASK_TIMELESS, 'valid_pixels'),
                                     fill_value=fill_value)
        task.execute(eopatch)
        neg_transition_mask = [
            [[1, 1], [fill_value, fill_value], [fill_value, fill_value], [1, 0]],
            [[1, 0], [fill_value, fill_value], [0, 1], [fill_value, fill_value]]
        ]
        neg_der_mask = [
            [[603, 6, -33.5], [fill_value, fill_value, fill_value], [fill_value, fill_value, fill_value],
             [63, 12, -0.08333333]],
            [[72, 6, -3], [fill_value, fill_value, fill_value], [78, 6, -1.3333333],
             [fill_value, fill_value, fill_value]],
        ]

        self.assertTrue(np.array_equal(out_patch.mask_timeless['derivative_transition'],
                                       np.array(neg_transition_mask, dtype=bool)))
        self.assertTrue(np.allclose(out_patch.data_timeless['derivative'], np.array(neg_der_mask)))

    def test_max_mean_len_task(self):
        """Test case for computation of max mean len task

        Fictional ndvi is so that corner cases are considered.
        """
        eopatch = EOPatch(timestamp=list(perdelta(date(2020, 3, 1), date(2020, 3, 23), timedelta(days=6))))
        # Fill EOPatch
        ndvi = np.array([
            [
                [[0], [4], [5], [4]],
                [[2], [3], [16], [11]],
            ],
            [
                [[0], [3], [22], [5]],
                [[6], [19], [8], [9]],
            ],
            [
                [[0], [4], [17], [4]],
                [[20], [-3], [6], [11]],
            ],
            [
                [[0], [4], [5], [4]],
                [[2], [3], [16], [12]],
            ],
        ])
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi)
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'NDVI_limit', np.mean(ndvi, 0))
        in_feature = (FeatureType.DATA, 'NDVI')

        task = MaxMeanLenTask(in_feature, (FeatureType.DATA_TIMELESS, 'NDVI_limit'), 'ndvi',
                              0.9, -1)

        eopatch = task(eopatch)

        mean_len = [
            [[18], [18], [18], [18], ],
            [[18], [6], [18], [18], ],
        ]
        mean_surf = [
            [[18], [84], [282], [96], ],
            [[186], [72], [198], [207], ],
        ]
        self.assertTrue(np.array_equal(mean_len, eopatch.data_timeless['ndvi_max_mean_len']))
        self.assertTrue(np.array_equal(mean_surf, eopatch.data_timeless['ndvi_max_mean_surf']))

        fill_value = -3
        valid_mask = np.array([
            [[0], [1], [0], [1]],
            [[1], [0], [1], [1]],
        ])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'valid_pixels', valid_mask)
        task = MaxMeanLenTask(in_feature, (FeatureType.DATA_TIMELESS, 'NDVI_limit'), 'ndvi', 0.9, -1,
                              valid_mask_feature=(FeatureType.MASK_TIMELESS, 'valid_pixels'), fill_value=fill_value)

        task.execute(eopatch)
        mean_len = [
            [[fill_value], [18], [fill_value], [18], ],
            [[18], [fill_value], [18], [18], ],
        ]
        mean_surf = [
            [[fill_value], [84], [fill_value], [96], ],
            [[186], [fill_value], [198], [207], ],
        ]
        self.assertTrue(np.array_equal(mean_len, eopatch.data_timeless['ndvi_max_mean_len']))
        self.assertTrue(np.array_equal(mean_surf, eopatch.data_timeless['ndvi_max_mean_surf']))

    def test_temporal_sliding_window_task(self):
        """Test case for computation of temporal sliding window

        Fictional ndvi is so that corner cases are considered.
        """
        # EOPatch
        eopatch = EOPatch()
        # Fill EOPatch
        ndvi = np.array([
            [
                [[1], [4], [5], [4]],
                [[2], [3], [16], [11]],
            ],
            [
                [[200], [3], [22], [5]],
                [[6], [19], [8], [9]],
            ],
            [
                [[-1], [4], [17], [4]],
                [[20], [-3], [6], [11]],
            ],
            [
                [[1], [4], [5], [4]],
                [[2], [3], [16], [12]],
            ],
        ])
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi)

        max_task = TemporalRollingWindowTask((FeatureType.DATA, 'NDVI'),
                                             (FeatureType.DATA_TIMELESS, 'NDVI_rolling_max'), np.max, 2)
        min_task = TemporalRollingWindowTask((FeatureType.DATA, 'NDVI'),
                                             (FeatureType.DATA_TIMELESS, 'NDVI_rolling_min'), np.min, 2)
        mean_task = TemporalRollingWindowTask((FeatureType.DATA, 'NDVI'),
                                              (FeatureType.DATA_TIMELESS, 'NDVI_rolling_mean'), np.min, 2)
        eopatch = (max_task * min_task * mean_task)(eopatch)

        correct_max = np.array([
            [[200, 200, 1], [4, 4, 4], [22, 22, 17], [5, 5, 4]],
            [[6, 20, 20], [19, 19, 3], [16, 8, 16], [11, 11, 12]],
        ])
        correct_min = np.array([
            [[1, -1, -1], [3, 3, 4], [5, 17, 5], [4, 4, 4]],
            [[2, 6, 2], [3, -3, -3], [8, 6, 6], [9, 9, 11]],
        ])
        correct_mean = [
            [[1, -1, -1], [3, 3, 4], [5, 17, 5], [4, 4, 4], ],
            [[2, 6, 2], [3, - 3, - 3], [8, 6, 6], [9, 9, 11], ]
        ]

        self.assertTrue(np.array_equal(eopatch.data_timeless["NDVI_rolling_max"], correct_max))
        self.assertTrue(np.array_equal(eopatch.data_timeless["NDVI_rolling_min"], correct_min))
        self.assertTrue(np.array_equal(eopatch.data_timeless["NDVI_rolling_mean"], correct_mean))

        task = TemporalRollingWindowTask((FeatureType.DATA, 'NDVI'), (FeatureType.DATA_TIMELESS, 'temporal_max'),
                                         np.max, 1)
        eopatch = task(eopatch)
        self.assertTrue(np.array_equal(eopatch.data_timeless['temporal_max'], np.moveaxis(ndvi, 0, 2).squeeze(-1)))

        task = TemporalRollingWindowTask((FeatureType.DATA, 'NDVI'), (FeatureType.DATA_TIMELESS, 'temporal_max'),
                                         np.max, 4)
        eopatch = task(eopatch)
        self.assertTrue(np.array_equal(eopatch.data_timeless['temporal_max'], np.max(ndvi, axis=0)))

if __name__ == '__main__':
    unittest.main()
