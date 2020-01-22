"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import unittest
import os

import numpy as np

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import VectorToRaster, RasterToVector
from shapely.geometry import Polygon


class TestVectorToRaster(unittest.TestCase):
    """ Testing transformation vector -> raster
    """
    class TestCase:
        """
        Container for each test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                           'TestEOPatch')

        def __init__(self, name, task, img_min=0, img_max=0, img_mean=0, img_median=0, img_dtype=None, img_shape=None):
            self.name = name
            self.task = task
            self.img_min = img_min
            self.img_max = img_max
            self.img_mean = img_mean
            self.img_median = img_median
            self.img_dtype = img_dtype
            self.img_shape = img_shape

            self.result = None

        def execute(self):
            eopatch = EOPatch.load(self.TEST_PATCH_FILENAME)
            self.result = self.task.execute(eopatch)
            self.result = self.task.execute(self.result)

    @classmethod
    def setUpClass(cls):
        cls.vector_feature = FeatureType.VECTOR_TIMELESS, 'LULC'
        cls.raster_feature = FeatureType.MASK_TIMELESS, 'RASTERIZED_LULC'

        custom_dataframe = EOPatch.load(cls.TestCase.TEST_PATCH_FILENAME).vector_timeless['LULC']
        custom_dataframe = custom_dataframe[(custom_dataframe['AREA'] < 10 ** 3)]

        reprojected_dataframe = custom_dataframe.to_crs(epsg=3857)

        cls.test_cases = [
            cls.TestCase('basic test',
                         VectorToRaster(cls.vector_feature, cls.raster_feature, values_column='LULC_ID',
                                        raster_shape=(FeatureType.DATA, 'BANDS-S2-L1C'), no_data_value=20),
                         img_min=0, img_max=8, img_mean=2.33267, img_median=2, img_dtype=np.uint8,
                         img_shape=(101, 100, 1)),
            cls.TestCase('single value filter, fixed shape',
                         VectorToRaster(cls.vector_feature, cls.raster_feature, values=8, values_column='LULC_ID',
                                        raster_shape=(50, 50), no_data_value=20, write_to_existing=True,
                                        raster_dtype=np.int32),
                         img_min=8, img_max=20, img_mean=19.76, img_median=20, img_dtype=np.int32,
                         img_shape=(50, 50, 1)),
            cls.TestCase('multiple values filter, resolution, all touched',
                         VectorToRaster(cls.vector_feature, cls.raster_feature, values=[1, 5], values_column='LULC_ID',
                                        raster_resolution='60m', no_data_value=13, raster_dtype=np.uint16,
                                        all_touched=True, write_to_existing=False),
                         img_min=1, img_max=13, img_mean=12.7093, img_median=13, img_dtype=np.uint16,
                         img_shape=(17, 17, 1)),
            cls.TestCase('deprecated parameters, single value, custom resolution',
                         VectorToRaster(vector_input=custom_dataframe, raster_feature=cls.raster_feature, values=14,
                                        raster_resolution=(32, 15), no_data_value=-1, raster_dtype=np.int32),
                         img_min=-1, img_max=14, img_mean=-0.8411, img_median=-1, img_dtype=np.int32,
                         img_shape=(67, 31, 1)),
            cls.TestCase('empty vector data test',
                         VectorToRaster(vector_input=custom_dataframe[
                                            (custom_dataframe.LULC_NAME == 'some_none_existent_name')],
                                        raster_feature=cls.raster_feature,
                                        values_column='LULC_ID',
                                        raster_shape=(FeatureType.DATA, 'BANDS-S2-L1C'), no_data_value=0),
                         img_min=0, img_max=0, img_mean=0, img_median=0, img_dtype=np.uint8,
                         img_shape=(101, 100, 1)),
            cls.TestCase('negative polygon buffering',
                         VectorToRaster(vector_input=custom_dataframe,
                                        raster_feature=cls.raster_feature,
                                        values_column='LULC_ID',
                                        buffer=-2,
                                        raster_shape=(FeatureType.DATA, 'BANDS-S2-L1C'), no_data_value=0),
                         img_min=0, img_max=8, img_mean=0.0229, img_median=0, img_dtype=np.uint8,
                         img_shape=(101, 100, 1)),
            cls.TestCase('positive polygon buffering',
                         VectorToRaster(vector_input=custom_dataframe,
                                        raster_feature=cls.raster_feature,
                                        values_column='LULC_ID',
                                        buffer=2,
                                        raster_shape=(FeatureType.DATA, 'BANDS-S2-L1C'), no_data_value=0),
                         img_min=0, img_max=8, img_mean=0.0664, img_median=0, img_dtype=np.uint8,
                         img_shape=(101, 100, 1)),
            cls.TestCase('different crs',
                         VectorToRaster(vector_input=reprojected_dataframe,
                                        raster_feature=cls.raster_feature,
                                        values_column='LULC_ID',
                                        raster_shape=(FeatureType.DATA, 'BANDS-S2-L1C'), no_data_value=0),
                         img_min=0, img_max=8, img_mean=0.042079, img_median=0, img_dtype=np.uint8,
                         img_shape=(101, 100, 1)),
        ]

        for test_case in cls.test_cases:
            test_case.execute()

    def test_result(self):
        for test_case in self.test_cases:
            delta = 1e-3
            data = test_case.result[self.raster_feature[0]][self.raster_feature[1]]

            min_val = np.amin(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.img_min, min_val, delta=delta,
                                       msg="Expected min {}, got {}".format(test_case.img_min, min_val))

            max_val = np.amax(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.img_max, max_val, delta=delta,
                                       msg="Expected max {}, got {}".format(test_case.img_max, max_val))

            mean_val = np.mean(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.img_mean, mean_val, delta=delta,
                                       msg="Expected mean {}, got {}".format(test_case.img_mean, mean_val))

            median_val = np.median(data)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertAlmostEqual(test_case.img_median, median_val, delta=delta,
                                       msg="Expected median {}, got {}".format(test_case.img_median, median_val))

            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertTrue(test_case.img_dtype == data.dtype,
                                msg="Expected dtype {}, got {}".format(test_case.img_dtype, data.dtype))

            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertEqual(test_case.img_shape, data.shape,
                                 msg="Expected shape {}, got {}".format(test_case.img_shape, data.shape))

    def test_polygon_overlap(self):
        patch_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data', 'TestEOPatch')
        patch = EOPatch.load(patch_path)

        # create two test bboxes to overlap existing classes
        bounds = patch.vector_timeless['LULC'].total_bounds
        test_bounds1 = bounds[0] + 500, bounds[1] + 1000, bounds[2] - 1450, bounds[3] - 1650
        test_bounds2 = bounds[0] + 300, bounds[1] + 1400, bounds[2] - 1750, bounds[3] - 1300

        dframe = patch.vector_timeless['LULC'][0:50]

        # override 0th row with a test polygon of class 10
        test_row = dframe.index[0]
        dframe.at[test_row, 'LULC_ID'] = 10
        dframe.at[test_row, 'geometry'] = Polygon.from_bounds(*test_bounds1)

        # override the last row with a test polygon of class 5
        test_row = dframe.index[-1]
        dframe.at[test_row, 'LULC_ID'] = 5
        dframe.at[test_row, 'geometry'] = Polygon.from_bounds(*test_bounds2)

        patch.vector_timeless['TEST'] = dframe

        shape_feature = FeatureType.DATA, 'BANDS-S2-L1C'

        # no overlap
        patch = VectorToRaster(dframe[1:-1], (FeatureType.MASK_TIMELESS, 'OVERLAP_0'),
                               values_column='LULC_ID', raster_shape=shape_feature, overlap_value=5)(patch)

        # overlap without taking intersection into account
        patch = VectorToRaster(dframe, (FeatureType.MASK_TIMELESS, 'OVERLAP_1'),
                               values_column='LULC_ID', raster_shape=shape_feature, overlap_value=None)(patch)

        # overlap without setting intersections to 0
        patch = VectorToRaster(dframe, (FeatureType.MASK_TIMELESS, 'OVERLAP_2'),
                               values_column='LULC_ID', raster_shape=shape_feature, overlap_value=0)(patch)

        # overlap without setting intersections to class 7
        patch = VectorToRaster(dframe, (FeatureType.MASK_TIMELESS, 'OVERLAP_3'),
                               values_column='LULC_ID', raster_shape=shape_feature, overlap_value=7)(patch)

        # separately render bboxes for comparisons in asserts
        patch = VectorToRaster(dframe[:1], (FeatureType.MASK_TIMELESS, 'TEST_BBOX1'),
                               values_column='LULC_ID', raster_shape=shape_feature)(patch)
        patch = VectorToRaster(dframe[-1:], (FeatureType.MASK_TIMELESS, 'TEST_BBOX2'),
                               values_column='LULC_ID', raster_shape=shape_feature)(patch)

        bbox1 = patch.mask_timeless['TEST_BBOX1']
        bbox2 = patch.mask_timeless['TEST_BBOX2']

        overlap0 = patch.mask_timeless['OVERLAP_0']
        overlap1 = patch.mask_timeless['OVERLAP_1']
        overlap2 = patch.mask_timeless['OVERLAP_2']

        # 4 gets partially covered by 5
        self.assertTrue(np.count_nonzero(overlap0 == 4) > np.count_nonzero(overlap1 == 4))
        # 2 doesn't get covered, stays the same
        self.assertTrue(np.count_nonzero(overlap0 == 2) == np.count_nonzero(overlap1 == 2))
        # 10 is bbox2 and it gets covered by other classes
        self.assertTrue(np.count_nonzero(bbox1) > np.count_nonzero(overlap1 == 10))
        # 5 is bbox1 and it is rendered on top of all others, so it doesn't get covered
        self.assertTrue(np.count_nonzero(bbox2) == np.count_nonzero(overlap1 == 5))

        # all classes have their parts intersected, so the sum should reduce
        self.assertTrue(np.count_nonzero(bbox1) > np.count_nonzero(overlap2 == 10))
        self.assertTrue(np.count_nonzero(bbox2) > np.count_nonzero(overlap2 == 5))
        self.assertTrue(np.count_nonzero(overlap0 == 4) > np.count_nonzero(overlap2 == 4))
        # 2 gets covered completely
        self.assertTrue(np.count_nonzero(overlap2 == 2) == 0)


class TestRasterToVector(unittest.TestCase):
    """ Testing transformation raster -> vector
    """
    class TestCase:
        """
        Container for each test case
        """
        TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../example_data',
                                           'TestEOPatch')

        def __init__(self, name, task, feature, data_len, test_reverse=False):
            self.name = name
            self.task = task
            self.feature = feature
            self.data_len = data_len
            self.test_reverse = test_reverse

            self.result = None

        @property
        def vector_feature(self):
            feature_type = FeatureType.VECTOR_TIMELESS if self.feature[0].is_timeless() else FeatureType.VECTOR
            return feature_type, self.feature[-1]

        def execute(self):
            eopatch = EOPatch.load(self.TEST_PATCH_FILENAME)
            self.result = self.task.execute(eopatch)

    @classmethod
    def setUpClass(cls):
        feature1 = FeatureType.MASK_TIMELESS, 'LULC', 'NEW_LULC'
        feature2 = FeatureType.MASK, 'CLM'

        cls.test_cases = [
            cls.TestCase('reverse test',
                         RasterToVector(feature1), feature=feature1, data_len=126, test_reverse=True),
            cls.TestCase('parameters test',
                         RasterToVector(feature2, values=[1, 2], values_column='IS_CLOUD', raster_dtype=np.int16,
                                        connectivity=8),
                         feature=feature2, data_len=54),
        ]

        for test_case in cls.test_cases:
            test_case.execute()

    def test_result(self):
        for test_case in self.test_cases:

            ft, fn = test_case.vector_feature
            data = test_case.result[ft][fn]

            data_len = len(data.index)
            with self.subTest(msg='Test case {}'.format(test_case.name)):
                self.assertEqual(test_case.data_len, data_len,
                                 msg="Expected number of shapes {}, got {}".format(test_case.data_len, data_len))

    def test_transformation_back(self):
        for test_case in self.test_cases:
            if test_case.test_reverse:
                with self.subTest(msg='Test case {}'.format(test_case.name)):

                    new_raster_feature = test_case.feature[0], '{}_NEW'.format(test_case.feature[1])
                    old_raster_feature = test_case.feature[:2]
                    vector2raster_task = VectorToRaster(test_case.vector_feature, new_raster_feature,
                                                        values_column=test_case.task.values_column,
                                                        raster_shape=old_raster_feature)

                    eop = vector2raster_task(test_case.result)

                    new_raster = eop[new_raster_feature[0]][new_raster_feature[1]]
                    old_raster = eop[old_raster_feature[0]][old_raster_feature[1]]
                    self.assertTrue(np.array_equal(new_raster, old_raster),
                                    msg='Old and new raster features should be the same')


if __name__ == '__main__':
    unittest.main()
