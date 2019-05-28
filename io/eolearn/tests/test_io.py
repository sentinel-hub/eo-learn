import unittest
import logging

import os
import datetime
import numpy as np

from sentinelhub import BBox, CRS, DataSource, ServiceType, MimeType

from eolearn.io import *
from eolearn.core import EOPatch, FeatureType

logging.basicConfig(level=logging.DEBUG)


class TestEOPatch(unittest.TestCase):
    class TaskTestCase:
        """
        Container for each task case of eolearn-io functionalities
        """

        def __init__(self, name, request, bbox, time_interval,
                     eop=None, layer=None, data_size=None, timestamp_length=None, feature_type=FeatureType.DATA):
            self.name = name
            self.request = request
            self.layer = layer
            self.data_size = data_size
            self.timestamp_length = timestamp_length
            self.feature_type = feature_type
            self.time_interval = time_interval
            self.bbox = bbox

            if eop is None:
                self.eop = request.execute(bbox=bbox, time_interval=time_interval)
            elif isinstance(eop, EOPatch):
                self.eop = request.execute(eop, bbox=bbox, time_interval=time_interval)
            else:
                raise TypeError('Task {}: Argument \'eop\' should be an EOPatch, not {}'.format(
                    name, eop.__class__.__name__))

    @classmethod
    def setUpClass(cls):

        bbox = BBox(bbox=(378450, 5345200, 383650, 5350000), crs=CRS.UTM_30N)
        long_time_interval = ('2019-3-18', '2019-4-18')
        long_time_interval_datetime = (datetime.datetime(2019, 3, 18), datetime.datetime(2019, 4, 18))
        single_time = ('2019-3-26', '2019-3-26')
        single_time_l8 = ('2019-4-2', '2019-4-2')
        img_width = 100
        img_height = 100
        resx = '52m'
        resy = '48m'

        instance_id = os.environ.get('INSTANCE_ID')

        # existing eopatch
        cls.eeop = SentinelHubWMSInput(
            layer='BANDS-S2-L1C',
            height=img_height,
            width=img_width,
            data_source=DataSource.SENTINEL2_L1C,
            instance_id=instance_id
        ).execute(bbox=bbox, time_interval=long_time_interval)

        cls.create_patches = [

            cls.TaskTestCase(
                name='generalWmsTask',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=6,
                request=SentinelHubWMSInput(
                    layer='BANDS-S2-L1C',
                    height=img_height,
                    width=img_width,
                    data_source=DataSource.SENTINEL2_L1C,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=long_time_interval,
                eop=None
            ),

            cls.TaskTestCase(
                name='generalWcsTask',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=1,
                request=SentinelHubWCSInput(
                    layer='BANDS-S2-L1C',
                    resx=resx,
                    resy=resy,
                    data_source=DataSource.SENTINEL2_L1C,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
                eop=EOPatch()
            ),

            cls.TaskTestCase(
                name='S2 L1C WMS',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=1,
                request=S2L1CWMSInput(
                    layer='BANDS-S2-L1C',
                    height=img_height,
                    width=img_width,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
                eop=EOPatch()
            ),

            cls.TaskTestCase(
                name='S2 L1C WCS',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=1,
                request=S2L1CWCSInput(
                    layer='BANDS-S2-L1C',
                    resx=resx,
                    resy=resy,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
                eop=EOPatch()
            ),

            cls.TaskTestCase(
                name='L8 L1C WMS',
                layer='TRUE-COLOR-L8',
                data_size=3,
                timestamp_length=1,
                request=L8L1CWMSInput(
                    layer='TRUE-COLOR-L8',
                    height=img_height,
                    width=img_width,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time_l8,
                eop=EOPatch()
            ),

            cls.TaskTestCase(
                name='L8 L1C WCS',
                layer='TRUE-COLOR-L8',
                data_size=3,
                timestamp_length=1,
                request=L8L1CWCSInput(
                    layer='TRUE-COLOR-L8',
                    resx=resx,
                    resy=resy,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time_l8,
                eop=EOPatch()
            ),

            cls.TaskTestCase(
                name='S2 L2A WMS',
                layer='BANDS-S2-L2A',
                data_size=12,
                timestamp_length=1,
                request=S2L2AWMSInput(
                    layer='BANDS-S2-L2A',
                    height=img_height,
                    width=img_width,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
                eop=EOPatch()
            ),

            cls.TaskTestCase(
                name='S2 L2A WCS',
                layer='BANDS-S2-L2A',
                data_size=12,
                timestamp_length=1,
                request=S2L2AWCSInput(
                    layer='BANDS-S2-L2A',
                    resx=resx,
                    resy=resy,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
                eop=EOPatch()
            )
        ]

        cls.create_patches_datetime = [

            cls.TaskTestCase(
                name='generalWmsTask',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=3,
                request=SentinelHubWMSInput(
                    layer='BANDS-S2-L1C',
                    height=img_height,
                    width=img_width,
                    data_source=DataSource.SENTINEL2_L1C,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=long_time_interval_datetime,
                eop=None
            )
        ]

        cls.update_patches = [
            cls.TaskTestCase(
                name='generalWmsTask_to_empty',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=1,
                eop=EOPatch(),
                request=SentinelHubWMSInput(
                    layer='BANDS-S2-L1C',
                    height=img_height,
                    width=img_width,
                    data_source=DataSource.SENTINEL2_L1C,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
            ),

            cls.TaskTestCase(
                name='DEM_to_existing_patch',
                layer='DEM',
                data_size=1,
                timestamp_length=6,
                eop=cls.eeop.__deepcopy__(),
                request=DEMWCSInput(
                    layer='DEM',
                    resx=resx,
                    resy=resy,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=long_time_interval,
                feature_type=FeatureType.DATA_TIMELESS
            ),
            cls.TaskTestCase(
                name='Sen2Cor_to_existing_patch',
                layer='SCL',
                data_size=1,
                timestamp_length=6,
                eop=cls.eeop.__deepcopy__(),
                request=AddSen2CorClassificationFeature(
                    sen2cor_classification='SCL',
                    layer='BANDS-S2-L2A',
                    service_type=ServiceType.WCS,
                    size_x=resx,
                    size_y=resy,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=long_time_interval,
                feature_type=FeatureType.MASK
            ),
        ]

        cls.task_cases_image_format = [

            cls.TaskTestCase(
                name='generalWcsTask_image_format',
                layer='BANDS-S2-L1C',
                data_size=13,
                timestamp_length=1,
                request=SentinelHubWCSInput(
                    layer='BANDS-S2-L1C',
                    resx=resx,
                    resy=resy,
                    image_format=MimeType.TIFF_d32f,
                    data_source=DataSource.SENTINEL2_L1C,
                    instance_id=instance_id
                ),
                bbox=bbox,
                time_interval=single_time,
                eop=EOPatch()
            )

        ]

        cls.task_cases = cls.create_patches + cls.update_patches + cls.task_cases_image_format
        cls.task_cases_datetime = cls.create_patches_datetime

    def test_return_type(self):
        for task in self.task_cases:
            with self.subTest(msg='Test case {}'.format(task.name)):
                self.assertTrue(isinstance(task.eop, EOPatch),
                                "Expected return type of task is EOPatch!")

    def test_time_interval(self):
        for task in self.task_cases:
            with self.subTest(msg='Test case {}'.format(task.name)):
                self.assertEqual(task.eop.meta_info['time_interval'], task.time_interval)

    def test_timestamps_size(self):
        for task in self.task_cases:
            with self.subTest(msg='Test case {}'.format(task.name)):
                self.assertEqual(len(task.eop.timestamp), task.timestamp_length)

    def test_auto_feature_presence(self):
        for task in self.task_cases:
            with self.subTest(msg='Test case {}'.format(task.name)):
                self.assertTrue(task.layer in task.eop[task.feature_type],
                                msg='Feature {} should be in {}'.format(task.layer, task.feature_type))
                self.assertTrue('IS_DATA' in task.eop.mask)

    def test_feature_dimension(self):
        for task in self.task_cases:
            with self.subTest(msg='Test case {}'.format(task.name)):

                masks = [task.layer]
                for mask in masks:
                    if task.eop.data and mask in task.eop.data:
                        self.assertTrue(isinstance(task.eop.data[mask], np.ndarray))
                        self.assertEqual(task.eop.data[mask].shape[-1], task.data_size)

                masks = ['DEM']
                for mask in masks:
                    if task.eop.data_timeless and mask in task.eop.data_timeless:
                        self.assertTrue(isinstance(task.eop.data_timeless[mask], np.ndarray))
                        self.assertEqual(task.eop.data_timeless[mask].shape[-1], task.data_size)

                masks = ['IS_DATA']
                for mask in masks:
                    if task.eop.mask and mask in task.eop.mask:
                        self.assertTrue(isinstance(task.eop.mask[mask], np.ndarray))
                        self.assertEqual(task.eop.mask[mask].shape[-1], 1)
                        mask_dtype = task.eop.mask[mask].dtype
                        self.assertEqual(mask_dtype, np.dtype(np.bool),
                                         msg='Valid data mask should be boolean type, found {}'.format(mask_dtype))

    def test_time_interval_datetime(self):
        for task in self.task_cases_datetime:
            with self.subTest(msg='Test case {}'.format(task.name)):
                self.assertEqual(task.eop.meta_info['time_interval'], task.time_interval)

    def test_bbox(self):
        for task in self.task_cases:
            with self.subTest(msg='Test case {}'.format(task.name)):
                self.assertEqual(task.eop.bbox, task.bbox)


class TestTimelessFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res_x = '53m'
        cls.res_y = '78m'
        cls.instance_id = os.environ.get('INSTANCE_ID')
        cls.time_interval = ('2018-1-5', '2018-1-6')
        cls.bbox = BBox(bbox=(-5.05, 48.0, -5.00, 48.05), crs=CRS.WGS84)
        cls.dem_request = DEMWCSInput(
            layer='DEM',
            resx=cls.res_x,
            resy=cls.res_y,
            instance_id=cls.instance_id
        )
        cls.timed_request = SentinelHubWCSInput(
            layer='BANDS-S2-L1C',
            resx=cls.res_x,
            resy=cls.res_y,
            data_source=DataSource.SENTINEL2_L1C,
            instance_id=cls.instance_id
        )

    def test_dem_to_empty_eopatch(self):
        with self.subTest(msg='Test to add DEM (timeless) to empty eopatch'):
            data = self.dem_request.execute(bbox=self.bbox)
            self.assertTrue(len(data.timestamp) == 0)
            self.assertFalse(data.meta_info['time_interval'])

    def test_dem_to_eopatch_with_timestamps(self):
        with self.subTest(msg='Test to add DEM (timeless) to eopatch with non-empty timestamps'):
            timed_eopatch = self.timed_request.execute(bbox=self.bbox, time_interval=self.time_interval)
            prior_timestamps = timed_eopatch.timestamp.copy()
            dem_eopatch = self.dem_request.execute(eopatch=timed_eopatch, bbox=self.bbox)
            self.assertListEqual(dem_eopatch.timestamp, prior_timestamps,
                                 'Timestamps before and after adding dem differ!')
            self.assertTrue(dem_eopatch.meta_info['time_interval'], '\'time_interval\' in meta_info is expected!')

    def test_time_data_to_dem_eopatch(self):
        with self.subTest(msg='Test to add timed data to DEM (timeless) eopatch'):
            dem_eopatch = self.dem_request.execute(bbox=self.bbox)
            timed_eopatch = self.timed_request.execute(eopatch=dem_eopatch, bbox=self.bbox,
                                                       time_interval=self.time_interval)
            self.assertTrue(len(timed_eopatch.timestamp) == 1, 'timed_eopatch should have 1 timestamp!')
            self.assertTrue(timed_eopatch.meta_info['time_interval'], '\'time_interval\' in meta_info is expected!')


class TestImageFormats(unittest.TestCase):
    class ImageFormatTestCase:
        """
        Container for each task case of eolearn-io functionalities
        """

        def __init__(self, name, request, feature_name, feature_type):
            self.name = name
            self.request = request
            self.feature_name = feature_name
            self.feature_type = feature_type

            bbox = BBox(bbox=(-5.05, 48.0, -5.00, 48.05), crs=CRS.WGS84)
            time_interval = ('2018-1-5', '2018-1-10')

            # default mime type is TIFF (d16)
            self.eop = request.execute(bbox=bbox, time_interval=time_interval)
            request.image_format = MimeType.TIFF_d32f
            self.eop_32bit = request.execute(bbox=bbox, time_interval=time_interval)

    @classmethod
    def setUpClass(cls):

        resx = '53m'
        resy = '78m'
        instance_id = os.environ.get('INSTANCE_ID')

        cls.tests = [
            cls.ImageFormatTestCase(
                name='SentinelHubWCSInput',
                feature_type=FeatureType.DATA,
                feature_name='BANDS-S2-L1C',
                request=SentinelHubWCSInput(
                    layer='BANDS-S2-L1C',
                    resx=resx,
                    resy=resy,
                    data_source=DataSource.SENTINEL2_L1C,
                    instance_id=instance_id
                )
            ),
            cls.ImageFormatTestCase(
                name='S2_L1C_WCS_task',
                feature_type=FeatureType.DATA,
                feature_name='BANDS-S2-L1C',
                request=S2L1CWCSInput(
                    layer='BANDS-S2-L1C',
                    resx=resx,
                    resy=resy,
                    instance_id=instance_id
                )
            ),
            cls.ImageFormatTestCase(
                name='L8L1CWCSInput',
                feature_type=FeatureType.DATA,
                feature_name='TRUE-COLOR-L8',
                request=L8L1CWCSInput(
                    layer='TRUE-COLOR-L8',
                    resx=resx,
                    resy=resy,
                    instance_id=instance_id
                )
            ),
            cls.ImageFormatTestCase(
                name='Sen2CorClassificationFeature - CLD',
                feature_type=FeatureType.DATA,
                feature_name='CLD',
                request=AddSen2CorClassificationFeature(
                    sen2cor_classification='CLD',
                    layer='BANDS-S2-L2A',
                    service_type=ServiceType.WCS,
                    size_x=resx,
                    size_y=resy,
                    instance_id=instance_id
                )
            ),
            cls.ImageFormatTestCase(
                name='AddSen2CorClassificationFeature - SNW',
                feature_type=FeatureType.DATA,
                feature_name='SNW',
                request=AddSen2CorClassificationFeature(
                    sen2cor_classification='SNW',
                    layer='BANDS-S2-L2A',
                    service_type=ServiceType.WCS,
                    size_x=resx,
                    size_y=resy,
                    instance_id=instance_id
                )
            ),
            cls.ImageFormatTestCase(
                name='AddSen2CorClassificationFeature - SCL',
                feature_type=FeatureType.MASK,
                feature_name='SCL',
                request=AddSen2CorClassificationFeature(
                    sen2cor_classification='SCL',
                    layer='BANDS-S2-L2A',
                    service_type=ServiceType.WCS,
                    size_x=resx,
                    size_y=resy,
                    instance_id=instance_id
                )
            ),
            # cls.ImageFormatTestCase(
            #     name='DEMWCSInput',
            #     feature_type=FeatureType.MASK_TIMELESS,
            #     feature_name='DEM',
            #     request=DEMWCSInput(
            #         layer='DEM',
            #         resx=resx,
            #         resy=resy,
            #         instance_id=instance_id
            #     )
            # )

        ]

        cls.dem_test = [
        ]

    def test_image_format(self):
        for test in self.tests:
            with self.subTest(msg='Test case {}'.format(test.name)):
                if test.feature_type == FeatureType.DATA:
                    np.testing.assert_allclose(test.eop.data[test.feature_name],
                                               test.eop_32bit.data[test.feature_name],
                                               atol=1e-4,
                                               err_msg='The data values are too different')
                if test.feature_type == FeatureType.MASK:
                    np.testing.assert_allclose(test.eop.mask[test.feature_name],
                                               test.eop_32bit.mask[test.feature_name],
                                               atol=0,
                                               err_msg='The data values are too different')


if __name__ == '__main__':
    unittest.main()
