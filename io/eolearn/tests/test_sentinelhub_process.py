"""
Credits:
Copyright (c) 2019-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2019-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2021 Beno Šircelj

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import dataclasses
import datetime as dt
import os
import shutil
from concurrent import futures
from typing import Any, Optional

import numpy as np
import pytest
from pytest import approx

from sentinelhub import CRS, Band, BBox, DataCollection, SHConfig, Unit

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.io import (
    SentinelHubDemTask,
    SentinelHubEvalscriptTask,
    SentinelHubInputTask,
    SentinelHubSen2corTask,
    get_available_timestamps,
)


@pytest.fixture(name="cache_folder")
def cache_folder_fixture():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    cache_folder = os.path.join(test_dir, "cache_test")

    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)

    yield cache_folder

    shutil.rmtree(cache_folder)


@dataclasses.dataclass
class IoTestCase:
    name: str
    task: EOTask
    bbox: BBox
    time_interval: tuple
    feature: str = "BANDS"
    feature_type: FeatureType = FeatureType.DATA
    data_size: Optional[int] = None
    timestamp_length: Optional[int] = None
    stats: Any = None


def calculate_stats(array):
    time, height, width, _ = array.shape

    slices = [
        array[int(time / 2) :, 0, 0, :],
        array[: max(int(time / 2), 1), -1, -1, :],
        array[:, int(height / 2), int(width / 2), :],
    ]
    values = [(np.nanmean(slice) if not np.isnan(slice).all() else np.nan) for slice in slices]
    return np.round(np.array(values), 4)


@pytest.mark.sh_integration
class TestProcessingIO:
    """Test cases for SentinelHubInputTask"""

    size = (99, 101)
    bbox = BBox(bbox=[268892, 4624365, 268892 + size[0] * 10, 4624365 + size[1] * 10], crs=CRS.UTM_33N)
    time_interval = ("2017-12-15", "2017-12-30")
    maxcc = 0.8
    time_difference = dt.timedelta(minutes=60)
    max_threads = 3

    def test_s2l1c_float32_uint16(self, cache_folder):
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            additional_data=[(FeatureType.MASK, "dataMask")],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L1C,
            max_threads=self.max_threads,
            cache_folder=cache_folder,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, "BANDS")]
        is_data = eopatch[(FeatureType.MASK, "dataMask")]

        assert calculate_stats(bands) == approx([0.0233, 0.0468, 0.0252])

        width, height = self.size
        assert bands.shape == (4, height, width, 13)
        assert is_data.shape == (4, height, width, 1)
        assert len(eopatch.timestamp) == 4
        assert bands.dtype == np.float32

        assert os.path.exists(cache_folder)

        # change task's bans_dtype and run it again
        task.bands_dtype = np.uint16

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, "BANDS")]

        assert calculate_stats(bands) == approx([232.5769, 467.5385, 251.8654])

        assert bands.dtype == np.uint16

    def test_specific_bands(self):
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            bands=["B01", "B02", "B03"],
            additional_data=[(FeatureType.MASK, "dataMask")],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L1C,
            max_threads=self.max_threads,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, "BANDS")]

        assert calculate_stats(bands) == approx([0.0648, 0.1193, 0.063])

        width, height = self.size
        assert bands.shape == (4, height, width, 3)

    def test_scl_only(self):
        """Download just SCL, without any other bands"""
        task = SentinelHubInputTask(
            bands_feature=None,
            additional_data=[(FeatureType.DATA, "SCL")],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L2A,
            max_threads=self.max_threads,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        scl = eopatch[(FeatureType.DATA, "SCL")]

        width, height = self.size
        assert scl.shape == (4, height, width, 1)

    def test_single_scene(self):
        """Download S2L1C bands and dataMask"""
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            additional_data=[(FeatureType.MASK, "dataMask")],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L1C,
            max_threads=self.max_threads,
            single_scene=True,
            mosaicking_order="leastCC",
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, "BANDS")]
        is_data = eopatch[(FeatureType.MASK, "dataMask")]

        width, height = self.size
        assert bands.shape == (1, height, width, 13)
        assert is_data.shape == (1, height, width, 1)
        assert len(eopatch.timestamp) == 1

    def test_additional_data(self):
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            bands=["B01", "B02", "B05"],
            additional_data=[
                (FeatureType.MASK, "dataMask", "IS_DATA"),
                (FeatureType.MASK, "CLM"),
                (FeatureType.MASK, "SCL"),
                (FeatureType.MASK, "SNW"),
                (FeatureType.MASK, "CLD"),
                (FeatureType.DATA, "CLP"),
                (FeatureType.DATA, "viewAzimuthMean", "view_azimuth_mean"),
                (FeatureType.DATA, "sunAzimuthAngles"),
                (FeatureType.DATA, "sunZenithAngles"),
            ],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L2A,
            max_threads=self.max_threads,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)

        bands = eopatch[(FeatureType.DATA, "BANDS")]
        is_data = eopatch[(FeatureType.MASK, "IS_DATA")]
        clm = eopatch[(FeatureType.MASK, "CLM")]
        scl = eopatch[(FeatureType.MASK, "SCL")]
        snw = eopatch[(FeatureType.MASK, "SNW")]
        cld = eopatch[(FeatureType.MASK, "CLD")]
        clp = eopatch[(FeatureType.DATA, "CLP")]
        view_azimuth_mean = eopatch[(FeatureType.DATA, "view_azimuth_mean")]
        sun_azimuth_angles = eopatch[(FeatureType.DATA, "sunAzimuthAngles")]
        sun_zenith_angles = eopatch[(FeatureType.DATA, "sunZenithAngles")]

        assert calculate_stats(bands) == approx([0.027, 0.0243, 0.0162])

        width, height = self.size
        assert bands.shape == (4, height, width, 3)
        assert is_data.shape == (4, height, width, 1)
        assert is_data.dtype == bool
        assert clm.shape == (4, height, width, 1)
        assert clm.dtype == np.uint8
        assert scl.shape == (4, height, width, 1)
        assert snw.shape == (4, height, width, 1)
        assert cld.shape == (4, height, width, 1)
        assert clp.shape == (4, height, width, 1)
        assert view_azimuth_mean.shape == (4, height, width, 1)
        assert sun_azimuth_angles.shape == (4, height, width, 1)
        assert sun_zenith_angles.shape == (4, height, width, 1)
        assert len(eopatch.timestamp) == 4

    def test_aux_request_args(self):
        """Download low resolution data with `PREVIEW` mode"""
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            resolution=260,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L1C,
            max_threads=self.max_threads,
            aux_request_args={"dataFilter": {"previewMode": "PREVIEW"}},
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)
        bands = eopatch[(FeatureType.DATA, "BANDS")]

        assert bands.shape == (4, 4, 4, 13)
        assert calculate_stats(bands) == approx([0.0, 0.0493, 0.0277])

    def test_dem(self):
        task = SentinelHubDemTask(resolution=10, feature=(FeatureType.DATA_TIMELESS, "DEM"), max_threads=3)

        eopatch = task.execute(bbox=self.bbox)
        dem = eopatch.data_timeless["DEM"]

        width, height = self.size
        assert dem.shape == (height, width, 1)

    def test_dem_cop(self):
        task = SentinelHubDemTask(
            data_collection=DataCollection.DEM_COPERNICUS_30,
            resolution=10,
            feature=(FeatureType.DATA_TIMELESS, "DEM_30"),
            max_threads=3,
        )
        eopatch = task.execute(bbox=self.bbox)
        dem = eopatch.data_timeless["DEM_30"]

        width, height = self.size
        assert dem.shape == (height, width, 1)

    def test_dem_wrong_feature(self):
        with pytest.raises(ValueError):
            SentinelHubDemTask(resolution=10, feature=(FeatureType.DATA, "DEM"), max_threads=3)

    def test_sen2cor(self):
        task = SentinelHubSen2corTask(
            sen2cor_classification=["SCL", "CLD"],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L2A,
            max_threads=self.max_threads,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=self.time_interval)

        scl = eopatch[(FeatureType.MASK, "SCL")]
        cld = eopatch[(FeatureType.DATA, "CLD")]

        width, height = self.size
        assert scl.shape == (4, height, width, 1)
        assert cld.shape == (4, height, width, 1)

    def test_metadata(self):
        evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands:["B02","dataMask"],
                        units: "DN"
                    }],
                    output:[
                    {
                        id:'bands',
                        bands: 1,
                        sampleType: SampleType.UINT16
                    }
                    ]
                }
            }

            function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
                outputMetadata.userData = { "metadata":  JSON.stringify(scenes) }
            }

            function evaluatePixel(sample) {
                return {
                    'bands': [sample.B02]
                };
            }
        """
        task = SentinelHubEvalscriptTask(
            evalscript=evalscript,
            data_collection=DataCollection.SENTINEL2_L1C,
            features=[(FeatureType.DATA, "bands"), (FeatureType.META_INFO, "meta_info")],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            max_threads=self.max_threads,
        )

        eop = task.execute(bbox=self.bbox, time_interval=self.time_interval)

        width, height = self.size
        assert eop.data["bands"].shape == (4, height, width, 1)
        assert len(eop.meta_info["meta_info"]) == 4

    def test_multi_processing(self):
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            bands=["B01", "B02", "B03"],
            additional_data=[(FeatureType.MASK, "dataMask")],
            size=self.size,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            data_collection=DataCollection.SENTINEL2_L1C,
            max_threads=self.max_threads,
        )

        time_intervals = [
            ("2017-01-01", "2017-01-30"),
            ("2017-02-01", "2017-02-28"),
            ("2017-03-01", "2017-03-30"),
            ("2017-04-01", "2017-04-30"),
            ("2017-05-01", "2017-05-30"),
            ("2017-06-01", "2017-06-30"),
        ]

        with futures.ProcessPoolExecutor(max_workers=3) as executor:
            tasks = [executor.submit(task.execute, None, self.bbox, interval) for interval in time_intervals]
            eopatches = [task.result() for task in futures.as_completed(tasks)]

        array = np.concatenate([eop.data["BANDS"] for eop in eopatches], axis=0)

        width, height = self.size
        assert array.shape == (20, height, width, 3)

    def test_get_available_timestamps_with_missing_data_collection_service_url(self):
        collection = DataCollection.SENTINEL2_L1C.define_from("COLLECTION_WITHOUT_URL", service_url=None)
        timestamps = get_available_timestamps(
            bbox=self.bbox,
            config=SHConfig(),
            data_collection=collection,
            time_difference=self.time_difference,
            time_interval=self.time_interval,
            maxcc=self.maxcc,
        )

        assert len(timestamps) == 4
        assert all(timestamp.tzinfo is not None for timestamp in timestamps)

    def test_no_data_input_task_request(self):
        task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, "BANDS"),
            additional_data=[(FeatureType.MASK, "dataMask")],
            size=self.size,
            maxcc=0.0,
            data_collection=DataCollection.SENTINEL2_L1C,
        )
        eopatch = task.execute(bbox=self.bbox, time_interval=("2021-01-01", "2021-01-20"))

        bands = eopatch[FeatureType.DATA, "BANDS"]
        assert bands.shape == (0, 101, 99, 13)
        masks = eopatch[FeatureType.MASK, "dataMask"]
        assert masks.shape == (0, 101, 99, 1)

    def test_no_data_evalscript_task_request(self):
        evalscript = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands:["B02", "dataMask"],
                    units: "DN"
                }],
                output:[
                  {
                    id:'bands',
                    bands: 2,
                    sampleType: SampleType.UINT16
                  },
                  {
                    id:'mask',
                    bands: 1,
                    sampleType: SampleType.UINT8
                  }
                ]
            }
        }

        function evaluatePixel(sample) {
            return {
                'bands': [sample.B02, sample.B02],
                'mask': [sample.dataMask]
            };
        }
    """
        task = SentinelHubEvalscriptTask(
            evalscript=evalscript,
            data_collection=DataCollection.SENTINEL2_L1C,
            features=[(FeatureType.DATA, "bands"), (FeatureType.MASK, "mask")],
            size=self.size,
            maxcc=0.0,
        )

        eopatch = task.execute(bbox=self.bbox, time_interval=("2021-01-01", "2021-01-20"))

        bands = eopatch[FeatureType.DATA, "bands"]
        assert bands.shape == (0, 101, 99, 2)
        masks = eopatch[FeatureType.MASK, "mask"]
        assert masks.shape == (0, 101, 99, 1)


@pytest.mark.sh_integration
class TestSentinelHubInputTaskDataCollections:
    """Integration tests for all supported data collections"""

    bbox = BBox(bbox=(-5.05, 48.0, -5.00, 48.05), crs=CRS.WGS84)
    bbox2 = BBox(bbox=(-72.2, -70.4, -71.6, -70.2), crs=CRS.WGS84)
    size = (50, 40)
    time_interval = ("2020-06-1", "2020-06-10")
    time_difference = dt.timedelta(minutes=60)
    data_feature = FeatureType.DATA, "BANDS"
    mask_feature = FeatureType.MASK, "dataMask"

    s3slstr_500m = DataCollection.SENTINEL3_SLSTR.define_from(
        "SENTINEL3_SLSTR_500m",
        bands=(
            Band("S2", (Unit.REFLECTANCE,), (np.float32,)),
            Band("S3", (Unit.REFLECTANCE,), (np.float32,)),
            Band("S6", (Unit.REFLECTANCE,), (np.float32,)),
        ),
    )
    s5p_co = DataCollection.SENTINEL5P.define_from("SENTINEL5P_CO", bands=(Band("CO", (Unit.DN,), (np.float32,)),))

    ndvi_evalscript = """
        //VERSION=3
        function setup() {
            return {
            input: [{
                bands: ["B04", "B08", "dataMask"],
                units: ["REFLECTANCE", "REFLECTANCE", "DN"]
            }],
            output: [
                { id:"ndvi", bands:1, sampleType: SampleType.FLOAT32 },
                { id:"dataMask", bands:1, sampleType: SampleType.UINT8 }
            ]
            }
        }
        function evaluatePixel(sample) {
        return {
            ndvi: [index(sample.B08, sample.B04)],
            dataMask: [sample.dataMask]};
        }
    """

    test_cases = [
        IoTestCase(
            name="Sentinel-2 L2A",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.SENTINEL2_L2A,
            ),
            bbox=bbox,
            time_interval=time_interval,
            data_size=12,
            timestamp_length=2,
            stats=[0.4676, 0.6313, 0.7688],
        ),
        IoTestCase(
            name="Sentinel-2 L2A - NDVI evalscript",
            task=SentinelHubEvalscriptTask(
                features={
                    FeatureType.DATA: [("ndvi", "NDVI-FEATURE")],
                    FeatureType.MASK: ["dataMask"],
                },
                evalscript=ndvi_evalscript,
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.SENTINEL2_L2A,
            ),
            feature="NDVI-FEATURE",
            bbox=bbox,
            time_interval=time_interval,
            data_size=1,
            timestamp_length=2,
            stats=[0.0088, 0.0083, 0.0008],
        ),
        IoTestCase(
            name="Landsat8",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.LANDSAT_OT_L1,
            ),
            bbox=bbox,
            time_interval=time_interval,
            data_size=11,
            timestamp_length=1,
            stats=[48.7592, 48.726, 48.9168],
        ),
        IoTestCase(
            name="MODIS",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.MODIS,
            ),
            bbox=bbox,
            time_interval=time_interval,
            data_size=7,
            timestamp_length=10,
            stats=[0.0073, 0.0101, 0.1448],
        ),
        IoTestCase(
            name="Sentinel-1 IW",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.SENTINEL1_IW,
            ),
            bbox=bbox,
            time_interval=time_interval,
            data_size=2,
            timestamp_length=5,
            stats=[0.016, 0.0022, 0.0087],
        ),
        IoTestCase(
            name="Sentinel-1 IW ASCENDING",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.SENTINEL1_IW_ASC,
            ),
            bbox=bbox,
            time_interval=time_interval,
            data_size=2,
            timestamp_length=1,
            stats=[0.0407, 0.0206, 0.0216],
        ),
        IoTestCase(
            name="Sentinel-1 EW DESCENDING",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.SENTINEL1_EW_DES,
            ),
            bbox=bbox2,
            time_interval=time_interval,
            data_size=2,
            timestamp_length=1,
            stats=[np.nan, 0.1944, 0.3800],
        ),
        IoTestCase(
            name="Sentinel-3 OLCI",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=DataCollection.SENTINEL3_OLCI,
            ),
            bbox=bbox,
            time_interval=time_interval,
            data_size=21,
            timestamp_length=11,
            stats=[0.317, 0.1946, 0.2884],
        ),
        IoTestCase(
            name="Sentinel-3 SLSTR 500m resolution",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=s3slstr_500m,
            ),
            bbox=bbox,
            time_interval=("2021-02-10", "2021-02-15"),
            data_size=3,
            timestamp_length=13,
            stats=[0.3173, 0.4804, 0.4041],
        ),
        IoTestCase(
            name="Sentinel-5P",
            task=SentinelHubInputTask(
                bands_feature=data_feature,
                additional_data=[mask_feature],
                size=size,
                time_difference=time_difference,
                data_collection=s5p_co,
            ),
            bbox=bbox,
            time_interval=("2020-06-1", "2020-06-1"),
            data_size=1,
            timestamp_length=1,
            stats=[0.0351, 0.034, 0.0351],
        ),
    ]

    @pytest.mark.parametrize("test_case", test_cases)
    def test_data_collections(self, test_case):
        eopatch = test_case.task.execute(bbox=test_case.bbox, time_interval=test_case.time_interval)

        assert isinstance(eopatch, EOPatch), "Expected return type is EOPatch"

        width, height = self.size
        data = eopatch[(test_case.feature_type, test_case.feature)]
        assert data.shape == (test_case.timestamp_length, height, width, test_case.data_size)

        timestamps = eopatch.timestamp
        assert all(timestamp.tzinfo is None for timestamp in timestamps), f"`tzinfo` present in timestamps {timestamps}"
        assert len(timestamps) == test_case.timestamp_length

        stats = calculate_stats(data)
        assert stats == approx(test_case.stats, nan_ok=True), f"Expected stats {test_case.stats}, got {stats}"
