"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import dataclasses
from functools import partial
from typing import Any, Optional

import numpy as np
import pytest
import shapely
from conftest import TEST_EOPATCH_PATH
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.geometry import RasterToVectorTask, VectorToRasterTask

VECTOR_FEATURE = FeatureType.VECTOR_TIMELESS, "LULC"
RASTER_FEATURE = FeatureType.MASK_TIMELESS, "RASTERIZED_LULC"

VECTOR_FEATURE_TIMED = FeatureType.VECTOR, "CLM_VECTOR"
RASTER_FEATURE_TIMED = FeatureType.MASK, "RASTERIZED_CLM"


CUSTOM_DATAFRAME = EOPatch.load(TEST_EOPATCH_PATH, lazy_loading=True)[VECTOR_FEATURE]
CUSTOM_DATAFRAME = CUSTOM_DATAFRAME[(CUSTOM_DATAFRAME["AREA"] < 10**3)]
CUSTOM_DATAFRAME_3D = CUSTOM_DATAFRAME.copy()
CUSTOM_DATAFRAME_3D.geometry = CUSTOM_DATAFRAME_3D.geometry.map(partial(shapely.ops.transform, lambda x, y: (x, y, 0)))


@dataclasses.dataclass(frozen=True)
class VectorToRasterTestCase:
    name: str
    task: EOTask
    img_shape: tuple
    img_max: int
    img_min: int = 0
    img_mean: Optional[float] = None
    img_median: Optional[int] = None
    img_dtype: type = np.uint8


VECTOR_TO_RASTER_TEST_CASES = (
    VectorToRasterTestCase(
        name="basic test",
        task=VectorToRasterTask(
            VECTOR_FEATURE,
            RASTER_FEATURE,
            values_column="LULC_ID",
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=20,
        ),
        img_max=8,
        img_mean=2.33267,
        img_median=2,
        img_shape=(101, 100, 1),
    ),
    VectorToRasterTestCase(
        name="basic test timed",
        task=VectorToRasterTask(
            VECTOR_FEATURE_TIMED,
            RASTER_FEATURE_TIMED,
            values_column="VALUE",
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=20,
        ),
        img_min=1,
        img_max=20,
        img_mean=12.4854,
        img_median=20,
        img_shape=(68, 101, 100, 1),
    ),
    VectorToRasterTestCase(
        name="single value filter, fixed shape",
        task=VectorToRasterTask(
            VECTOR_FEATURE,
            RASTER_FEATURE,
            values=8,
            values_column="LULC_ID",
            raster_shape=(50, 50),
            no_data_value=20,
            write_to_existing=True,
            raster_dtype=np.int32,
        ),
        img_min=8,
        img_max=20,
        img_mean=19.76,
        img_median=20,
        img_dtype=np.int32,
        img_shape=(50, 50, 1),
    ),
    VectorToRasterTestCase(
        name="multiple values filter, resolution, all touched",
        task=VectorToRasterTask(
            VECTOR_FEATURE,
            RASTER_FEATURE,
            values=[1, 5],
            values_column="LULC_ID",
            raster_resolution="60m",
            no_data_value=13,
            raster_dtype=np.uint16,
            all_touched=True,
            write_to_existing=False,
        ),
        img_min=1,
        img_max=13,
        img_mean=12.7093,
        img_median=13,
        img_dtype=np.uint16,
        img_shape=(17, 17, 1),
    ),
    VectorToRasterTestCase(
        name="deprecated parameters, single value, custom resolution",
        task=VectorToRasterTask(
            vector_input=CUSTOM_DATAFRAME,
            raster_feature=RASTER_FEATURE,
            values=14,
            raster_resolution=(32, 15),
            no_data_value=-1,
            raster_dtype=np.int32,
        ),
        img_min=-1,
        img_max=14,
        img_mean=-0.8411,
        img_median=-1,
        img_dtype=np.int32,
        img_shape=(67, 31, 1),
    ),
    VectorToRasterTestCase(
        name="empty vector data test",
        task=VectorToRasterTask(
            vector_input=CUSTOM_DATAFRAME[(CUSTOM_DATAFRAME.LULC_NAME == "some_none_existent_name")],
            raster_feature=RASTER_FEATURE,
            values_column="LULC_ID",
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=0,
        ),
        img_max=0,
        img_mean=0,
        img_median=0,
        img_shape=(101, 100, 1),
    ),
    VectorToRasterTestCase(
        name="negative polygon buffering",
        task=VectorToRasterTask(
            vector_input=CUSTOM_DATAFRAME,
            raster_feature=RASTER_FEATURE,
            values_column="LULC_ID",
            buffer=-2,
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=0,
        ),
        img_max=8,
        img_mean=0.0229,
        img_median=0,
        img_shape=(101, 100, 1),
    ),
    VectorToRasterTestCase(
        name="positive polygon buffering",
        task=VectorToRasterTask(
            vector_input=CUSTOM_DATAFRAME,
            raster_feature=RASTER_FEATURE,
            values_column="LULC_ID",
            buffer=2,
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=0,
        ),
        img_max=8,
        img_mean=0.0664,
        img_median=0,
        img_shape=(101, 100, 1),
    ),
    VectorToRasterTestCase(
        name="different crs",
        task=VectorToRasterTask(
            vector_input=CUSTOM_DATAFRAME.to_crs(epsg=3857),
            raster_feature=RASTER_FEATURE,
            values_column="LULC_ID",
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=0,
        ),
        img_max=8,
        img_mean=0.042079,
        img_median=0,
        img_shape=(101, 100, 1),
    ),
    VectorToRasterTestCase(
        name="3D polygons, np.int8",
        task=VectorToRasterTask(
            vector_input=CUSTOM_DATAFRAME_3D,
            raster_feature=RASTER_FEATURE,
            values_column="LULC_ID",
            raster_shape=(FeatureType.DATA, "BANDS-S2-L1C"),
            no_data_value=-1,
            raster_dtype=np.dtype("int8"),
        ),
        img_min=-1,
        img_max=8,
        img_mean=-0.9461386,
        img_median=-1,
        img_shape=(101, 100, 1),
        img_dtype=np.int8,
    ),
    VectorToRasterTestCase(
        name="bool dtype",
        task=VectorToRasterTask(
            VECTOR_FEATURE,
            RASTER_FEATURE,
            values=[1],
            values_column="LULC_ID",
            raster_shape=(100, 150),
            no_data_value=0,
            raster_dtype=bool,
        ),
        img_min=False,
        img_max=True,
        img_dtype=bool,
        img_shape=(100, 150, 1),
    ),
)


@pytest.mark.parametrize("test_case", VECTOR_TO_RASTER_TEST_CASES)
def test_vector_to_raster_result(test_case, test_eopatch):
    result = test_case.task(test_eopatch)[test_case.task.raster_feature]
    delta = 1e-3

    assert np.amin(result) == pytest.approx(test_case.img_min, abs=delta), "Minimum values do not match."
    assert np.amax(result) == pytest.approx(test_case.img_max, abs=delta), "Maximum values do not match."
    if test_case.img_mean is not None:
        assert np.mean(result) == pytest.approx(test_case.img_mean, abs=delta), "Mean values do not match."
    if test_case.img_median is not None:
        assert np.median(result) == pytest.approx(test_case.img_median, abs=delta), "Median values do not match."
    assert result.dtype == test_case.img_dtype, "Wrong dtype of result."
    assert result.shape == test_case.img_shape, "Result is of wrong shape."


def test_polygon_overlap(test_eopatch):

    # create two test bboxes to overlap existing classes
    bounds = test_eopatch[VECTOR_FEATURE].total_bounds
    test_bounds1 = bounds[0] + 500, bounds[1] + 1000, bounds[2] - 1450, bounds[3] - 1650
    test_bounds2 = bounds[0] + 300, bounds[1] + 1400, bounds[2] - 1750, bounds[3] - 1300

    dframe = test_eopatch[VECTOR_FEATURE][0:50]

    # override 0th row with a test polygon of class 10
    test_row = dframe.index[0]
    dframe.at[test_row, "LULC_ID"] = 10
    dframe.at[test_row, "geometry"] = Polygon.from_bounds(*test_bounds1)

    # override the last row with a test polygon of class 5
    test_row = dframe.index[-1]
    dframe.at[test_row, "LULC_ID"] = 5
    dframe.at[test_row, "geometry"] = Polygon.from_bounds(*test_bounds2)

    test_eopatch.vector_timeless["TEST"] = dframe

    kwargs = {"raster_shape": (FeatureType.DATA, "BANDS-S2-L1C"), "values_column": "LULC_ID"}
    eop = test_eopatch

    # no overlap
    eop = VectorToRasterTask(dframe[1:-1], (FeatureType.MASK_TIMELESS, "OVERLAP_0"), overlap_value=5, **kwargs)(eop)

    # overlap without taking intersection into account
    eop = VectorToRasterTask(dframe, (FeatureType.MASK_TIMELESS, "OVERLAP_1"), overlap_value=None, **kwargs)(eop)

    # overlap by setting intersections to 0
    eop = VectorToRasterTask(dframe, (FeatureType.MASK_TIMELESS, "OVERLAP_2"), overlap_value=0, **kwargs)(eop)

    # overlap by setting intersections to class 7
    eop = VectorToRasterTask(dframe, (FeatureType.MASK_TIMELESS, "OVERLAP_3"), overlap_value=7, **kwargs)(eop)

    # separately render bboxes for comparisons in asserts
    eop = VectorToRasterTask(dframe[:1], (FeatureType.MASK_TIMELESS, "TEST_BBOX1"), **kwargs)(eop)
    eop = VectorToRasterTask(dframe[-1:], (FeatureType.MASK_TIMELESS, "TEST_BBOX2"), **kwargs)(eop)

    bbox1 = eop.mask_timeless["TEST_BBOX1"]
    bbox2 = eop.mask_timeless["TEST_BBOX2"]

    overlap0 = eop.mask_timeless["OVERLAP_0"]
    overlap1 = eop.mask_timeless["OVERLAP_1"]
    overlap2 = eop.mask_timeless["OVERLAP_2"]

    # 4 gets partially covered by 5
    assert np.count_nonzero(overlap0 == 4) > np.count_nonzero(overlap1 == 4)
    # 2 doesn't get covered, stays the same
    assert np.count_nonzero(overlap0 == 2) == np.count_nonzero(overlap1 == 2)
    # 10 is bbox2 and it gets covered by other classes
    assert np.count_nonzero(bbox1) > np.count_nonzero(overlap1 == 10)
    # 5 is bbox1 and it is rendered on top of all others, so it doesn't get covered
    assert np.count_nonzero(bbox2) == np.count_nonzero(overlap1 == 5)

    # all classes have their parts intersected, so the sum should reduce
    assert np.count_nonzero(bbox1) > np.count_nonzero(overlap2 == 10)
    assert np.count_nonzero(bbox2) > np.count_nonzero(overlap2 == 5)
    assert np.count_nonzero(overlap0 == 4) > np.count_nonzero(overlap2 == 4)
    # 2 gets covered completely
    assert np.count_nonzero(overlap2 == 2) == 0


@dataclasses.dataclass(frozen=True)
class RasterToVectorTestCase:
    name: str
    task: EOTask
    feature: Any
    vector_feature: Any
    data_len: int
    test_reverse: bool = False


RASTER_TO_VECTOR_TEST_CASES = (
    RasterToVectorTestCase(
        name="reverse test",
        task=RasterToVectorTask((FeatureType.MASK_TIMELESS, "LULC", "NEW_LULC")),
        feature=(FeatureType.MASK_TIMELESS, "LULC"),
        vector_feature=(FeatureType.VECTOR_TIMELESS, "NEW_LULC"),
        data_len=126,
        test_reverse=True,
    ),
    RasterToVectorTestCase(
        name="parameters test",
        task=RasterToVectorTask(
            (FeatureType.MASK, "CLM"), values=[1, 2], values_column="IS_CLOUD", raster_dtype=np.int16, connectivity=8
        ),
        feature=(FeatureType.MASK, "CLM"),
        vector_feature=(FeatureType.VECTOR, "CLM"),
        data_len=54,
    ),
)


@pytest.mark.parametrize("test_case", RASTER_TO_VECTOR_TEST_CASES)
def test_raster_to_vector_result(test_case, test_eopatch):
    eop_vectorized = test_case.task(test_eopatch)
    assert test_case.data_len == len(eop_vectorized[test_case.vector_feature].index), "Got wrong number of shapes."

    if test_case.test_reverse:
        old_feature = test_case.feature
        new_feature = old_feature[0], f"{old_feature[1]}_NEW"

        vector2raster_task = VectorToRasterTask(
            test_case.vector_feature, new_feature, values_column=test_case.task.values_column, raster_shape=old_feature
        )

        eop_rerasterized = vector2raster_task(eop_vectorized)
        assert_array_equal(
            eop_rerasterized[new_feature],
            eop_rerasterized[old_feature],
            err_msg="Old and new raster features should be the same",
        )
