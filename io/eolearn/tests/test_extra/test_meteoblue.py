"""
Module containing tests for Meteoblue tasks

Credits:
Copyright (c) 2021-2022 Patrick Zippenfenig (Meteoblue), Matej Aleksandrov (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime as dt
import os

import numpy as np
from meteoblue_dataset_sdk.protobuf.dataset_pb2 import DatasetApiProtobuf

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureType
from eolearn.io.extra.meteoblue import MeteoblueRasterTask, MeteoblueVectorTask

RASTER_QUERY = {
    "domain": "NEMS4",
    "gapFillDomain": None,
    "timeResolution": "hourly",
    "codes": [{"code": 11, "level": "2 m above gnd"}],
    "transformations": [
        {"type": "aggregateTimeInterval", "aggregation": "mean"},
        {
            "type": "spatialTransformV2",
            "gridResolution": 0.02,
            "interpolationMethod": "linear",
            "spatialAggregation": "mean",
            "disjointArea": "keep",
        },
    ],
}

VECTOR_QUERY = {
    "domain": "NEMS4",
    "gapFillDomain": None,
    "timeResolution": "daily",
    "codes": [{"code": 11, "level": "2 m above gnd", "aggregation": "mean"}],
    "transformations": None,
}

UNITS = {
    "temperature": "C",
    "velocity": "km/h",
    "length": "metric",
    "energy": "watts",
}

BBOX = BBox([7.52, 47.50, 7.7, 47.6], crs=CRS.WGS84)
TIME_INTERVAL = dt.datetime(year=2020, month=8, day=1), dt.datetime(year=2020, month=8, day=3)


def test_meteoblue_raster_task(mocker):
    """Unit test for MeteoblueRasterTask"""
    mocker.patch(
        "meteoblue_dataset_sdk.Client.querySync",
        return_value=_load_meteoblue_client_response("test_meteoblue_raster_input.bin"),
    )

    feature = FeatureType.DATA, "WEATHER-DATA"
    meteoblue_task = MeteoblueRasterTask(feature, "dummy-api-key", query=RASTER_QUERY, units=UNITS)

    eopatch = meteoblue_task.execute(bbox=BBOX, time_interval=TIME_INTERVAL)

    assert eopatch.bbox == BBOX
    assert eopatch.timestamp == [dt.datetime(2020, 8, 1)]

    data = eopatch[feature]
    assert data.shape == (1, 6, 10, 1)
    assert data.dtype == np.float64

    assert round(np.mean(data), 5) == 23.79214
    assert round(np.std(data), 5) == 0.3996
    assert round(data[0, 0, 0, 0], 5) == 23.74646


def test_meteoblue_vector_task(mocker):
    """Unit test for MeteoblueVectorTask"""
    mocker.patch(
        "meteoblue_dataset_sdk.Client.querySync",
        return_value=_load_meteoblue_client_response("test_meteoblue_vector_input.bin"),
    )

    feature = FeatureType.VECTOR, "WEATHER-DATA"
    meteoblue_task = MeteoblueVectorTask(feature, "dummy-api-key", query=VECTOR_QUERY, units=UNITS)

    eopatch = EOPatch(bbox=BBOX)
    eopatch = meteoblue_task.execute(eopatch, time_interval=TIME_INTERVAL)

    assert eopatch.bbox == BBOX

    data = eopatch[feature]
    assert len(data.index) == 18
    assert data.crs.to_epsg() == 4326

    data_series = data["11_2 m above gnd_mean"]
    assert round(data_series.mean(), 5) == 23.75278
    assert round(data_series.std(), 5) == 2.99785


def _load_meteoblue_client_response(filename):
    """Loads locally stored responses of Meteoblue client

    To update content of saved files use:
    with open('<path>', 'wb') as fp:
        fp.write(result.SerializeToString())
    """
    path = os.path.join(os.path.dirname(__file__), "..", "TestInputs", filename)

    with open(path, "rb") as fp:
        return DatasetApiProtobuf.FromString(fp.read())
