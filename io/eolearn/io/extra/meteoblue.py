"""
Module with tasks that provide data from meteoblue services

To use tasks from this module you have to install METEOBLUE package extension:

.. code-block::

    pip install eo-learn-io[METEOBLUE]

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import datetime as dt
from abc import ABCMeta, abstractmethod
from typing import Any

import dateutil.parser
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry

try:
    import meteoblue_dataset_sdk
    from meteoblue_dataset_sdk.caching import FileCache
except ImportError as exception:
    raise ImportError("This module requires an installation of meteoblue_dataset_sdk package") from exception


from sentinelhub import CRS, BBox, Geometry, parse_time_interval, serialize_time
from sentinelhub.types import RawTimeIntervalType

from eolearn.core import EOPatch, EOTask
from eolearn.core.constants import TIMESTAMP_COLUMN, FeatureType


class BaseMeteoblueTask(EOTask, metaclass=ABCMeta):
    """A base task implementing the logic that is common for all meteoblue tasks"""

    def __init__(
        self,
        feature: tuple[FeatureType, str],
        apikey: str,
        query: dict | None = None,
        units: dict | None = None,
        time_difference: dt.timedelta = dt.timedelta(minutes=30),  # noqa: B008, RUF100
        cache_folder: str | None = None,
        cache_max_age: int = 604800,
    ):
        """
        :param feature: A feature in which meteoblue data will be stored
        :param apikey: meteoblue API key
        :param query: meteoblue dataset API query definition. If set to None (default) the query has to be set
            in the execute method instead.
        :param units: meteoblue dataset API units definition. If set to None (default) request will use default units
            as specified in https://docs.meteoblue.com/en/weather-apis/dataset-api/dataset-api#units
        :param time_difference: The size of a time interval around each timestamp for which data will be collected. It
            is used only in a combination with ``time_interval`` parameter from ``execute`` method.
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :param cache_max_age: Maximum age in seconds to use a cached result. Default 1 week.
        """
        self.feature = self.parse_feature(feature)
        cache = None
        if cache_folder:
            cache = FileCache(path=cache_folder, max_age=cache_max_age)

        self.client = meteoblue_dataset_sdk.Client(apikey=apikey, cache=cache)
        self.query = query
        self.units = units
        self.time_difference = time_difference

    @staticmethod
    def _get_modified_eopatch(eopatch: EOPatch | None, bbox: BBox | None) -> tuple[EOPatch, BBox]:
        if bbox is not None:
            if eopatch is None:
                eopatch = EOPatch(bbox=bbox)
            elif eopatch.bbox is None:
                eopatch.bbox = bbox
            elif eopatch.bbox != bbox:
                raise ValueError("Provided eopatch.bbox and bbox are not the same")
            return eopatch, bbox

        if eopatch is None or eopatch.bbox is None:
            raise ValueError("Bounding box is not provided")
        return eopatch, eopatch.bbox

    def _prepare_time_intervals(self, eopatch: EOPatch, time_interval: RawTimeIntervalType | None) -> list[str]:
        """Prepare a list of time intervals for which data will be collected from meteoblue services"""
        if not eopatch.timestamps and not time_interval:
            raise ValueError(
                "Time interval should either be defined with `eopatch.timestamps` or `time_interval` parameter"
            )

        if time_interval:
            serialized_start_time, serialized_end_time = serialize_time(parse_time_interval(time_interval))
            return [f"{serialized_start_time}/{serialized_end_time}"]

        timestamps = eopatch.timestamps
        time_intervals: list[str] = []
        for timestamp in timestamps:
            start_time = timestamp - self.time_difference
            end_time = timestamp + self.time_difference

            serizalized_start_time, serizalized_end_time = serialize_time((start_time, end_time))

            time_intervals.append(f"{serizalized_start_time}/{serizalized_end_time}")

        return time_intervals

    @abstractmethod
    def _get_data(self, query: dict) -> tuple[Any, list[dt.datetime]]:
        """It should return an output feature object and a list of timestamps"""

    def execute(
        self,
        eopatch: EOPatch | None = None,
        *,
        query: dict | None = None,
        bbox: BBox | None = None,
        time_interval: RawTimeIntervalType | None = None,
    ) -> EOPatch:
        """Execute method that adds new meteoblue data into an EOPatch

        :param eopatch: An EOPatch in which data will be added. If not provided a new EOPatch will be created.
        :param bbox: A bounding box of a request. Should be provided if eopatch parameter is not provided.
        :param query: meteoblue dataset API query definition. This query takes precedence over one defined in __init__.
        :param time_interval: An interval for which data should be downloaded. If not provided then timestamps from
            provided eopatch will be used.
        :raises ValueError: Raises an exception when no query is set during Task initialization or the execute method.
        """
        eopatch, bbox = self._get_modified_eopatch(eopatch, bbox)

        time_intervals = self._prepare_time_intervals(eopatch, time_interval)

        geometry = Geometry(bbox.geometry, bbox.crs).transform(CRS.WGS84)
        geojson = shapely.geometry.mapping(geometry.geometry)

        query = query if query is not None else self.query
        if query is None:
            raise ValueError("Query has to specified in execute method or during task initialization")

        executable_query = {
            "units": self.units,
            "geometry": geojson,
            "format": "protobuf",
            "timeIntervals": time_intervals,
            "queries": [query],
        }
        result_data, result_timestamps = self._get_data(executable_query)

        if not eopatch.timestamps and result_timestamps:
            eopatch.timestamps = result_timestamps

        eopatch[self.feature] = result_data
        return eopatch


class MeteoblueVectorTask(BaseMeteoblueTask):
    """Obtains weather data from meteoblue services as a vector feature

    The data is obtained as a VECTOR feature in a ``geopandas.GeoDataFrame`` where columns include latitude, longitude,
    timestamp and a column for each weather variable. All data is downloaded from the
    meteoblue dataset API (<https://docs.meteoblue.com/en/weather-apis/dataset-api/dataset-api>).

    A meteoblue API key is required to retrieve data.
    """

    def _get_data(self, query: dict) -> tuple[gpd.GeoDataFrame, list[dt.datetime]]:
        """Provides a GeoDataFrame with information about weather control points and an empty list of timestamps"""
        result = self.client.querySync(query)
        dataframe = meteoblue_to_dataframe(result)
        geometry = gpd.points_from_xy(dataframe.Longitude, dataframe.Latitude)
        crs = CRS.WGS84.pyproj_crs()
        gdf = gpd.GeoDataFrame(dataframe, geometry=geometry, crs=crs)
        return gdf, []


class MeteoblueRasterTask(BaseMeteoblueTask):
    """Obtains weather data from meteoblue services as a raster feature

    It returns a 4D numpy array with dimensions (time, height, width, weather variables) which should be stored as a
    DATA feature. Data is resampled to WGS84 plate carrée to a specified resolution using the
    meteoblue dataset API (<https://docs.meteoblue.com/en/weather-apis/dataset-api/dataset-api>).

    A meteoblue API key is required to retrieve data.
    """

    def _get_data(self, query: dict) -> tuple[np.ndarray, list[dt.datetime]]:
        """Return a 4-dimensional numpy array of shape (time, height, width, weather variables) and a list of
        timestamps
        """
        result = self.client.querySync(query)

        data = meteoblue_to_numpy(result)
        timestamps = _meteoblue_timestamps_from_geometry(result.geometries[0])
        return data, timestamps


def meteoblue_to_dataframe(result: Any) -> pd.DataFrame:
    """Transform a meteoblue dataset API result to a dataframe

    :param result: A response of meteoblue API of type `Dataset_pb2.DatasetApiProtobuf`
    :returns: A dataframe with columns TIMESTAMP, Longitude, Latitude and aggregation columns
    """
    geometry = result.geometries[0]
    code_names = [f"{code.code}_{code.level}_{code.aggregation}" for code in geometry.codes]

    if not geometry.timeIntervals:
        return pd.DataFrame(columns=[TIMESTAMP_COLUMN, "Longitude", "Latitude", *code_names])

    dataframes = []
    for index, time_interval in enumerate(geometry.timeIntervals):
        timestamps = _meteoblue_timestamps_from_time_interval(time_interval)

        n_locations = len(geometry.lats)
        n_timesteps = len(timestamps)

        dataframe = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: np.tile(timestamps, n_locations),  # type: ignore[arg-type] # numpy can do this
                "Longitude": np.repeat(geometry.lons, n_timesteps),
                "Latitude": np.repeat(geometry.lats, n_timesteps),
            }
        )

        for code, code_name in zip(geometry.codes, code_names):
            dataframe[code_name] = np.array(code.timeIntervals[index].data)

        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


def meteoblue_to_numpy(result: Any) -> np.ndarray:
    """Transform a meteoblue dataset API result to a dataframe

    :param result: A response of meteoblue API of type `Dataset_pb2.DatasetApiProtobuf`
    :returns: A 4D numpy array with shape (time, height, width, weather variables)
    """
    geometry = result.geometries[0]

    n_locations = len(geometry.lats)
    n_codes = len(geometry.codes)
    n_time_intervals = len(geometry.timeIntervals)
    geo_ny = geometry.ny
    geo_nx = geometry.nx

    # meteoblue data is using dimensions (n_variables, n_time_intervals, ny, nx, n_timesteps)
    # Individual time intervals may have different number of timesteps (not a dimension)
    # Therefore we have to first transpose each code individually and then transpose everything again
    def map_code(code: Any) -> np.ndarray:
        """Transpose a single code"""
        code_data = np.array([t.data for t in code.timeIntervals])

        code_n_timesteps = code_data.size // n_locations // n_time_intervals
        code_data = code_data.reshape((n_time_intervals, geo_ny, geo_nx, code_n_timesteps))

        # transpose from shape (n_time_intervals, geo_ny, geo_nx, code_n_timesteps)
        # to (n_time_intervals, code_n_timesteps, geo_ny, geo_nx)
        # and flip the y-axis (meteoblue is using northward facing axis
        # but the standard for EOPatch features is a southward facing axis)
        return np.flip(code_data.transpose((0, 3, 1, 2)), axis=2)

    data = np.array(list(map(map_code, geometry.codes)))

    n_timesteps = data.size // n_locations // n_codes
    data = data.reshape((n_codes, n_timesteps, geo_ny, geo_nx))

    return data.transpose((1, 2, 3, 0))


def _meteoblue_timestamps_from_geometry(geometry_pb: Any) -> list[dt.datetime]:
    """Transforms a protobuf geometry object into a list of datetime objects"""
    return list(pd.core.common.flatten(map(_meteoblue_timestamps_from_time_interval, geometry_pb.timeIntervals)))


def _meteoblue_timestamps_from_time_interval(timestamp_pb: Any) -> list[dt.datetime]:
    """Transforms a protobuf timestamp object into a list of datetime objects"""
    if timestamp_pb.timestrings:
        # Time intervals like weekly data, return an `array of strings` as timestamps
        # For time indications like `20200801T0000-20200802T235959` we only return the first date as datetime
        return list(map(_parse_timestring, timestamp_pb.timestrings))

    # Regular time intervals return `start, end and stride` as a time axis
    # We convert it into an array of daytime
    time_range = range(timestamp_pb.start, timestamp_pb.end, timestamp_pb.stride)
    return list(map(dt.datetime.fromtimestamp, time_range))


def _parse_timestring(timestring: str) -> dt.datetime:
    """A helper method to parse specific timestrings obtained from meteoblue service"""
    if "-" in timestring:
        timestring = timestring.split("-")[0]
    return dateutil.parser.parse(timestring)
