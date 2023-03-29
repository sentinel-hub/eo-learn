""" Input tasks that collect data from `Sentinel-Hub Process API
<https://docs.sentinel-hub.com/api/latest/api/process/>`__

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import datetime as dt
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from typing_extensions import Literal

from sentinelhub import (
    BBox,
    DataCollection,
    Geometry,
    MimeType,
    MosaickingOrder,
    ResamplingType,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SentinelHubSession,
    SHConfig,
    bbox_to_dimensions,
    filter_times,
    parse_time_interval,
)
from sentinelhub.evalscript import generate_evalscript
from sentinelhub.types import JsonDict, RawTimeIntervalType

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import FeatureRenameSpec, FeatureSpec, FeaturesSpecification
from eolearn.core.utils.parsing import parse_renamed_features

LOGGER = logging.getLogger(__name__)


class SentinelHubInputBaseTask(EOTask, metaclass=ABCMeta):
    """Base class for Processing API input tasks"""

    def __init__(
        self,
        data_collection: DataCollection,
        size: Optional[Tuple[int, int]] = None,
        resolution: Optional[Union[float, Tuple[float, float]]] = None,
        cache_folder: Optional[str] = None,
        config: Optional[SHConfig] = None,
        max_threads: Optional[int] = None,
        upsampling: Optional[ResamplingType] = None,
        downsampling: Optional[ResamplingType] = None,
        session_loader: Optional[Callable[[], SentinelHubSession]] = None,
    ):
        """
        :param data_collection: A collection of requested satellite data.
        :param size: Number of pixels in x and y dimension.
        :param resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :param config: An instance of SHConfig defining the service
        :param max_threads: Maximum threads to be used when downloading data.
        :param upsampling: A type of upsampling to apply on data
        :param downsampling: A type of downsampling to apply on data
        :param session_loader: A callable that returns a valid SentinelHubSession, used for session sharing.
            Creates a new session if set to `None`, which should be avoided in large scale parallelization.
        """
        if (size is None) == (resolution is None):
            raise ValueError("Exactly one of the parameters 'size' and 'resolution' should be given.")

        self.size = size
        self.resolution = resolution
        self.config = config or SHConfig()
        self.max_threads = max_threads
        self.data_collection: DataCollection = DataCollection(data_collection)
        self.cache_folder = cache_folder
        self.session_loader = session_loader
        self.upsampling = upsampling
        self.downsampling = downsampling

    def execute(
        self,
        eopatch: Optional[EOPatch] = None,
        bbox: Optional[BBox] = None,
        time_interval: Optional[RawTimeIntervalType] = None,  # should be kept at this to prevent code-breaks
        geometry: Optional[Geometry] = None,
    ) -> EOPatch:
        """Main execute method for the Process API tasks.
        The `geometry` is used only in conjunction with the `bbox` and does not act as a replacement."""

        eopatch_bbox = eopatch.bbox if eopatch is not None else None
        area_bbox = self._consolidate_bbox(bbox, eopatch_bbox)

        eopatch = eopatch or EOPatch(bbox=area_bbox)
        eopatch.bbox = area_bbox
        size_x, size_y = self._get_size(area_bbox)

        if time_interval:
            time_interval = parse_time_interval(time_interval)
            timestamps = self._get_timestamps(time_interval, area_bbox)
            timestamps = [time_point.replace(tzinfo=None) for time_point in timestamps]
        elif self.data_collection.is_timeless:
            timestamps = None  # should this be [] to match next branch in case of a fresh eopatch?
        else:
            timestamps = eopatch.timestamps

        if timestamps is not None:
            if not eopatch.timestamps:
                eopatch.timestamps = timestamps
            elif timestamps != eopatch.timestamps:
                raise ValueError("Trying to write data to an existing EOPatch with a different timestamp.")

        sh_requests = self._build_requests(area_bbox, size_x, size_y, timestamps, time_interval, geometry)
        requests = [request.download_list[0] for request in sh_requests]

        LOGGER.debug("Downloading %d requests of type %s", len(requests), str(self.data_collection))
        session = None if self.session_loader is None else self.session_loader()
        client = SentinelHubDownloadClient(config=self.config, session=session)
        responses = client.download(requests, max_threads=self.max_threads)
        LOGGER.debug("Downloads complete")

        temporal_dim = 1 if timestamps is None else len(timestamps)
        shape = temporal_dim, size_y, size_x
        self._extract_data(eopatch, responses, shape)

        return eopatch

    def _get_size(self, bbox: BBox) -> Tuple[int, int]:
        """Get the size (width, height) for the request either from inputs, or from the (existing) eopatch"""
        if self.size is not None:
            return self.size

        if self.resolution is not None:
            return bbox_to_dimensions(bbox, self.resolution)

        raise ValueError("Size or resolution for the requests should be provided!")

    @staticmethod
    def _consolidate_bbox(bbox: Optional[BBox], eopatch_bbox: Optional[BBox]) -> BBox:
        if eopatch_bbox is None:
            if bbox is None:
                raise ValueError("Either the eopatch or the task must provide valid bbox.")
            return bbox

        if bbox is None or eopatch_bbox == bbox:
            return eopatch_bbox
        raise ValueError("Either the eopatch or the task must provide bbox, or they must be the same.")

    @abstractmethod
    def _extract_data(self, eopatch: EOPatch, responses: List[Any], shape: Tuple[int, ...]) -> EOPatch:
        """Extract data from the received images and assign them to eopatch features"""

    @abstractmethod
    def _build_requests(
        self,
        bbox: Optional[BBox],
        size_x: int,
        size_y: int,
        timestamps: Optional[List[dt.datetime]],
        time_interval: Optional[RawTimeIntervalType],
        geometry: Optional[Geometry],
    ) -> List[SentinelHubRequest]:
        """Build requests"""

    @abstractmethod
    def _get_timestamps(self, time_interval: Optional[RawTimeIntervalType], bbox: BBox) -> List[dt.datetime]:
        """Get the timestamp array needed as a parameter for downloading the images"""


class SentinelHubEvalscriptTask(SentinelHubInputBaseTask):
    """Process API task to download data using evalscript"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        features: FeaturesSpecification,
        evalscript: str,
        data_collection: DataCollection,
        size: Optional[Tuple[int, int]] = None,
        resolution: Optional[Union[float, Tuple[float, float]]] = None,
        maxcc: Optional[float] = None,
        time_difference: Optional[dt.timedelta] = None,
        mosaicking_order: Optional[Union[str, MosaickingOrder]] = None,
        cache_folder: Optional[str] = None,
        config: Optional[SHConfig] = None,
        max_threads: Optional[int] = None,
        upsampling: Optional[ResamplingType] = None,
        downsampling: Optional[ResamplingType] = None,
        aux_request_args: Optional[dict] = None,
        session_loader: Optional[Callable[[], SentinelHubSession]] = None,
        timestamp_filter: Callable[[List[dt.datetime], dt.timedelta], List[dt.datetime]] = filter_times,
    ):
        """
        :param features: Features to construct from the evalscript.
        :param evalscript: Evalscript for the request. Beware that all outputs from SentinelHub services should be named
            and should have the same name as corresponding feature
        :param data_collection: Source of requested satellite data.
        :param size: Number of pixels in x and y dimension.
        :param resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :param maxcc: Maximum cloud coverage, a float in interval [0, 1]
        :param time_difference: Minimum allowed time difference, used when filtering dates. Also used by the service
            for mosaicking, timestamps might be misleading for large values.
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :param config: An instance of SHConfig defining the service
        :param max_threads: Maximum threads to be used when downloading data.
        :param upsampling: A type of upsampling to apply on data
        :param downsampling: A type of downsampling to apply on data
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :param aux_request_args: a dictionary with auxiliary information for the input_data part of the SH request
        :param session_loader: A callable that returns a valid SentinelHubSession, used for session sharing.
            Creates a new session if set to `None`, which should be avoided in large scale parallelization.
        :param timestamp_filter: A function that performs the final filtering of timestamps, usually to remove multiple
            occurrences within the time_difference window. Check `get_available_timestamps` for more info.
        """
        super().__init__(
            data_collection=data_collection,
            size=size,
            resolution=resolution,
            cache_folder=cache_folder,
            config=config,
            max_threads=max_threads,
            upsampling=upsampling,
            downsampling=downsampling,
            session_loader=session_loader,
        )

        self.features = self._parse_and_validate_features(features)
        self.responses = self._create_response_objects()
        self.evalscript = evalscript

        if maxcc and isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError("maxcc should be a float on an interval [0, 1]")

        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.timestamp_filter = timestamp_filter
        self.mosaicking_order = None if mosaicking_order is None else MosaickingOrder(mosaicking_order)
        self.aux_request_args = aux_request_args

    def _parse_and_validate_features(self, features: FeaturesSpecification) -> List[FeatureRenameSpec]:
        _features = parse_renamed_features(
            features, allowed_feature_types=lambda fty: fty.is_array() or fty == FeatureType.META_INFO
        )

        ftr_data_types = {ft for ft, _, _ in _features if not ft.is_meta()}
        if all(ft.is_timeless() for ft in ftr_data_types) or all(ft.is_temporal() for ft in ftr_data_types):
            return _features

        raise ValueError("Cannot mix time dependent and timeless requests!")

    def _create_response_objects(self) -> List[JsonDict]:
        """Construct SentinelHubRequest output_responses from features"""
        responses = []
        for feat_type, feat_name, _ in self.features:
            if feat_type.is_array():
                feat_name = cast(str, feat_name)  # can only be string since it's an array type
                responses.append(SentinelHubRequest.output_response(feat_name, MimeType.TIFF))
            elif feat_type.is_meta():
                responses.append(SentinelHubRequest.output_response("userdata", MimeType.JSON))
            else:
                # should not happen as features have already been validated
                raise ValueError(f"{feat_type} not supported!")

        return responses

    def _get_timestamps(self, time_interval: Optional[RawTimeIntervalType], bbox: BBox) -> List[dt.datetime]:
        """Get the timestamp array needed as a parameter for downloading the images"""
        if any(feat_type.is_timeless() for feat_type, _, _ in self.features if feat_type.is_array()):
            return []

        return get_available_timestamps(
            bbox=bbox,
            time_interval=time_interval,
            timestamp_filter=self.timestamp_filter,
            data_collection=self.data_collection,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            config=self.config,
        )

    def _build_requests(
        self,
        bbox: Optional[BBox],
        size_x: int,
        size_y: int,
        timestamps: Optional[List[dt.datetime]],
        time_interval: Optional[RawTimeIntervalType],
        geometry: Optional[Geometry],
    ) -> List[SentinelHubRequest]:
        """Defines request timestamps and builds requests. In case `timestamps` is either `None` or an empty list it
        still has to create at least one request in order to obtain back number of bands of responses."""
        dates: List[Optional[Tuple[Optional[dt.datetime], Optional[dt.datetime]]]]
        if timestamps:
            dates = [(date - self.time_difference, date + self.time_difference) for date in timestamps]
        elif timestamps is None:
            dates = [None]
        else:
            dates = [parse_time_interval(time_interval, allow_undefined=True)]

        return [self._create_sh_request(date, bbox, size_x, size_y, geometry) for date in dates]

    def _create_sh_request(
        self,
        time_interval: Optional[RawTimeIntervalType],
        bbox: Optional[BBox],
        size_x: int,
        size_y: int,
        geometry: Optional[Geometry],
    ) -> SentinelHubRequest:
        """Create an instance of SentinelHubRequest"""
        return SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    mosaicking_order=self.mosaicking_order,
                    time_interval=time_interval,
                    maxcc=self.maxcc,
                    upsampling=self.upsampling,
                    downsampling=self.downsampling,
                    other_args=self.aux_request_args,
                )
            ],
            responses=self.responses,
            bbox=bbox,
            geometry=geometry,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config,
        )

    def _extract_data(self, eopatch: EOPatch, responses: List[Any], shape: Tuple[int, ...]) -> EOPatch:
        """Extract data from the received images and assign them to eopatch features"""
        # pylint: disable=arguments-renamed
        if len(self.features) == 1:
            ftype, fname, _ = self.features[0]
            extension = "json" if ftype.is_meta() else "tif"
            responses = [{f"{fname}.{extension}": data} for data in responses]

        for ftype, fname, new_fname in self.features:
            if ftype.is_meta():
                eopatch[ftype][new_fname] = [data["userdata.json"] for data in responses]

            elif ftype.is_temporal():
                data = np.asarray([data[f"{fname}.tif"] for data in responses])
                data = data[..., np.newaxis] if data.ndim == 3 else data
                time_dim = shape[0]
                eopatch[ftype][new_fname] = data[:time_dim] if time_dim != data.shape[0] else data

            else:
                eopatch[ftype][new_fname] = np.asarray(responses[0][f"{fname}.tif"])[..., np.newaxis]

        return eopatch


class SentinelHubInputTask(SentinelHubInputBaseTask):
    """Process API input task that loads 16bit integer data and converts it to a 32bit float feature."""

    # pylint: disable=too-many-locals
    def __init__(
        self,
        data_collection: DataCollection,
        size: Optional[Tuple[int, int]] = None,
        resolution: Optional[Union[float, Tuple[float, float]]] = None,
        bands_feature: Optional[Tuple[FeatureType, str]] = None,
        bands: Optional[List[str]] = None,
        additional_data: Optional[List[Tuple[FeatureType, str]]] = None,
        evalscript: Optional[str] = None,
        maxcc: Optional[float] = None,
        time_difference: Optional[dt.timedelta] = None,
        cache_folder: Optional[str] = None,
        config: Optional[SHConfig] = None,
        max_threads: Optional[int] = None,
        bands_dtype: Union[None, np.dtype, type] = None,
        single_scene: bool = False,
        mosaicking_order: Optional[Union[str, MosaickingOrder]] = None,
        upsampling: Optional[ResamplingType] = None,
        downsampling: Optional[ResamplingType] = None,
        aux_request_args: Optional[dict] = None,
        session_loader: Optional[Callable[[], SentinelHubSession]] = None,
        timestamp_filter: Callable[[List[dt.datetime], dt.timedelta], List[dt.datetime]] = filter_times,
    ):
        """
        :param data_collection: Source of requested satellite data.
        :param size: Number of pixels in x and y dimension.
        :param resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :param bands_feature: A target feature into which to save the downloaded images.
        :param bands: An array of band names. If not specified it will download all bands specified for a given data
            collection.
        :param additional_data: A list of additional data to be downloaded, such as SCL, SNW, dataMask, etc.
        :param evalscript: An optional parameter to override an evalscript that is generated by default
        :param maxcc: Maximum cloud coverage.
        :param time_difference: Minimum allowed time difference, used when filtering dates. Also used by the service
            for mosaicking, timestamps might be misleading for large values.
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :param config: An instance of SHConfig defining the service
        :param max_threads: Maximum threads to be used when downloading data.
        :param bands_dtype: output type of the bands array, if set to None the default is used
        :param single_scene: If true, the service will compute a single image for the given time interval using
            mosaicking.
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :param upsampling: A type of upsampling to apply on data
        :param downsampling: A type of downsampling to apply on data
        :param aux_request_args: a dictionary with auxiliary information for the input_data part of the SH request
        :param session_loader: A callable that returns a valid SentinelHubSession, used for session sharing.
            Creates a new session if set to `None`, which should be avoided in large scale parallelization.
        :param timestamp_filter: A callable that performs the final filtering of timestamps, usually to remove multiple
            occurrences within the time_difference window. Check `get_available_timestamps` for more info.
        """
        super().__init__(
            data_collection=data_collection,
            size=size,
            resolution=resolution,
            cache_folder=cache_folder,
            config=config,
            max_threads=max_threads,
            upsampling=upsampling,
            downsampling=downsampling,
            session_loader=session_loader,
        )
        self.evalscript = evalscript
        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.timestamp_filter = timestamp_filter
        self.single_scene = single_scene
        self.bands_dtype = bands_dtype
        self.mosaicking_order = None if mosaicking_order is None else MosaickingOrder(mosaicking_order)
        self.aux_request_args = aux_request_args

        self.bands_feature = None
        self.requested_bands = []
        if bands_feature:
            self.bands_feature = self.parse_feature(bands_feature, allowed_feature_types=[FeatureType.DATA])
            if bands is not None:
                self.requested_bands = self._parse_requested_bands(bands, self.data_collection.bands)
            else:
                self.requested_bands = list(self.data_collection.bands)

        self.requested_additional_bands = []
        self.additional_data: Optional[List[FeatureRenameSpec]] = None
        if additional_data is not None:
            parsed_additional_data = parse_renamed_features(additional_data)  # parser gives too general type
            additional_bands = cast(List[str], [band for _, band, _ in parsed_additional_data])
            parsed_bands = self._parse_requested_bands(additional_bands, self.data_collection.metabands)
            self.requested_additional_bands = parsed_bands
            self.additional_data = parsed_additional_data

    def _generate_evalscript(self) -> str:
        """Generate the evalscript to be passed with the request, based on chosen bands"""
        return generate_evalscript(
            data_collection=self.data_collection,
            bands=[band.name for band in self.requested_bands],
            meta_bands=[band.name for band in self.requested_additional_bands],
            use_dn=not np.issubdtype(self.bands_dtype, np.floating),
        )

    def _get_timestamps(self, time_interval: Optional[RawTimeIntervalType], bbox: BBox) -> List[dt.datetime]:
        """Get the timestamp array needed as a parameter for downloading the images"""
        if self.single_scene:
            return [time_interval[0]]  # type: ignore[index, list-item]

        return get_available_timestamps(
            bbox=bbox,
            time_interval=time_interval,
            timestamp_filter=self.timestamp_filter,
            data_collection=self.data_collection,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            config=self.config,
        )

    def _build_requests(
        self,
        bbox: Optional[BBox],
        size_x: int,
        size_y: int,
        timestamps: Optional[List[dt.datetime]],
        time_interval: Optional[RawTimeIntervalType],
        geometry: Optional[Geometry],
    ) -> List[SentinelHubRequest]:
        """Build requests"""
        if timestamps is None:
            intervals: List[Optional[RawTimeIntervalType]] = [None]
        elif self.single_scene:
            intervals = [parse_time_interval(time_interval)]
        else:
            intervals = [(date - self.time_difference, date + self.time_difference) for date in timestamps]

        return [self._create_sh_request(time_interval, bbox, size_x, size_y, geometry) for time_interval in intervals]

    def _create_sh_request(
        self,
        time_interval: Optional[RawTimeIntervalType],
        bbox: Optional[BBox],
        size_x: int,
        size_y: int,
        geometry: Optional[Geometry],
    ) -> SentinelHubRequest:
        """Create an instance of SentinelHubRequest"""
        responses = [
            SentinelHubRequest.output_response(band.name, MimeType.TIFF)
            for band in self.requested_bands + self.requested_additional_bands
        ]
        evalscript = generate_evalscript(
            data_collection=self.data_collection,
            bands=[band.name for band in self.requested_bands],
            meta_bands=[band.name for band in self.requested_additional_bands],
            use_dn=not np.issubdtype(self.bands_dtype, np.floating),
        )

        return SentinelHubRequest(
            evalscript=self.evalscript or evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    time_interval=time_interval,
                    mosaicking_order=self.mosaicking_order,
                    maxcc=self.maxcc,
                    upsampling=self.upsampling,
                    downsampling=self.downsampling,
                    other_args=self.aux_request_args,
                )
            ],
            responses=responses,
            bbox=bbox,
            geometry=geometry,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config,
        )

    def _extract_data(self, eopatch: EOPatch, responses: List[Any], shape: Tuple[int, ...]) -> EOPatch:
        """Extract data from the received images and assign them to eopatch features"""
        if len(self.requested_bands) + len(self.requested_additional_bands) == 1:
            # if only one band is requested the response is not a tar so we reshape it
            only_band = (self.requested_bands + self.requested_additional_bands)[0]
            responses = [{only_band.name + ".tif": image} for image in responses]

        if self.additional_data:
            self._extract_additional_features(eopatch, responses, shape)

        if self.bands_feature:
            self._extract_bands_feature(eopatch, responses, shape)

        return eopatch

    def _extract_additional_features(
        self, eopatch: EOPatch, images: Iterable[np.ndarray], shape: Tuple[int, ...]
    ) -> None:
        """Extracts additional features from response into an EOPatch"""
        additional_data = cast(List[FeatureRenameSpec], self.additional_data)  # verified by `if` in _extract_data
        for (ftype, _, new_name), band_info in zip(additional_data, self.requested_additional_bands):
            tiffs = [tar[band_info.name + ".tif"] for tar in images]
            eopatch[ftype, new_name] = self._extract_array(tiffs, 0, shape, band_info.output_types[0])

    def _extract_bands_feature(self, eopatch: EOPatch, images: Iterable[np.ndarray], shape: Tuple[int, ...]) -> None:
        """Extract the bands feature arrays and concatenate them along the last axis"""
        processed_bands = []
        for band_info in self.requested_bands:
            tiffs = [tar[band_info.name + ".tif"] for tar in images]
            dtype = self.bands_dtype or band_info.output_types[0]
            processed_bands.append(self._extract_array(tiffs, 0, shape, dtype))

        bands_feature = cast(Tuple[FeatureType, str], self.bands_feature)  # verified by `if` in _extract_data
        eopatch[bands_feature] = np.concatenate(processed_bands, axis=-1)

    @staticmethod
    def _extract_array(
        tiffs: List[np.ndarray], idx: int, shape: Tuple[int, ...], dtype: Union[type, np.dtype]
    ) -> np.ndarray:
        """Extract a numpy array from the received tiffs"""
        feature_arrays = (np.atleast_3d(img)[..., idx] for img in tiffs)
        return np.asarray(list(feature_arrays), dtype=dtype).reshape(*shape, 1)


class SentinelHubDemTask(SentinelHubEvalscriptTask):
    """
    Adds DEM data (one of the `collections <https://docs.sentinel-hub.com/api/latest/data/dem/#deminstance>`__) to
        DATA_TIMELESS EOPatch feature.
    """

    def __init__(
        self,
        feature: Union[None, str, FeatureSpec] = None,
        data_collection: DataCollection = DataCollection.DEM,
        **kwargs: Any,
    ):
        if feature is None:
            feature = (FeatureType.DATA_TIMELESS, "dem")
        elif isinstance(feature, str):
            feature = (FeatureType.DATA_TIMELESS, feature)

        feature_type, feature_name = feature
        if feature_type.is_temporal():
            raise ValueError("DEM feature should be timeless!")

        band = data_collection.bands[0]

        evalscript = f"""
            //VERSION=3

            function setup() {{
                return {{
                    input: [{{
                        bands: ["{band.name}"],
                        units: ["{band.units[0].value}"]
                    }}],
                    output: {{
                        id: "{feature_name}",
                        bands: 1,
                        sampleType: SampleType.UINT16
                    }}
                }}
            }}

            function evaluatePixel(sample) {{
                return {{ {feature_name}: [sample.{band.name}] }}
            }}
        """

        super().__init__(evalscript=evalscript, features=[feature], data_collection=data_collection, **kwargs)


class SentinelHubSen2corTask(SentinelHubInputTask):
    """
    Adds SCL (scene classification), CLD (cloud probability) or SNW (snow probability) (or their combination)
    Sen2Cor classification results to EOPatch's MASK or DATA feature. The feature is added to MASK (SCL) or
    DATA (CLD, SNW) feature types of EOPatch. The feature names are set to be SCL, CLD or SNW.

    Sen2Cor's scene classification (SCL) contains 11 classes with the following values and meanings:
       * 1 - SC_SATURATED_DEFECTIVE
       * 2 - SC_DARK_FEATURE_SHADOW
       * 3 - SC_CLOUD_SHADOW
       * 4 - VEGETATION
       * 5 - NOT-VEGETATED
       * 6 - WATER
       * 7 - SC_CLOUD_LOW_PROBA / UNCLASSIFIED
       * 8 - SC_CLOUD_MEDIUM_PROBA
       * 9 - CLOUD_HIGH_PROBABILITY
       * 10 - THIN_CIRRUS
       * 11 - SNOW
    """

    def __init__(
        self,
        sen2cor_classification: Union[Literal["SCL", "CLD", "SNW"], List[Literal["SCL", "CLD", "SNW"]]],
        data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
        **kwargs: Any,
    ):
        """
        :param sen2cor_classification: "SCL" (scene classification), "CLD" (cloud probability) or "SNW"
            (snow probability) masks to be retrieved. Also, a list of their combination (e.g. ["SCL","CLD"])
        :param kwargs: Additional arguments that will be passed to the `SentinelHubInputTask`
        """
        # definition of possible types and target features
        classification_types = {"SCL": FeatureType.MASK, "CLD": FeatureType.DATA, "SNW": FeatureType.DATA}

        if isinstance(sen2cor_classification, str):
            sen2cor_classification = [sen2cor_classification]

        for s2c in sen2cor_classification:
            if s2c not in classification_types:
                raise ValueError(
                    f"Unsupported Sen2Cor classification type: {s2c}. Possible types are: {classification_types}!"
                )

        if data_collection != DataCollection.SENTINEL2_L2A:
            raise ValueError("Sen2Cor classification layers are only available on Sentinel-2 L2A data.")

        features: List[Tuple[FeatureType, str]] = [(classification_types[s2c], s2c) for s2c in sen2cor_classification]
        super().__init__(additional_data=features, data_collection=data_collection, **kwargs)


def get_available_timestamps(
    bbox: BBox,
    data_collection: DataCollection,
    *,
    time_interval: Optional[RawTimeIntervalType] = None,
    time_difference: dt.timedelta = dt.timedelta(seconds=-1),  # noqa: B008
    timestamp_filter: Callable[[List[dt.datetime], dt.timedelta], List[dt.datetime]] = filter_times,
    maxcc: Optional[float] = None,
    config: Optional[SHConfig] = None,
) -> List[dt.datetime]:
    """Helper function to search for all available timestamps, based on query parameters.

    :param bbox: A bounding box of the search area.
    :param data_collection: A data collection for which to find available timestamps.
    :param time_interval: A time interval from which to provide the timestamps.
    :param time_difference: Minimum allowed time difference, used when filtering dates.
    :param timestamp_filter: A function that performs the final filtering of timestamps, usually to remove multiple
        occurrences within the time_difference window. The filtration is performed after all suitable timestamps for
        the given region are obtained (with maxcc filtering already done by SH). By default only keeps the oldest
        timestamp when multiple occur within `time_difference`.
    :param maxcc: Maximum cloud coverage filter from interval [0, 1], default is None.
    :param config: A configuration object.
    :return: A list of timestamps of available observations.
    """
    query_filter = None
    if maxcc is not None and data_collection.has_cloud_coverage:
        if isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError('Maximum cloud coverage "maxcc" parameter should be a float on an interval [0, 1]')
        query_filter = f"eo:cloud_cover < {int(maxcc * 100)}"

    fields = {"include": ["properties.datetime"], "exclude": []}

    if data_collection.service_url:
        config = config.copy() if config else SHConfig()
        config.sh_base_url = data_collection.service_url

    catalog = SentinelHubCatalog(config=config)
    search_iterator = catalog.search(
        collection=data_collection, bbox=bbox, time=time_interval, filter=query_filter, fields=fields
    )

    all_timestamps = search_iterator.get_timestamps()
    return timestamp_filter(all_timestamps, time_difference)
