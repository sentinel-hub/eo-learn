""" Input tasks that collect data from `Sentinel-Hub Process API
<https://docs.sentinel-hub.com/api/latest/api/process/>`__

Credits:
Copyright (c) 2019-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2019-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2021 Beno Šircelj

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import datetime as dt
import logging
from typing import List, Optional, Tuple

import numpy as np

from sentinelhub import (
    Band,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
    Unit,
    bbox_to_dimensions,
    filter_times,
    parse_time_interval,
    serialize_time,
)

from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet

LOGGER = logging.getLogger(__name__)


class SentinelHubInputBaseTask(EOTask):
    """Base class for Processing API input tasks"""

    def __init__(self, data_collection, size=None, resolution=None, cache_folder=None, config=None, max_threads=None):
        """
        :param data_collection: A collection of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :param resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        """
        if (size is None) == (resolution is None):
            raise ValueError("Exactly one of the parameters 'size' and 'resolution' should be given.")

        self.size = size
        self.resolution = resolution
        self.config = config or SHConfig()
        self.max_threads = max_threads
        self.data_collection = DataCollection(data_collection)
        self.cache_folder = cache_folder

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """Main execute method for the Process API tasks"""

        eopatch = eopatch or EOPatch()

        self._check_and_set_eopatch_bbox(bbox, eopatch)
        size_x, size_y = self._get_size(eopatch)

        if time_interval:
            time_interval = parse_time_interval(time_interval)
            timestamp = self._get_timestamp(time_interval, eopatch.bbox)
        elif self.data_collection.is_timeless:
            timestamp = None
        else:
            timestamp = eopatch.timestamp

        if timestamp is not None:
            eop_timestamp = [time_point.replace(tzinfo=None) for time_point in timestamp]
            if eopatch.timestamp:
                self.check_timestamp_difference(eop_timestamp, eopatch.timestamp)
            else:
                eopatch.timestamp = eop_timestamp

        requests = self._build_requests(eopatch.bbox, size_x, size_y, timestamp, time_interval)
        requests = [request.download_list[0] for request in requests]

        LOGGER.debug("Downloading %d requests of type %s", len(requests), str(self.data_collection))
        client = SentinelHubDownloadClient(config=self.config)
        responses = client.download(requests, max_threads=self.max_threads)
        LOGGER.debug("Downloads complete")

        temporal_dim = 1 if timestamp is None else len(timestamp)
        shape = temporal_dim, size_y, size_x
        self._extract_data(eopatch, responses, shape)

        eopatch.meta_info["size_x"] = size_x
        eopatch.meta_info["size_y"] = size_y
        if timestamp is not None:  # do not overwrite time interval in case of timeless features
            eopatch.meta_info["time_interval"] = serialize_time(time_interval)

        self._add_meta_info(eopatch)

        return eopatch

    def _get_size(self, eopatch):
        """Get the size (width, height) for the request either from inputs, or from the (existing) eopatch"""
        if self.size is not None:
            return self.size

        if self.resolution is not None:
            return bbox_to_dimensions(eopatch.bbox, self.resolution)

        if eopatch.meta_info and eopatch.meta_info.get("size_x") and eopatch.meta_info.get("size_y"):
            return eopatch.meta_info.get("size_x"), eopatch.meta_info.get("size_y")

        raise ValueError("Size or resolution for the requests should be provided!")

    def _add_meta_info(self, eopatch):
        """Add information to eopatch metadata"""
        if self.maxcc:
            eopatch.meta_info["maxcc"] = self.maxcc
        if self.time_difference:
            eopatch.meta_info["time_difference"] = self.time_difference.total_seconds()

    @staticmethod
    def _check_and_set_eopatch_bbox(bbox, eopatch):
        if eopatch.bbox is None:
            if bbox is None:
                raise ValueError("Either the eopatch or the task must provide valid bbox.")
            eopatch.bbox = bbox
            return

        if bbox is None or eopatch.bbox == bbox:
            return
        raise ValueError("Either the eopatch or the task must provide bbox, or they must be the same.")

    @staticmethod
    def check_timestamp_difference(timestamp1, timestamp2):
        """Raises an error if the two timestamps are not the same"""
        error_msg = "Trying to write data to an existing EOPatch with a different timestamp."
        if len(timestamp1) != len(timestamp2):
            raise ValueError(error_msg)

        for ts1, ts2 in zip(timestamp1, timestamp2):
            if ts1 != ts2:
                raise ValueError(error_msg)

    def _extract_data(self, eopatch, images, shape):
        """Extract data from the received images and assign them to eopatch features"""
        raise NotImplementedError("The _extract_data method should be implemented by the subclass.")

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """Build requests"""
        raise NotImplementedError("The _build_requests method should be implemented by the subclass.")

    def _get_timestamp(self, time_interval, bbox):
        """Get the timestamp array needed as a parameter for downloading the images"""
        raise NotImplementedError("The _get_timestamp method should be implemented by the subclass.")


class SentinelHubEvalscriptTask(SentinelHubInputBaseTask):
    """Process API task to download data using evalscript"""

    def __init__(
        self,
        features=None,
        evalscript=None,
        data_collection=None,
        size=None,
        resolution=None,
        maxcc=None,
        time_difference=None,
        cache_folder=None,
        max_threads=None,
        config=None,
        mosaicking_order=None,
        aux_request_args=None,
    ):
        """
        :param features: Features to construct from the evalscript.
        :param evalscript: Evalscript for the request. Beware that all outputs from SentinelHub services should be named
            and should have the same name as corresponding feature
        :type evalscript: str
        :param data_collection: Source of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :param resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param maxcc: Maximum cloud coverage, a float in interval [0, 1]
        :type maxcc: float
        :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
        :type time_difference: datetime.timedelta
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :type mosaicking_order: str
        :param aux_request_args: a dictionary with auxiliary information for the input_data part of the SH request
        :type aux_request_args: dict
        """
        super().__init__(
            data_collection=data_collection,
            size=size,
            resolution=resolution,
            cache_folder=cache_folder,
            config=config,
            max_threads=max_threads,
        )

        self.features = self._parse_and_validate_features(features)
        self.responses = self._create_response_objects()

        if not evalscript:
            raise ValueError("evalscript parameter must not be missing/empty")
        self.evalscript = evalscript

        if maxcc and isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError("maxcc should be a float on an interval [0, 1]")

        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.mosaicking_order = mosaicking_order
        self.aux_request_args = aux_request_args

    def _parse_and_validate_features(self, features):
        if not features:
            raise ValueError("features must be defined")

        allowed_features = FeatureTypeSet.RASTER_TYPES.union({FeatureType.META_INFO})
        _features = self.parse_renamed_features(features, allowed_feature_types=allowed_features)

        ftr_data_types = set(ft for ft, _, _ in _features if not ft.is_meta())
        if all(ft.is_timeless() for ft in ftr_data_types) or all(ft.is_temporal() for ft in ftr_data_types):
            return _features

        raise ValueError("Cannot mix time dependent and timeless requests!")

    def _create_response_objects(self):
        """Construct SentinelHubRequest output_responses from features"""
        responses = []
        for feat_type, feat_name, _ in self.features:
            if feat_type.is_raster():
                responses.append(SentinelHubRequest.output_response(feat_name, MimeType.TIFF))
            elif feat_type.is_meta():
                responses.append(SentinelHubRequest.output_response("userdata", MimeType.JSON))
            else:
                # should not happen as features have already been validated
                raise ValueError(f"{feat_type} not supported!")

        return responses

    def _get_timestamp(self, time_interval, bbox):
        """Get the timestamp array needed as a parameter for downloading the images"""
        if any(feat_type.is_timeless() for feat_type, _, _ in self.features if feat_type.is_raster()):
            return []

        return get_available_timestamps(
            bbox=bbox,
            time_interval=time_interval,
            data_collection=self.data_collection,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            config=self.config,
        )

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """Defines request timestamps and builds requests. In case `timestamp` is either `None` or an empty list it
        still has to create at least one request in order to obtain back number of bands of responses."""
        if timestamp:
            dates = [(date - self.time_difference, date + self.time_difference) for date in timestamp]
        elif timestamp is None:
            dates = [None]
        else:
            dates = [parse_time_interval(time_interval, allow_undefined=True)]

        return [self._create_sh_request(date, bbox, size_x, size_y) for date in dates]

    def _create_sh_request(self, time_interval, bbox, size_x, size_y):
        """Create an instance of SentinelHubRequest"""
        return SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    mosaicking_order=self.mosaicking_order,
                    time_interval=time_interval,
                    maxcc=self.maxcc,
                    other_args=self.aux_request_args,
                )
            ],
            responses=self.responses,
            bbox=bbox,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config,
        )

    def _extract_data(self, eopatch, data_responses, shape):
        """Extract data from the received images and assign them to eopatch features"""
        # pylint: disable=arguments-renamed
        if len(self.features) == 1:
            ftype, fname, _ = self.features[0]
            extension = "json" if ftype.is_meta() else "tif"
            data_responses = [{f"{fname}.{extension}": data} for data in data_responses]

        for ftype, fname, new_fname in self.features:
            if ftype.is_meta():
                data = [data["userdata.json"] for data in data_responses]

            elif ftype.is_temporal():
                data = np.asarray([data[f"{fname}.tif"] for data in data_responses])
                data = data[..., np.newaxis] if data.ndim == 3 else data
                time_dim = shape[0]
                data = data[:time_dim] if time_dim != data.shape[0] else data

            else:
                data = np.asarray(data_responses[0][f"{fname}.tif"])[..., np.newaxis]

            eopatch[ftype][new_fname] = data

        return eopatch


class SentinelHubInputTask(SentinelHubInputBaseTask):
    """Process API input task that loads 16bit integer data and converts it to a 32bit float feature."""

    # pylint: disable=too-many-arguments
    DTYPE_TO_SAMPLE_TYPE = {
        bool: "SampleType.UINT8",
        np.uint8: "SampleType.UINT8",
        np.uint16: "SampleType.UINT16",
        np.float32: "SampleType.FLOAT32",
    }

    def __init__(
        self,
        data_collection=None,
        size=None,
        resolution=None,
        bands_feature=None,
        bands=None,
        additional_data=None,
        evalscript=None,
        maxcc=None,
        time_difference=None,
        cache_folder=None,
        max_threads=None,
        config=None,
        bands_dtype=None,
        single_scene=False,
        mosaicking_order=None,
        aux_request_args=None,
    ):
        """
        :param data_collection: Source of requested satellite data.
        :type data_collection: DataCollection
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :param resolution: Resolution in meters, passed as a single number or a tuple of two numbers -
            resolution in horizontal and resolution in vertical direction.
        :type resolution: float or (float, float)
        :param bands_feature: A target feature into which to save the downloaded images.
        :type bands_feature: tuple(sentinelhub.FeatureType, str)
        :param bands: An array of band names. If not specified it will download all bands specified for a given data
            collection.
        :type bands: list[str]
        :param additional_data: A list of additional data to be downloaded, such as SCL, SNW, dataMask, etc.
        :type additional_data: list[tuple(sentinelhub.FeatureType, str)]
        :param evalscript: An optional parameter to override an evalscript that is generated by default
        :type evalscript: str or None
        :param maxcc: Maximum cloud coverage.
        :type maxcc: float
        :param time_difference: Minimum allowed time difference, used when filtering dates, None by default.
        :type time_difference: datetime.timedelta
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param config: An instance of SHConfig defining the service
        :type config: SHConfig or None
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param bands_dtype: output type of the bands array, if set to None the default is used
        :type bands_dtype: type or None
        :param single_scene: If true, the service will compute a single image for the given time interval using
            mosaicking.
        :type single_scene: bool
        :param mosaicking_order: Mosaicking order, which has to be either 'mostRecent', 'leastRecent' or 'leastCC'.
        :type mosaicking_order: str
        :param aux_request_args: a dictionary with auxiliary information for the input_data part of the SH request
        :type aux_request_args: dict
        """
        super().__init__(
            data_collection=data_collection,
            size=size,
            resolution=resolution,
            cache_folder=cache_folder,
            config=config,
            max_threads=max_threads,
        )
        self.evalscript = evalscript
        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=1)
        self.single_scene = single_scene
        self.bands_dtype = bands_dtype
        self.mosaicking_order = mosaicking_order
        self.aux_request_args = aux_request_args

        self.bands_feature = None
        self.requested_bands = []
        if bands_feature:
            self.bands_feature = self.parse_feature(bands_feature, allowed_feature_types=[FeatureType.DATA])
            if bands:
                self.requested_bands = self._parse_requested_bands(bands, self.data_collection.bands)
            else:
                self.requested_bands = list(self.data_collection.bands)

        self.requested_additional_bands = []
        if additional_data is not None:
            additional_data = self.parse_renamed_features(additional_data)
            additional_bands = [band for _, band, _ in additional_data]
            parsed_bands = self._parse_requested_bands(additional_bands, self.data_collection.metabands)
            self.requested_additional_bands = parsed_bands

        self.additional_data = additional_data

    def _parse_requested_bands(self, bands, available_bands):
        """Checks that all requested bands are available and returns the band information for further processing"""
        requested_bands = []
        band_info_dict = {band_info.name: band_info for band_info in available_bands}
        for band_name in bands:
            if band_name in band_info_dict:
                requested_bands.append(band_info_dict[band_name])
            elif self.data_collection.is_batch or self.data_collection.is_byoc:
                requested_bands.append(Band(band_name, (Unit.DN,), (np.float32,)))
            else:
                raise ValueError(
                    f"Data collection {self.data_collection} does not have specifications for {band_name}."
                    f"Available bands are {[band.name for band in self.data_collection.bands]} and meta-bands"
                    f"{[band.name for band in self.data_collection.metabands]}"
                )
        return requested_bands

    def generate_evalscript(self):
        """Generate the evalscript to be passed with the request, based on chosen bands"""
        evalscript = """
            //VERSION=3

            function setup() {{
                return {{
                    input: [{{
                        bands: [{bands}],
                        units: [{units}]
                    }}],
                    output: [
                        {outputs}
                    ]
                }}
            }}

            function evaluatePixel(sample) {{
                return {{ {samples} }}
            }}
        """

        bands, units, outputs, samples = [], [], [], []
        for band in self.requested_bands + self.requested_additional_bands:
            unit_choice = 0  # use default units
            if band in self.requested_bands and self.bands_dtype is not None:
                if self.bands_dtype not in band.output_types:
                    raise ValueError(
                        f"Band {band.name} only supports output types {band.output_types} but `bands_dtype` is set to "
                        f"{self.bands_dtype}. To use default types set `bands_dtype` to None."
                    )
                unit_choice = band.output_types.index(self.bands_dtype)

            sample_type = SentinelHubInputTask.DTYPE_TO_SAMPLE_TYPE[band.output_types[unit_choice]]

            bands.append(f'"{band.name}"')
            units.append(f'"{band.units[unit_choice].value}"')
            samples.append(f"{band.name}: [sample.{band.name}]")
            outputs.append(f'{{ id: "{band.name}", bands: 1, sampleType: {sample_type} }}')

        evalscript = evalscript.format(
            bands=", ".join(bands), units=", ".join(units), outputs=", ".join(outputs), samples=", ".join(samples)
        )

        return evalscript

    def _get_timestamp(self, time_interval, bbox):
        """Get the timestamp array needed as a parameter for downloading the images"""
        if self.single_scene:
            return [time_interval[0]]

        return get_available_timestamps(
            bbox=bbox,
            time_interval=time_interval,
            data_collection=self.data_collection,
            maxcc=self.maxcc,
            time_difference=self.time_difference,
            config=self.config,
        )

    def _build_requests(self, bbox, size_x, size_y, timestamp, time_interval):
        """Build requests"""
        if timestamp is None:
            dates = [None]
        elif self.single_scene:
            dates = [parse_time_interval(time_interval)]
        else:
            dates = [(date - self.time_difference, date + self.time_difference) for date in timestamp]

        return [self._create_sh_request(date1, date2, bbox, size_x, size_y) for date1, date2 in dates]

    def _create_sh_request(self, date_from, date_to, bbox, size_x, size_y):
        """Create an instance of SentinelHubRequest"""
        responses = [
            SentinelHubRequest.output_response(band.name, MimeType.TIFF)
            for band in self.requested_bands + self.requested_additional_bands
        ]

        return SentinelHubRequest(
            evalscript=self.evalscript or self.generate_evalscript(),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    time_interval=(date_from, date_to),
                    mosaicking_order=self.mosaicking_order,
                    maxcc=self.maxcc,
                    other_args=self.aux_request_args,
                )
            ],
            responses=responses,
            bbox=bbox,
            size=(size_x, size_y),
            data_folder=self.cache_folder,
            config=self.config,
        )

    def _extract_data(self, eopatch, images, shape):
        """Extract data from the received images and assign them to eopatch features"""
        if len(self.requested_bands) + len(self.requested_additional_bands) == 1:
            # if only one band is requested the response is not a tar so we reshape it
            only_band = (self.requested_bands + self.requested_additional_bands)[0]
            images = [{only_band.name + ".tif": image} for image in images]

        if self.additional_data:
            self._extract_additional_features(eopatch, images, shape)

        if self.bands_feature:
            self._extract_bands_feature(eopatch, images, shape)

        return eopatch

    def _extract_additional_features(self, eopatch, images, shape):
        """Extracts additional features from response into an EOPatch"""
        for (ftype, _, new_name), band_info in zip(self.additional_data, self.requested_additional_bands):
            tiffs = [tar[band_info.name + ".tif"] for tar in images]
            eopatch[ftype, new_name] = self._extract_array(tiffs, 0, shape, band_info.output_types[0])

    def _extract_bands_feature(self, eopatch, images, shape):
        """Extract the bands feature arrays and concatenate them along the last axis"""
        processed_bands = []
        for band_info in self.requested_bands:
            tiffs = [tar[band_info.name + ".tif"] for tar in images]
            dtype = self.bands_dtype or band_info.output_types[0]
            processed_bands.append(self._extract_array(tiffs, 0, shape, dtype))

        eopatch[self.bands_feature] = np.concatenate(processed_bands, axis=-1)

    @staticmethod
    def _extract_array(tiffs, idx, shape, dtype):
        """Extract a numpy array from the received tiffs"""
        feature_arrays = (np.atleast_3d(img)[..., idx] for img in tiffs)
        return np.asarray(list(feature_arrays), dtype=dtype).reshape(*shape, 1)


class SentinelHubDemTask(SentinelHubEvalscriptTask):
    """
    Adds DEM data (one of the `collections <https://docs.sentinel-hub.com/api/latest/data/dem/#deminstance>`__) to
        DATA_TIMELESS EOPatch feature.
    """

    def __init__(self, feature=None, data_collection=DataCollection.DEM, **kwargs):
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

    def __init__(self, sen2cor_classification, data_collection=DataCollection.SENTINEL2_L2A, **kwargs):
        """
        :param sen2cor_classification: "SCL" (scene classification), "CLD" (cloud probability) or "SNW"
            (snow probability) masks to be retrieved. Also, a list of their combination (e.g. ["SCL","CLD"])
        :type sen2cor_classification: str or [str]
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

        features = [(classification_types[s2c], s2c) for s2c in sen2cor_classification]
        super().__init__(additional_data=features, data_collection=data_collection, **kwargs)


def get_available_timestamps(
    bbox: BBox,
    data_collection: DataCollection,
    *,
    time_interval: Optional[Tuple[dt.datetime, dt.datetime]] = None,
    time_difference: dt.timedelta = dt.timedelta(seconds=-1),
    maxcc: Optional[float] = None,
    config: Optional[SHConfig] = None,
) -> List[dt.datetime]:
    """Helper function to search for all available timestamps, based on query parameters.

    :param bbox: A bounding box of the search area.
    :param data_collection: A data collection for which to find available timestamps.
    :param time_interval: A time interval from which to provide the timestamps.
    :param time_difference: Minimum allowed time difference, used when filtering dates.
    :param maxcc: Maximum cloud coverage filter from interval [0, 1], default is None.
    :param config: A configuration object.
    :return: A list of timestamps of available observations.
    """
    query = None
    if maxcc is not None and data_collection.has_cloud_coverage:
        if isinstance(maxcc, (int, float)) and (maxcc < 0 or maxcc > 1):
            raise ValueError('Maximum cloud coverage "maxcc" parameter should be a float on an interval [0, 1]')
        query = {"eo:cloud_cover": {"lte": int(maxcc * 100)}}

    fields = {"include": ["properties.datetime"], "exclude": []}

    if data_collection.service_url:
        config = config.copy() if config else SHConfig()
        config.sh_base_url = data_collection.service_url

    catalog = SentinelHubCatalog(config=config)
    search_iterator = catalog.search(
        collection=data_collection, bbox=bbox, time=time_interval, query=query, fields=fields
    )

    all_timestamps = search_iterator.get_timestamps()
    return filter_times(all_timestamps, time_difference)
