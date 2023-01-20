"""
Module for radiometric normalization

Credits:
Copyright (c) 2018-2019 Johannes Schmid (GeoVille)
Copyright (c) 2017-2022 Matej Aleksandrov, Matic Lubej, Devis Peressutti, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import SingleFeatureSpec


class ReferenceScenesTask(EOTask):
    """Creates a layer of reference scenes which have the highest fraction of valid pixels.

    The number of reference scenes is limited to a definable number.

    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018

    :param feature: Name of the eopatch data layer. Needs to be of the FeatureType "DATA".
    :param valid_fraction_feature: Name of the layer containing the valid fraction obtained with the EOTask
        'AddValidDataFraction'. Needs to be of the FeatureType "SCALAR".
    :param max_scene_number: Maximum number of reference scenes taken for the creation of the composite. By default,
        the maximum number of scenes equals the number of time frames

    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        valid_fraction_feature: SingleFeatureSpec,
        max_scene_number: Optional[int] = None,
    ):
        self.renamed_feature = self.parse_renamed_feature(feature)
        self.valid_fraction_feature = self.parse_feature(valid_fraction_feature)
        self.number = max_scene_number

    def execute(self, eopatch: EOPatch) -> EOPatch:
        feature_type, feature_name, new_feature_name = self.renamed_feature
        valid_fraction_feature_type, valid_fraction_feature_name = self.valid_fraction_feature

        valid_frac = list(eopatch[valid_fraction_feature_type][valid_fraction_feature_name].flatten())
        data = eopatch[feature_type, feature_name]

        number = data.shape[0] if self.number is None else self.number

        eopatch[feature_type, new_feature_name] = np.array(
            [data[x] for _, x in sorted(zip(valid_frac, range(data.shape[0])), reverse=True) if x <= number - 1]
        )

        return eopatch


class BaseCompositingTask(EOTask, metaclass=ABCMeta):
    """Base class to create a composite of reference scenes

    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018

    :param feature: Feature holding the input time-series. Default type is FeatureType.DATA
    :param feature_composite: Type and name of output composite image. Default type is FeatureType.DATA_TIMELESS
    :param percentile: Percentile along the time dimension used for compositing. Methods use different percentiles
    :param max_index: Value used to flag indices with NaNs. Could be integer or NaN. Default is 255
    :param interpolation: Method used to compute percentile. Allowed values are {'geoville', 'linear', 'lower',
        'higher', 'midpoint', 'nearest'}. 'geoville' interpolation performs a custom implementation, while the other
        methods use the numpy `percentile` function. Default is 'lower'.
    :param no_data_value: Value in the composite assigned to non-valid data points. Default is NaN
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        feature_composite: SingleFeatureSpec,
        percentile: int,
        max_index: int = 255,
        interpolation: str = "lower",
        no_data_value: float = np.nan,
    ):
        self.feature = self.parse_feature(feature)
        self.composite_type, self.composite_name = self.parse_feature(feature_composite)
        self.percentile = percentile
        self.max_index = max_index
        self.interpolation = interpolation
        self._index_by_percentile = (
            self._geoville_index_by_percentile
            if self.interpolation.lower() == "geoville"
            else self._numpy_index_by_percentile
        )
        self.no_data_value = no_data_value

    def _numpy_index_by_percentile(self, data: np.ndarray, percentile: float) -> np.ndarray:
        """Calculate percentile of numpy stack and return the index of the chosen pixel.

        numpy percentile function is used with one of the following interpolations {'linear', 'lower', 'higher',
        'midpoint', 'nearest'}
        """
        data_perc_low = np.nanpercentile(  # type: ignore[call-overload]  # numpy is weird here
            data, percentile, axis=0, interpolation=self.interpolation
        )

        indices = np.empty(data_perc_low.shape, dtype=np.uint8)
        indices[:] = np.nan

        abs_diff = np.where(np.isnan(data_perc_low), np.inf, abs(data - data_perc_low))

        indices = np.where(np.isnan(data_perc_low), self.max_index, np.nanargmin(abs_diff, axis=0))

        return indices

    def _geoville_index_by_percentile(self, data: np.ndarray, percentile: int) -> np.ndarray:
        """Calculate percentile of numpy stack and return the index of the chosen pixel."""
        # no_obs = bn.allnan(arr_tmp["data"], axis=0)
        data_tmp = np.array(data, copy=True)
        valid_obs = np.sum(np.isfinite(data_tmp), axis=0)
        # replace NaN with maximum
        max_val = np.nanmax(data_tmp) + 1
        data_tmp[np.isnan(data_tmp)] = max_val
        # sort - former NaNs will move to the end
        ind_tmp = np.argsort(data_tmp, kind="mergesort", axis=0)
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (percentile / 100.0)
        k_arr = np.where(k_arr < 0, 0, k_arr)
        f_arr = np.floor(k_arr + 0.5)
        f_arr = f_arr.astype(int)
        # get floor value of reference band and index band
        ind = f_arr.astype("int16")
        y_val, x_val = ind_tmp.shape[1], ind_tmp.shape[2]
        y_val, x_val = np.ogrid[0:y_val, 0:x_val]  # type: ignore[assignment]
        idx = np.where(valid_obs == 0, self.max_index, ind_tmp[ind, y_val, x_val])
        return idx

    @abstractmethod
    def _get_reference_band(self, data: np.ndarray) -> np.ndarray:
        """Extract reference band from input 4D data according to compositing method

        :param data: 4D array from which to extract reference band (e.g. blue, maxNDVI, ..)
        :return: 3D array containing reference band according to compositing method
        """

    def _get_indices(self, data: np.ndarray) -> np.ndarray:
        """Compute indices along temporal dimension corresponding to the sought percentile

        :param data: Input 3D array holding the reference band
        :return: 2D array holding the temporal index corresponding to percentile
        """
        indices = self._index_by_percentile(data, self.percentile)
        return indices

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Compute composite array merging temporal frames according to the compositing method

        :param eopatch: eopatch holding time-series
        :return: eopatch with composite image of time-series
        """
        data = eopatch[self.feature].copy()

        # compute band according to compositing method (e.g. blue, maxNDVI, maxNDWI)
        reference_bands = self._get_reference_band(data)

        # find temporal indices corresponding to pre-defined percentile
        indices = self._get_indices(reference_bands)

        # compute composite image selecting values along temporal dimension corresponding to percentile indices
        composite_image = np.empty((data.shape[1:]), np.float32)
        composite_image[:] = self.no_data_value
        for scene_id, scene in enumerate(data):
            composite_image = np.where(np.dstack([indices]) == scene_id, scene, composite_image)

        eopatch[self.composite_type, self.composite_name] = composite_image

        return eopatch


class BlueCompositingTask(BaseCompositingTask):
    """Blue band compositing method

    - blue     (25th percentile of the blue band)

    :param blue_idx: Index of blue band in `feature` array
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        feature_composite: SingleFeatureSpec,
        blue_idx: int,
        interpolation: str = "lower",
    ):
        super().__init__(feature, feature_composite, percentile=25, interpolation=interpolation)
        self.blue_idx = blue_idx
        if not isinstance(blue_idx, int):
            raise ValueError("Incorrect value of blue band index specified")

    def _get_reference_band(self, data: np.ndarray) -> np.ndarray:
        """Extract the blue band from time-series

        :param data: 4D array from which to extract the blue reference band
        :return: 3D array containing the blue reference band
        """
        return data[..., self.blue_idx].astype("float32")


class HOTCompositingTask(BaseCompositingTask):
    """HOT compositing method

    - HOT      (Index using bands blue and red)

    The HOT index is defined as per
            Zhu, Z., & Woodcock, C. E. (2012). "Object-based cloud and cloud shadow detection in Landsat imagery."
            Remote Sensing of Environment, 118, 83-94.

    :param blue_idx: Index of blue band in `feature` array
    :param red_idx: Index of red band in `feature` array
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        feature_composite: SingleFeatureSpec,
        blue_idx: int,
        red_idx: int,
        interpolation: str = "lower",
    ):
        super().__init__(feature, feature_composite, percentile=25, interpolation=interpolation)
        self.blue_idx = blue_idx
        self.red_idx = red_idx
        if not isinstance(blue_idx, int) or not isinstance(red_idx, int):
            raise ValueError("Incorrect values of blue and red band indices specified")

    def _get_reference_band(self, data: np.ndarray) -> np.ndarray:
        """Extract the HOT band from time-series

        :param data: 4D array from which to extract the HOT reference band
        :return: 3D array containing the HOT reference band
        """
        return data[..., self.blue_idx] - 0.5 * data[..., self.red_idx] - 0.08


class MaxNDVICompositingTask(BaseCompositingTask):
    """maxNDVI compositing method

    - maxNDVI  (temporal maximum of NDVI)

    :param red_idx: Index of red band in `feature` array
    :param nir_idx: Index of NIR band in `feature` array
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        feature_composite: SingleFeatureSpec,
        red_idx: int,
        nir_idx: int,
        interpolation: str = "lower",
    ):
        super().__init__(  # No real value for percentile
            feature, feature_composite, percentile=None, interpolation=interpolation  # type: ignore[arg-type]
        )
        self.percentiles = [0, 100]
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        if not isinstance(nir_idx, int) or not isinstance(red_idx, int):
            raise ValueError("Incorrect values of red and NIR band indices specified")

    def _get_reference_band(self, data: np.ndarray) -> np.ndarray:
        """Extract the NDVI band from time-series

        :param data: 4D array from which to compute the NDVI reference band
        :return: 3D array containing the NDVI reference band
        """
        nir = data[..., self.nir_idx].astype("float32")
        red = data[..., self.red_idx].astype("float32")
        return (nir - red) / (nir + red)

    def _get_indices(self, data: np.ndarray) -> np.ndarray:
        median = np.nanmedian(data, axis=0)
        indices_min = self._index_by_percentile(data, self.percentiles[0])
        indices_max = self._index_by_percentile(data, self.percentiles[1])
        indices = np.where(median < -0.05, indices_min, indices_max)
        return indices


class MaxNDWICompositingTask(BaseCompositingTask):
    """maxNDWI compositing method

    - maxNDWI  (temporal maximum of NDWI)

    :param nir_idx: Index of NIR band in `feature` array
    :param swir1_idx: Index of SWIR1 band in `feature` array
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        feature_composite: SingleFeatureSpec,
        nir_idx: int,
        swir1_idx: int,
        interpolation: str = "lower",
    ):
        super().__init__(feature, feature_composite, percentile=100, interpolation=interpolation)
        self.nir_idx = nir_idx
        self.swir1_idx = swir1_idx
        if not isinstance(nir_idx, int) or not isinstance(swir1_idx, int):
            raise ValueError("Incorrect values of NIR and SWIR1 band indices specified")

    def _get_reference_band(self, data: np.ndarray) -> np.ndarray:
        """Extract the NDWI band from time-series

        :param data: 4D array from which to compute the NDWI reference band
        :return: 3D array containing the NDWI reference band
        """
        nir = data[..., self.nir_idx].astype("float32")
        swir1 = data[..., self.swir1_idx].astype("float32")
        return (nir - swir1) / (nir + swir1)


class MaxRatioCompositingTask(BaseCompositingTask):
    """maxRatio compositing method

    - maxRatio (temporal maximum of a ratio using bands blue, NIR and SWIR)

    :param blue_idx: Index of blue band in `feature` array
    :param nir_idx: Index of NIR band in `feature` array
    :param swir1_idx: Index of SWIR1 band in `feature` array
    """

    def __init__(
        self,
        feature: SingleFeatureSpec,
        feature_composite: SingleFeatureSpec,
        blue_idx: int,
        nir_idx: int,
        swir1_idx: int,
        interpolation: str = "lower",
    ):
        super().__init__(feature, feature_composite, percentile=100, interpolation=interpolation)
        self.blue_idx = blue_idx
        self.nir_idx = nir_idx
        self.swir1_idx = swir1_idx
        if not isinstance(blue_idx, int) or not isinstance(nir_idx, int) or not isinstance(swir1_idx, int):
            raise ValueError("Incorrect values for either blue, NIR or SWIR1 band indices specified")

    def _get_reference_band(self, data: np.ndarray) -> np.ndarray:
        """Extract the max-ratio band from time-series

        The max-ratio is defined as max(NIR,SWIR1)/BLUE

        :param data: 4D array from which to compute the max-ratio reference band
        :return: 3D array containing the max-ratio reference band
        """
        blue = data[..., self.blue_idx].astype("float32")
        nir = data[..., self.nir_idx].astype("float32")
        swir1 = data[..., self.swir1_idx].astype("float32")
        return np.nanmax(np.array([nir, swir1]), axis=0) / blue


class HistogramMatchingTask(EOTask):
    """Histogram match of each band of each scene within a time-series with respect to the corresponding band of a
    reference composite.

    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018

    :param feature: Name of the eopatch data layer that will undergo a histogram match.
        Should be of the FeatureType "DATA".
    :param reference: Name of the eopatch data layer that represents the reference for the histogram match.
        Should be of the FeatureType "DATA_TIMELESS".
    """

    def __init__(self, feature: SingleFeatureSpec, reference: SingleFeatureSpec):
        self.renamed_feature = self.parse_renamed_feature(feature)
        self.reference = self.parse_feature(reference)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Perform histogram matching of the time-series with respect to a reference scene

        :param eopatch: eopatch holding the time-series and reference data
        :return: The same eopatch instance with the normalised time-series
        """
        feature_type, feature_name, new_feature_name = self.renamed_feature
        reference_type, reference_name = self.reference

        reference_scene = eopatch[reference_type, reference_name]
        # check if band dimension matches
        if reference_scene.shape[-1] != eopatch[feature_type, feature_name].shape[-1]:
            raise ValueError("Time-series and reference scene must have corresponding bands")

        eopatch[feature_type, new_feature_name] = np.zeros_like(eopatch[feature_type, feature_name])
        for source_id, source in enumerate(eopatch[feature_type, feature_name]):
            # mask-out same invalid pixels
            src_masked = np.where(np.isnan(reference_scene), np.nan, source)
            ref_masked = np.where(np.isnan(source), np.nan, reference_scene)
            # compute statistics
            std_ref = np.nanstd(ref_masked, axis=(0, 1), dtype=np.float64)
            std_src = np.nanstd(src_masked, axis=(0, 1), dtype=np.float64)
            mean_ref = np.nanmean(ref_masked, axis=(0, 1), dtype=np.float64)
            mean_src = np.nanmean(src_masked, axis=(0, 1), dtype=np.float64)
            # normalise values
            eopatch[feature_type, new_feature_name][source_id] = source * (std_ref / std_src) + (
                mean_ref - (mean_src * (std_ref / std_src))
            )

        return eopatch
