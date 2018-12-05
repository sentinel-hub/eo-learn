# Basic EOLearn libraries
from eolearn.core import EOTask, FeatureType

import numpy as np


class ReferenceScenes(EOTask):
    """ Creates a layer of reference scenes which have the highest fraction of valid pixels.

        The number of reference scenes is limited to a definable number.

        Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018

        :param feature: Name of the eopatch data layer. Needs to be of the FeatureType "DATA".
        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param valid_fraction_feature: Name of the layer containing the valid fraction obtained with the EOTask
                                        'AddValidDataFraction'. Needs to be of the FeatureType "SCALAR".
        :type valid_fraction_feature: (FeatureType, str)
        :param max_scene_number: Maximum number of reference scenes taken for the creation of the composite. By default,
                                the maximum number of scenes equals the number of time frames
        :type max_scene_number: int

    """
    def __init__(self, feature, valid_fraction_feature, max_scene_number=None):
        self.feature = self._parse_features(feature, new_names=True,
                                            default_feature_type=FeatureType.DATA,
                                            rename_function='{}_REFERENCE'.format)
        self.valid_fraction_feature = self._parse_features(valid_fraction_feature,
                                                           default_feature_type=FeatureType.SCALAR)
        self.number = max_scene_number

    def execute(self, eopatch):
        feature_type, feature_name, new_feature_name = next(self.feature(eopatch))
        valid_fraction_feature_type, valid_fraction_feature_name = next(self.valid_fraction_feature(eopatch))

        valid_frac = list(eopatch[valid_fraction_feature_type][valid_fraction_feature_name].flatten())
        data = eopatch[feature_type][feature_name]

        self.number = data.shape[0] if self.number is None else self.number

        eopatch[feature_type][new_feature_name] = np.array([data[x] for _, x in
                                                            sorted(zip(valid_frac, range(data.shape[0])), reverse=True)
                                                            if x <= self.number-1])

        return eopatch


class BaseCompositing(EOTask):
    """
    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018
    Creates a composite of reference scenes.
    """

    def __init__(self, feature, feature_composite, blue_idx=None, red_idx=None, nir_idx=None, swir1_idx=None,
                 method=None, percentile=None, max_index=255, interpolation='lower', no_data_value=-32768):
        """
        :param method: Different compositing methods can be chosen:
          - blue     (25th percentile of the blue band)
          - HOT      (Index using bands blue and red)
          - maxNDVI  (temporal maximum of NDVI)
          - maxNDWI  (temporal maximum of NDWI)
          - maxRatio (temporal maximum of a ratio using bands blue, NIR and SWIR)
        :type method: str
        :param layer: Name of the eopatch data layer. Needs to be of the FeatureType "DATA"
        :type layer: str
        """
        self.feature = self._parse_features(feature,
                                            default_feature_type=FeatureType.DATA,
                                            rename_function='{}_COMPOSITE'.format)
        self.composite_type, self.composite_name = next(
            self._parse_features(feature_composite, default_feature_type=FeatureType.DATA_TIMELESS)())
        self.blue_idx = blue_idx
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.swir1_idx = swir1_idx
        self.method = method
        self.percentile = percentile
        self.max_index = max_index
        self.interpolation = interpolation
        self._index_by_percentile = self._geoville_index_by_percentile \
            if self.interpolation.lower() == 'geoville' else self._numpy_index_by_percentile
        self.no_data_value = no_data_value

    def _numpy_index_by_percentile(self, data, percentile):
        """Calculate percentile of numpy stack and return the index of the chosen pixel. """
        data_perc_low = np.nanpercentile(data, percentile, axis=0, interpolation=self.interpolation)

        indices = np.empty(data_perc_low.shape, dtype=np.uint8)
        indices[:] = np.nan

        abs_diff = np.where(np.isnan(data_perc_low), np.inf, abs(data - data_perc_low))

        indices = np.where(np.isnan(data_perc_low), self.max_index, np.nanargmin(abs_diff, axis=0))

        return indices

    def _geoville_index_by_percentile(self, data, percentile):
        """Calculate percentile of numpy stack and return the index of the chosen pixel. """
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
        f_arr = f_arr.astype(np.int)

        # get floor value of reference band and index band
        ind = f_arr.astype("int16")
        y_val = ind_tmp.shape[1]
        x_val = ind_tmp.shape[2]
        y_val, x_val = np.ogrid[0:y_val, 0:x_val]
        floor_val = ind_tmp[ind, y_val, x_val]

        idx = np.where(valid_obs == 0, 255, floor_val)

        return idx

    def _get_reference_band(self, data):
        raise NotImplementedError

    def _get_indices(self, data):
        indices = self._index_by_percentile(data, self.percentile)
        return indices

    def execute(self, eopatch):
        feature_type, feature_name = next(self.feature(eopatch))
        data = eopatch[feature_type][feature_name].copy()

        reference_bands = self._get_reference_band(data)

        indices = self._get_indices(reference_bands)

        composite_image = np.empty((data.shape[1:]), np.float32)
        composite_image[:] = self.no_data_value
        for scene_id, scene in enumerate(data):
            composite_image = np.where(np.dstack([indices]) == scene_id, scene, composite_image)

        eopatch[self.composite_type][self.composite_name] = composite_image

        return eopatch


class BlueCompositing(BaseCompositing):
    def __init__(self, feature, feature_composite, blue_idx, interpolation='lower'):
        super().__init__(feature, feature_composite, blue_idx=blue_idx, method='blue', percentile=25,
                         interpolation=interpolation)
        if self.blue_idx is None:
            raise ValueError('Index of blue band must be specified')

    def _get_reference_band(self, data):
        return data[..., self.blue_idx].astype("float32")


class HOTCompositing(BaseCompositing):
    def __init__(self, feature, feature_composite, blue_idx, red_idx, interpolation='lower'):
        super().__init__(feature, feature_composite, blue_idx=blue_idx, red_idx=red_idx, method='HOT', percentile=25,
                         interpolation=interpolation)
        if self.blue_idx is None or self.red_idx is None:
            raise ValueError('Index of blue band and red band must be specified')

    def _get_reference_band(self, data):
        return data[..., self.blue_idx] - 0.5 * data[..., self.red_idx] - 0.08


class MaxNDVICompositing(BaseCompositing):
    def __init__(self, feature, feature_composite, red_idx, nir_idx, interpolation='lower'):
        super().__init__(feature, feature_composite, red_idx=red_idx, nir_idx=nir_idx, method='maxNDVI',
                         percentile=[0, 100], interpolation=interpolation)
        if self.red_idx is None or self.nir_idx is None:
            raise ValueError('Index of NIR band and red band must be specified')

    def _get_reference_band(self, data):
        nir = data[..., self.nir_idx].astype("float32")
        red = data[..., self.red_idx].astype("float32")
        return (nir - red) / (nir + red)

    def _get_indices(self, data):
        median = np.nanmedian(data, axis=0)
        indices_min = self._index_by_percentile(data, self.percentile[0])
        indices_max = self._index_by_percentile(data, self.percentile[1])
        indices = np.where(median < -0.05, indices_min, indices_max)
        return indices


class MaxNDWICompositing(BaseCompositing):
    def __init__(self, feature, feature_composite, nir_idx, swir1_idx, interpolation='lower'):
        super().__init__(feature, feature_composite, nir_idx=nir_idx, swir1_idx=swir1_idx, method='maxNDWI',
                         percentile=100, interpolation=interpolation)
        if self.nir_idx is None or self.swir1_idx is None:
            raise ValueError('Index of NIR band and SWIR1 band must be specified')

    def _get_reference_band(self, data):
        nir = data[..., self.nir_idx].astype("float32")
        swir1 = data[..., self.swir1_idx].astype("float32")
        return (nir - swir1) / (nir + swir1)


class MaxRatioCompositing(BaseCompositing):
    def __init__(self, feature, feature_composite, blue_idx, nir_idx, swir1_idx, interpolation='lower'):
        super().__init__(feature, feature_composite, blue_idx=blue_idx, nir_idx=nir_idx, method='maxRatio',
                         percentile=100, interpolation=interpolation)
        if self.blue_idx is None or self.nir_idx is None or self.swir1_idx is None:
            raise ValueError('Index of blue band, NIR band and SWIR1 band must be specified')

    def _get_reference_band(self, data):
        blue = data[..., self.blue_idx].astype("float32")
        nir = data[..., self.nir_idx].astype("float32")
        swir1 = data[..., self.swir1_idx].astype("float32")
        return np.nanmax(np.array([nir, swir1]), axis=0) / blue


class HistogramMatching(EOTask):
    """ Histogram match of each band of each scene within a time-series with respect to the corresponding band of a
        reference composite.

        Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018

        :param feature: Name of the eopatch data layer that will undergo a histogram match.
                        Should be of the FeatureType "DATA".
        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param reference: Name of the eopatch data layer that represents the reference for the histogram match.
                            Should be of the FeatureType "DATA_TIMELESS".
        :type reference: (FeatureType, str)
        """

    def __init__(self, feature, reference):
        self.feature = self._parse_features(feature, new_names=True,
                                            default_feature_type=FeatureType.DATA,
                                            rename_function='{}_NORMALISED'.format)
        self.reference = self._parse_features(reference, default_feature_type=FeatureType.DATA_TIMELESS)

    def execute(self, eopatch):
        """ Perform histogram matching of the time-series with respect to a reference scene

        :param eopatch: eopatch holding the time-series and reference data
        :return: The same eopatch instance with the normalised time-series
        """
        feature_type, feature_name, new_feature_name = next(self.feature(eopatch))
        reference_type, reference_name = next(self.reference(eopatch))

        reference_scene = eopatch[reference_type][reference_name]
        # check if band dimension matches
        if reference_scene.shape[-1] != eopatch[feature_type][feature_name].shape[-1]:
            raise ValueError('Time-series and reference scene must have corresponding bands')

        eopatch[feature_type][new_feature_name] = np.zeros_like(eopatch[feature_type][feature_name])
        for source_id, source in enumerate(eopatch[feature_type][feature_name]):
            # mask-out same invalid pixels
            src_masked = np.where(np.isnan(reference_scene), np.nan, source)
            ref_masked = np.where(np.isnan(source), np.nan, reference_scene)
            # compute statistics
            std_ref = np.nanstd(ref_masked, axis=(0, 1), dtype=np.float64)
            std_src = np.nanstd(src_masked, axis=(0, 1), dtype=np.float64)
            mean_ref = np.nanmean(ref_masked, axis=(0, 1), dtype=np.float64)
            mean_src = np.nanmean(src_masked, axis=(0, 1), dtype=np.float64)
            # normalise values
            eopatch[feature_type][new_feature_name][source_id] = \
                source * (std_ref / std_src) + (mean_ref - (mean_src * (std_ref / std_src)))

        return eopatch
