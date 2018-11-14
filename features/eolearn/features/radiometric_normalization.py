# Basic EOLearn libraries
from eolearn.core import EOTask, FeatureType

import numpy as np


class ReferenceScenes(EOTask):
    """
    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018
    Creates a layer of reference scenes which have the highest fraction of valid pixels.
    The number of reference scenes is limited to a definable number.
    """
    def __init__(self, number, layer):
        """
        :param number: Maximum number of reference scenes taken for the creation of the composite.
        :type number: int
        :param layer: Name of the eopatch data layer. Needs to be of the FeatureType "DATA"
        :type layer: str
        """
        self.number = number
        self.layer = layer

    def execute(self, eopatch):
        valid_frac = list(eopatch.scalar["VALID_FRAC"].flatten())
        data_layers = eopatch.data[self.layer]
        out = np.array([data_layers[x] for _, x in sorted(zip(valid_frac, range(data_layers.shape[0])), reverse=True)
                        if x <= self.number-1])

        eopatch.add_feature(FeatureType.DATA, 'reference_scenes', out)
        return eopatch


class CompositeReferenceScenes(EOTask):
    """
    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018
    Creates a composite of reference scenes.
    """
    def __init__(self, method, layer):
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
        self.method = method
        self.layer = layer
        self.perc_no = 25
        self.ref_stack = None

    def index_by_percentile(self, quant):
        """Calculate percentile of numpy stack and return the index of the chosen pixel. """
        # valid (non NaN) observations along the first axis
        arr_tmp = np.array(self.ref_stack, copy=True)

        # no_obs = bn.allnan(arr_tmp["data"], axis=0)
        valid_obs = np.sum(np.isfinite(arr_tmp["data"]), axis=0)
        # replace NaN with maximum
        max_val = np.nanmax(arr_tmp["data"]) + 1
        arr_tmp["data"][np.isnan(arr_tmp["data"])] = max_val
        # sort - former NaNs will move to the end
        arr_tmp = np.sort(arr_tmp, kind="mergesort", order="data", axis=0)
        arr_tmp["data"] = np.where(arr_tmp["data"] == max_val, np.nan, arr_tmp["data"])

        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        k_arr = np.where(k_arr < 0, 0, k_arr)
        f_arr = np.floor(k_arr + 0.5)
        f_arr = f_arr.astype(np.int)

        # get floor value of reference band and index band
        ind = f_arr.astype("int16")
        y_val = arr_tmp.shape[1]
        x_val = arr_tmp.shape[2]
        y_val, x_val = np.ogrid[0:y_val, 0:x_val]
        floor_val = arr_tmp[ind, y_val, x_val]

        idx = np.where(valid_obs == 0, 255, floor_val["ID"])

        quant_arr = floor_val["data"]

        del arr_tmp

        return quant_arr, idx, valid_obs

    def get_band(self, sce):
        success = False
        ref = None
        if self.method == "blue":
            ref = sce[:, :, 0].astype("float32")
            success = True
        elif self.method == "maxNDVI":
            nir = sce[:, :, 7].astype("float32")
            red = sce[:, :, 2].astype("float32")
            ref = (nir - red) / (nir + red)
            del nir, red
            success = True
        elif self.method == "HOT":
            blue = sce[:, :, 0].astype("float32")
            red = sce[:, :, 2].astype("float32")
            ref = blue - 0.5 * red - 800.
            del blue, red
            success = True
        elif self.method == "maxRatio":
            blue = sce[:, :, 0].astype("float32")
            nir = sce[:, :, 6].astype("float32")
            swir1 = sce[:, :, 8].astype("float32")
            ref = np.nanmax(np.array([nir, swir1]), axis=0) / blue
            del blue, nir, swir1
            success = True
        elif self.method == "maxNDWI":
            nir = sce[:, :, 6].astype("float32")
            swir1 = sce[:, :, 8].astype("float32")
            ref = (nir - swir1) / (nir + swir1)
            del nir, swir1
            success = True

        return success, ref

    def execute(self, eopatch):
        # Dictionary connecting sceneIDs and composite index number
        data_layers = eopatch.data[self.layer]
        scene_lookup = {}
        for j, dlayer in enumerate(data_layers):
            scene_lookup[j] = dlayer

        # Stack scenes with data and scene index number
        dims = (data_layers.shape[0], data_layers.shape[1], data_layers.shape[2])
        self.ref_stack = np.zeros(dims, dtype=[("data", "f4"), ("ID", "u1")])
        self.ref_stack["ID"] = 255

        # Reference bands
        for j, sce in enumerate(data_layers):  # for each scene/time

            # Read in bands depending on composite method
            success, ref = self.get_band(sce)
            if self.method not in ["blue", "maxNDVI", "HOT", "maxRatio", "maxNDWI"]:
                raise ValueError("{} is not a valid compositing method!".format(self.method))
            if not success:
                raise NameError("Bands for composite method {} were not found!".format(self.method))

            # Write data to stack
            self.ref_stack["data"][j] = ref
            self.ref_stack["ID"][j] = j
            del ref

        # Calculate composite index (which scene is used for the composite per pixel)
        valid_obs = None
        index = None
        if self.method == "blue":
            index, valid_obs = self.index_by_percentile(self.perc_no)[1:3]
        elif self.method == "maxNDVI":
            median = np.nanmedian(self.ref_stack["data"], axis=0)
            index_max, valid_obs = self.index_by_percentile(100)[1:3]
            index_min, valid_obs = self.index_by_percentile(0)[1:3]
            index = np.where(median < -0.05, index_min, index_max)
        elif self.method == "HOT":
            index, valid_obs = self.index_by_percentile(25)[1:3]
        elif self.method == "maxRatio":
            index, valid_obs = self.index_by_percentile(100)[1:3]
        elif self.method == "maxNDWI":
            index, valid_obs = self.index_by_percentile(100)[1:3]

        try:
            del self.ref_stack, valid_obs
            index_scenes = [scene_lookup[idx] for idx in np.unique(index) if idx != 255]
        except IndexError:
            raise IndexError("ID not in scene_lookup.")

        # Create Output
        composite_image = np.empty(shape=(data_layers.shape[1], data_layers.shape[2],
                                          data_layers.shape[3]), dtype="float32")
        composite_image[:] = np.nan
        for j, sce in zip(range(len(data_layers)), index_scenes):
            composite_image = np.where(np.dstack([index]) == j, sce, composite_image)

        composite_image = np.where(np.isnan(composite_image), -32768, composite_image)

        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'reference_composite', composite_image)

        return eopatch


class HistogramMatching(EOTask):
    """
    Contributor: Johannes Schmid, GeoVille Information Systems GmbH, 2018
    Histogram match of each band of each scene with the band of the respective reference composite.
    """
    def __init__(self, source, reference):
        """
        :param source: Name of the eopatch data layer that will undergo a histogram match.
        Needs to be of the FeatureType "DATA".
        :type source: str
        :param reference: Name of the eopatch data layer that represents the reference for the histogram match.
        Needs to be of the FeatureType "DATA_TIMELESS".
        :type reference: str
        """
        self.source_layer = source
        self.reference_layer = reference

    def execute(self, eopatch):

        source_scenes = eopatch.data[self.source_layer]
        reference_scene = eopatch.data_timeless[self.reference_layer]

        eopatch.add_feature(FeatureType.DATA, 'radiometric_normalized', np.zeros(source_scenes.shape))
        for sce_id, sce in enumerate(list(source_scenes)):
            for band in range(source_scenes[0].shape[2]):

                source_band = np.where(np.isnan(reference_scene[:, :, band]), np.nan, sce[:, :, band])
                reference_band = np.where(np.isnan(sce[:, :, band]), np.nan, reference_scene[:, :, band])

                std_src = np.nanstd(source_band)
                std_ref = np.nanstd(reference_band)
                mean_src = np.nanmean(source_band)
                mean_ref = np.nanmean(reference_band)

                eopatch.data['radiometric_normalized'][sce_id, :, :, band] = \
                    sce[:, :, band] * (std_ref / std_src) + (mean_ref - (mean_src * (std_ref / std_src)))

        return eopatch
