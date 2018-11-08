# Basic EOLearn libraries
from eolearn.core import EOTask, EOPatch, FeatureType

# Numpy for manipulating the rasters in form of arrays
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
        :type feature: int
        :param layer: Name of the eopatch data layer. Needs to be of the FeatureType "DATA"
        :type layer: str
        """
        self.number = number
        self.layer = layer
        
    def execute(self, eopatch):
        valid_frac = list(eopatch.scalar["VALID_FRAC"].flatten())
        data_layers = eopatch.data[self.layer]
        out = np.array([data_layers[x] for _,x in sorted(zip(valid_frac,range(data_layers.shape[0])), reverse=True) if x <= self.number-1])
            
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
        self.percNo = 25

    def _zvalue_from_index_ogrid(self, arr, ind):
        y = arr.shape[1]
        x = arr.shape[2]
        y,x =np.ogrid[0:y, 0:x]

        return arr[ind, y, x]

    def indexByPercentile(self, arr, quant):
        """Calculate percentile of numpy stack and return the index of the chosen pixel. """
        #valid (non NaN) observations along the first axis
        arr_tmp = np.array(arr, copy=True)

        #no_obs = bn.allnan(arr_tmp["data"], axis=0)
        valid_obs = np.sum(np.isfinite(arr_tmp["data"]), axis=0)
        # replace NaN with maximum
        max_val = np.nanmax(arr_tmp["data"]) + 1
        arr_tmp["data"][np.isnan(arr_tmp["data"])] = max_val
        # sort - former NaNs will move to the end
        arr_tmp = np.sort(arr_tmp, kind="mergesort", order="data", axis=0)
        arr_tmp["data"] = np.where(arr_tmp["data"]==max_val, np.nan, arr_tmp["data"])

        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        k_arr = np.where(k_arr < 0, 0, k_arr)
        f_arr = np.floor(k_arr + 0.5)
        f_arr = f_arr.astype(np.int)

        # get floor value of reference band and index band
        floor_val = self._zvalue_from_index_ogrid(arr=arr_tmp, ind=f_arr.astype("int16"))
        idx = np.where(valid_obs==0, 255, floor_val["ID"])

        quant_arr = floor_val["data"]

        del arr_tmp

        return (quant_arr, idx, valid_obs)

    def execute(self, eopatch):
        # Dictionary connecting sceneIDs and composite index number
        self.data_layers = eopatch.data[self.layer]
        self.sceneLookup = {}
        for j,f in enumerate(self.data_layers):
            self.sceneLookup[j] = f

        # Parameters indicating if
        self.indexScenes = []
        
        # Stack scenes with data and scene index number
        dims = (self.data_layers.shape[0], self.data_layers.shape[1], self.data_layers.shape[2])
        refStack = np.zeros(dims, dtype=[("data", "f4"),("ID", "u1")])
        refStack["ID"] = 255

        # Reference bands
        for j, sce in enumerate(self.data_layers): # for each scene/time

            # Read in bands depending on composite method
            if self.method == "blue":
                ref = sce[:,:,0].astype("float32")
            elif self.method == "maxNDVI":
                nir = sce[:,:,7].astype("float32")
                red = sce[:,:,2].astype("float32")
                ref = (nir - red) / (nir + red)
                del nir, red
            elif self.method == "HOT":
                blue = sce[:,:,0].astype("float32")
                red = sce[:,:,2].astype("float32")
                ref = blue - 0.5*red - 800.
                del blue, red
            elif self.method == "maxRatio":
                blue = sce[:,:,0].astype("float32")
                nir = sce[:,:,6].astype("float32")
                swir1 = sce[:,:,8].astype("float32")
                ref = np.nanmax(np.array([nir, swir1]), axis=0) / blue
                del blue, nir, swir1
            elif self.method == "maxNDWI":
                nir = sce[:,:,6].astype("float32")
                swir1 = sce[:,:,8].astype("float32")
                ref = (nir - swir1) / (nir + swir1)
                del nir, swir1
            else:
                print("Method not valid.")
                return None
            

            # Write data to stack
            refStack["data"][j] = ref
            refStack["ID"][j] = j
            del ref
            
            
        # Calculate composite index (which scene is used for the composite per pixel)
        if self.method == "blue":
            index, valid_obs = self.indexByPercentile(refStack, self.percNo)[1:3]
        elif self.method == "maxNDVI":
            median = bn.nanmedian(refStack["data"], axis=0)
            index_max, valid_obs = self.indexByPercentile(refStack, 100)[1:3]
            index_min, valid_obs = self.indexByPercentile(refStack, 0)[1:3]
            index = np.where(median < -0.05, index_min, index_max)
        elif self.method == "HOT":
            index, valid_obs = self.indexByPercentile(refStack, 25)[1:3]
        elif self.method == "maxRatio":
            index, valid_obs = self.indexByPercentile(refStack, 100)[1:3]
        elif self.method == "maxNDWI":
            index, valid_obs = self.indexByPercentile(refStack, 100)[1:3]
        else:
            print("Method unknown")

        del refStack, valid_obs
                            
        try:
            self.indexScenes = [self.sceneLookup[idx] for idx in np.unique(index) if idx != 255]
        except:
            print("ID not in sceneLookup.")
            sys.exit(1)
            
        # Create Output
        compositeImage = np.empty(shape=(self.data_layers.shape[1], self.data_layers.shape[2], self.data_layers.shape[3]), dtype="float32")
        compositeImage[:] = np.nan
        for j, sce in zip(range(len(self.data_layers)), self.indexScenes):
            compositeImage = np.where(np.dstack([index]) == j, sce, compositeImage)
                
        compositeImage = np.where(np.isnan(compositeImage), -32768, compositeImage)
                    
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'reference_composite', compositeImage)
        
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
        self.sourceLayer = source
        self.referenceLayer = reference
        
    def stats(self, sce, sce_id, b):

        sourceBand = np.where(np.isnan(self.reference_scene[:,:,b]), np.nan, sce[:,:,b])
        referenceBand = np.where(np.isnan(sce[:,:,b]), np.nan, self.reference_scene[:,:,b])

        std_src = np.nanstd(sourceBand)
        std_ref = np.nanstd(referenceBand)
        mean_src = np.nanmean(sourceBand)
        mean_ref = np.nanmean(referenceBand)
        
        out = sce[:,:,b] * (std_ref / std_src) + (mean_ref - (mean_src * (std_ref / std_src)))
        return out
        
    def execute(self, eopatch):  
        
        source_scenes = eopatch.data[self.sourceLayer]
        self.reference_scene = eopatch.data_timeless[self.referenceLayer]
        
        eopatch.add_feature(FeatureType.DATA, 'radiometric_normalized', np.zeros(source_scenes.shape))
        for sce_id, sce in enumerate(list(source_scenes)):
            for b in range(source_scenes[0].shape[2]):
                eopatch.data['radiometric_normalized'][sce_id,:,:,b] = self.stats(sce, sce_id, b)
                        
        return eopatch
