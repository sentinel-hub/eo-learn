# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:36:12 2019

@author: willing
"""

# Imports
from eolearn.core import EOTask, FeatureType, EOPatch
from eolearn.geometry import PointSamplingTask
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentinelhub import CRS, GeopediaFeatureIterator, GeopediaSession
from skimage.morphology import disk, binary_dilation, binary_erosion
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd

import os
import enum
import numpy as np
import matplotlib.pyplot as plt
import itertools

###########################################################################


# Define Plot characteristics of classes
class LPISCLASS(enum.Enum):
    NO_DATA = (0, 'No Data', 'white')
    Beets = (1, 'Beets', 'orange')
    Meadows = (2, 'Meadows', 'black')
    Fallow_land = (3, 'Fallow land', 'xkcd:azure')
    Peas = (4, 'Peas', 'xkcd:salmon')
    Pasture = (5, 'Pasture', 'xkcd:navy')
    Hop = (6, 'Hop', 'xkcd:lavender')
    Grass = (7, 'Grass', 'xkcd:lightblue')
    Poppy = (8, 'Poppy', 'xkcd:brown')
    Winter_rape = (9, 'Winter rape', 'xkcd:shit')
    Maize = (10, 'Maize', 'xkcd:beige')
    Winter_cereals = (11, 'Winter cereals', 'xkcd:apricot')
    LL_ao_GM = (12, 'LL and/or GM', 'crimson')
    Pumpkins = (13, 'Pumpkins', 'lightgrey')
    Soft_fruit = (14, 'Soft fruit', 'firebrick')
    Summer_cereals = (15, 'Summer cereals', 'xkcd:grey')
    Sun_flower = (16, 'Sun flower', 'xkcd:jade')
    Vegetables = (17, 'Vegetables', 'xkcd:ultramarine')
    Buckwheat = (18, 'Buckwheat', 'xkcd:tan')
    Alpine_Meadows = (19, 'Alpine meadows', 'xkcd:lime')
    Potatoes = (20, 'Potatoes', 'pink')
    Beans = (21, 'Beans', 'xkcd:darkgreen')
    Vineyards = (22, 'Vineyards', 'magenta')
    Other = (23, 'Other', 'xkcd:gold')
    Soybean = (24, 'Soybean', 'xkcd:clay')
    Orchards = (25, 'Orchards', 'olivedrab')
    Multi_use = (26, 'Multi use', 'orangered')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

class AddLPISLayerFromLocal(EOTask):
    """
    Add vector data from local LPIS database.
    """

    def __init__(self, feature, LPIS_path):
        self.feature_type, self.feature_name = next(self._parse_features(feature)())
        self.LPIS_path = LPIS_path

    def execute(self, eopatch):
        # load data from LPIS file, intersecting with bbox
        gdf = gpd.read_file(self.LPIS_path, bbox=tuple(eopatch.bbox))
        eopatch[self.feature_type][self.feature_name] = gdf
        return eopatch

# Task for 2.2 Prepare LPIS data
class GroupLPIS(EOTask):
    """
        Task to group the LPIS data into wanted groups
    """

    def __init__(self, year, lpis_to_group_file, crop_group_file):
        self.year = year
        self.lpis_to_group_file = lpis_to_group_file
        self.crop_group_file = crop_group_file

    def execute(self, eopatch, col_cropN_lpis, col_cropN_lpistogroup):
        """
        Returns the eopatch with the new grouping of the LPIS data. A column "GROUP_1_ID",
        is also added, with the ID associated to the groups.

        col_cropN_lpis is the name of the column of the crop type in the lpis dataframe.
        col_cropN_lpistogroup is the name of the column of the crop type in the csv file
        specified by self.lpis_to_group_file.
        """
        # Group LPIS classes
        lpis = eopatch.vector_timeless["LPIS_{}".format(self.year)]
        mapping = pd.read_csv(self.lpis_to_group_file, sep=";")
        result = pd.merge(lpis, mapping, how="left", left_on=[col_cropN_lpis], right_on=[col_cropN_lpistogroup])

        # Assign GroupID to GroupName
        group_id = pd.read_csv(self.crop_group_file, sep=";")
        resultend = pd.merge(result, group_id, how="left", on="GROUP_1")
        eopatch.vector_timeless["LPIS_{}".format(self.year)] = resultend

        # Fill GroupID NaN values with zeros
        group = eopatch.vector_timeless["LPIS_{}".format(self.year)]["GROUP_1_ID"]
        eopatch.vector_timeless["LPIS_{}".format(self.year)]["GROUP_1_ID"] = group.fillna(0)

        return eopatch


# Tasks for EOPatch preparation
class ConcatenateData(EOTask):
    """
        Task to concatenate data arrays along the last dimension
    """

    def __init__(self, feature_name, feature_names_to_concatenate):
        self.feature_name = feature_name
        self.feature_names_to_concatenate = feature_names_to_concatenate

    def execute(self, eopatch):
        arrays = [eopatch.data[name] for name in self.feature_names_to_concatenate]

        eopatch.add_feature(FeatureType.DATA, self.feature_name, np.concatenate(arrays, axis=-1))

        return eopatch

class CleanLPIS(EOTask):
    """
        Task to delete columns ['SNAR_BEZEI_NAME', "CROP_ID", "english", "slovenian", "latin", "GROUP_1", "GROUP_1_ID"]
        from vector dataset to enable LPIS regrouping
    """

    def __init__(self, year):
        self.year = year

    def execute(self, eopatch):
        lpis = eopatch.vector_timeless["LPIS_{}".format(self.year)]

        lpis.drop(columns=["SNAR_BEZEI_NAME", "CROP_ID", "english", "slovenian", "latin", "GROUP_1", "GROUP_1_ID"], axis=1, inplace=True)

        eopatch.vector_timeless["LPIS_{}".format(self.year)] = lpis

        return eopatch

class SamplingTaskTask(EOTask):
    """
        Adapted PointSamplingTask for customizing 'n_samples' and 'ref_labels' for each specific EOPatch
    """
    def __init__(self, grouping_id, pixel_thres, samp_class):
        self.grouping_id = grouping_id
        self.pixel_thres = pixel_thres
        self.samp_class = samp_class

    def execute(self, eopatch):

        classes = eopatch.mask_timeless['LPIS_class_{}_ERODED'.format(self.grouping_id)]
        w,h,c = classes.shape
        classes = classes.reshape(w * h, 1).squeeze()
        unique, counts = np.unique(classes, return_counts=True)
        classcount = dict(zip(unique, counts))

        ref_labels = []

        for i in classcount:
            if i != 0 and classcount[i] > self.pixel_thres:
                ref_labels.append(i)

        n_samples = len(ref_labels) * self.samp_class

        # TASK FOR SPATIAL SAMPLING
        # evenly sample about pixels from patches
        spatial_sampling = PointSamplingTask(
            even_sampling = True,
            n_samples=n_samples,
            ref_mask_feature='LPIS_class_{}_ERODED'.format(self.grouping_id),
            ref_labels=ref_labels,
            sample_features=[  # tag fields to sample
                (FeatureType.DATA, 'FEATURES', 'FEATURES_SAMPLED'), # feature dicts to sample and where to save samples
                (FeatureType.MASK_TIMELESS,
                 'LPIS_class_{}_ERODED'.format(self.grouping_id), # label dict to sample
                 'LPIS_class_{}_ERODED_SAMPLED'.format(self.grouping_id)) # label dict to save samples
            ])

        spatial_sampling.execute(eopatch)

        return eopatch

# Utility function for creating a list of all eopatches found in an Output folder
def get_patch_list(folder):
    """
    Returns a list with EOPatch names found in the provided folder. 
    """
    return os.listdir(folder)
    
# Function to split dataset into training and test data for multiple EOPatches
def train_test_split_eopatches(patch_array, test_ratio, features_dict, labels_dict):
    """
    Split dataset into train and test data
    """
    # define EOPatches for training and testing
    trainIDs = list(range(len(patch_array)))
    testIDs = trainIDs[0::test_ratio]  # take every xth patch for testing

    for elem in trainIDs:
        if elem in testIDs:
            trainIDs.remove(elem)

    # get number of features
    t, w, h, f = patch_array[0].data[features_dict].shape
    timeframes_count = t
    features_count = f
    f_count = t * f


    # create training and test dataset
    features_train = np.zeros([0, f_count])
    for eopatch in patch_array[trainIDs]:
        addfeatures_train = np.array([eopatch.data[features_dict]])
        p, t, w, h, f = addfeatures_train.shape
        addfeatures_train = np.moveaxis(addfeatures_train, 1, 3).reshape(p * w * h, t * f)
        features_train = np.concatenate((features_train, addfeatures_train))

    features_test = np.zeros([0, f_count])
    for eopatch in patch_array[testIDs]:
        addfeatures_test = np.array([eopatch.data[features_dict]])
        p, t, w, h, f = addfeatures_test.shape
        addfeatures_test = np.moveaxis(addfeatures_test, 1, 3).reshape(p * w * h, t * f)
        features_test = np.concatenate((features_test, addfeatures_test))

    labels_train = np.zeros([0, ])
    for eopatch in patch_array[trainIDs]:
        addlabels_train = np.array([eopatch.mask_timeless[labels_dict]])
        p, w, h, f = addlabels_train.shape
        addlabels_train = np.moveaxis(addlabels_train, 1, 2).reshape(p * w * h, 1).squeeze()
        labels_train = np.concatenate((labels_train, addlabels_train))

    labels_test = np.zeros([0, ])
    for eopatch in patch_array[testIDs]:
        addlabels_test = np.array([eopatch.mask_timeless[labels_dict]])
        p, w, h, f = addlabels_test.shape
        addlabels_test = np.moveaxis(addlabels_test, 1, 2).reshape(p * w * h, 1).squeeze()
        labels_test = np.concatenate((labels_test, addlabels_test))

    return features_train, features_test, labels_train, labels_test, timeframes_count, features_count

# Function to split dataset into training and test data for multiple EOPatches
def train_test_split_eopatch(patch_array, features_dict, labels_dict):
    """
    Split dataset into train and test data
    """
    # Set the features and the labels for train and test sets
    for eopatch in patch_array:
        features = np.array([eopatch.data[features_dict]])

    for eopatch in patch_array:
        labels = np.array([eopatch.mask_timeless[labels_dict]])

    # get shape
    p, t, w, h, f = features.shape

    timeframes_count = t
    features_count = f

    # reshape to n x m
    features = np.moveaxis(features, 1, 3).reshape(w * h, t * f)
    labels = np.moveaxis(labels, 1, 2).reshape(w * h, 1).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test, timeframes_count, features_count

# Function to mask out labels that are not in both train and test data and also mask out samples where features include NaN values
def masking(X_train, X_test, y_train, y_test):
    """
    mask out labels that are not in both train and test data and also mask out samples where features include NaN values
    """
    # create mask to exclude NaN-values from train data
    mask_train = np.zeros(X_train.shape[0], dtype=np.bool)

    for i, subfeat in enumerate(X_train):
        if True in np.isnan(subfeat):
            mask_train[i] = True
        else:
            mask_train[i] = False

    # create mask to exclude NaN-values from test data
    mask_test = np.zeros(X_test.shape[0], dtype=np.bool)

    for i, subfeat in enumerate(X_test):
        if True in np.isnan(subfeat):
            mask_test[i] = True
        else:
            mask_test[i] = False

    # masking
    X_train = X_train[~mask_train]
    y_train = y_train[~mask_train]

    X_test = X_test[~mask_test]
    y_test = y_test[~mask_test]

    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    # exclude classes that are not included in both, test and train data
    difflist1 = list(set(np.unique(y_train)) - set(np.unique(y_test)))

    for i in difflist1:
        mask_train = y_train == i
        X_train = X_train[~mask_train]
        y_train = y_train[~mask_train]

    difflist2 = list(set(np.unique(y_test)) - set(np.unique(y_train)))

    for i in difflist2:
        mask_test = y_test == i
        X_test = X_test[~mask_test]
        y_test = y_test[~mask_test]

    return(X_train, X_test, y_train, y_test)


# Function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, ylabel='True label', xlabel='Predicted label', filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2, suppress=True)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(np.float).eps)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=20)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="lightgrey" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylim(top=-0.5)
    plt.ylim(bottom=len(classes)-0.5)

# Task for predicting a whole EOPatch
class PredictPatch(EOTask):
    """
    Task to make model predictions on a patch. Provide the model and the feature,
    and the output names of labels and scores (optional)
    """

    def __init__(self, model, features_feature, predicted_labels_name, scaler):
        self.model = model
        self.features_feature = features_feature
        self.predicted_labels_name = predicted_labels_name
        self.scaler = scaler

    def execute(self, eopatch):
        ftrs = eopatch[self.features_feature[0]][self.features_feature[1]]
        t, w, h, f = ftrs.shape
        ftrs = np.moveaxis(ftrs, 0, 2).reshape(w * h, t * f)
        scaled_ftrs = self.scaler.transform(ftrs)

        try:  # LightGBM model prediction
            plabels = self.model.predict(scaled_ftrs)
            plabels = plabels.reshape(w, h)
            plabels = plabels[..., np.newaxis]
            eopatch.add_feature(FeatureType.MASK_TIMELESS, self.predicted_labels_name, plabels)

        except:  # TempCNN model prediction
            ftrs_tcnn = np.reshape(scaled_ftrs,
                                   (-1, eopatch.data['FEATURES'].shape[0], eopatch.data['FEATURES'].shape[3]))
            plabels = np.argmax(self.model.predict(ftrs_tcnn), axis=-1)
            plabels = plabels.reshape(w, h)
            plabels = plabels[..., np.newaxis]
            eopatch.add_feature(FeatureType.MASK_TIMELESS, self.predicted_labels_name, plabels)

        return eopatch


class AddAreaRatio(EOTask):
    """
    Calculates the ratio between

    area of all fields (vector data) / total area of the patch.

    This information can be used for example to exclude EOPatches with no or very small area of cultivated land.
    """

    def __init__(self, vector_feature, area_feature):
        self.in_feature_type, self.in_feature_name = next(self._parse_features(vector_feature)())
        self.out_feature_type, self.out_feature_name = next(self._parse_features(area_feature)())

    def execute(self, eopatch):
        ratio = np.array([-1.0])
        if self.in_feature_name not in eopatch[self.in_feature_type]:
            eopatch[self.out_feature_type][self.out_feature_name] = ratio
            return eopatch

        gdf = eopatch[self.in_feature_type][self.in_feature_name]
        ratio = np.array([0.0])
        if gdf is not None:
            bbox_poly = Polygon(eopatch.bbox.get_polygon())
            ratio = np.array([np.sum(gdf.area.values) / bbox_poly.area])

        eopatch[self.out_feature_type][self.out_feature_name] = ratio

        return eopatch

    
def get_crop_features(table_id):
    """
    Returns DataFrame of crops for table_id from Geopedia

    :return: pandas DataFrame 
    :rtype: pandas.DataFrame
    """
    gpd_session = GeopediaSession()
    crop_iterator = GeopediaFeatureIterator(layer=table_id, gpd_session=gpd_session)
    to_crop_id = [{'crop_geopedia_idx': code['id'], **code['properties']} for code in crop_iterator]

    df = pd.DataFrame(to_crop_id)
    df['crop_geopedia_idx'] = pd.to_numeric(df.crop_geopedia_idx)
    
    return df
    
    
# FixLPIS utilties
def get_slovenia_crop_geopedia_idx_to_crop_id_mapping():
    """
    Returns mapping between Geopedia's crop index and crop id for Slovenia.

    :return: pandas DataFrame with 'crop_geopedia_idx' and corresponding crop id
    :rtype: pandas.DataFrame
    """
    return get_crop_features(2036)


def get_austria_crop_geopedia_idx_to_crop_id_mapping():
    """
    Returns mapping between Geopedia's crop index and crop id for Austria.

    :return: pandas DataFrame with 'crop_geopedia_idx' and corresponding crop id
    :rtype: pandas.DataFrame
    """
    to_crop_id = get_crop_features(2032)
    to_crop_id.rename(index=str, columns={"SNAR_BEZEI": "SNAR_BEZEI_NAME"}, inplace=True)
    to_crop_id.rename(index=str, columns={"crop_geopedia_idx": "SNAR_BEZEI"}, inplace=True)

    return to_crop_id


def get_danish_crop_geopedia_idx_to_crop_id_mapping():
    """
    Returns mapping between Geopedia's crop index and crop id for Austria.

    :return: pandas DataFrame with 'crop_geopedia_idx' and corresponding crop id
    :rtype: pandas.DataFrame
    """
    return get_crop_features(2050)


class FixLPIS(EOTask):
    """
    Fixes known issues of LPIS data stored as vector_timeless feature in the EOPatch.

    Known issues depend on the country and are:
    * Slovenia:
        * column "SIFRA_KMRS" of vector_timeless["LPIS_{year}"] represents index in geopedia's
          table "Crop type classification for Slovenia" and not CROP ID as the name suggests
            * This task replaces "SIFRA_KMRS" with "SIFKMRS" that truly represents CROP ID
        * CROP IDs are strings and not integers, which represents a problem when burning in the
          LPIS data to raster.
            * This task replaces "204_a" with "1204"
            * column is casted to numeric
    * Austria:
        * column "SNAR_BEZEI" of vector_timeless["LPIS_{year}"] represents index in geopedia's
          table "Austria LPIS (SNAR_BEZEI)" and not CROP NAME as the name suggests
        * a new column is added "SNAR_BEZEI_NAME" with the CROP NAME as appears in Austrian LPIS data
    * Denmark:
        * columns "CropName" and "PreCropName" of vector_timeless["LPIS_{year}"] represents index in geopedia's
          table "DK LPIS crop type" and not CROP NAME as the name suggests
        * they are replaced with two new columns "Crop Name" and "PreCrop Name" with the CROP NAME as
          appears in Danish LPIS data

    :param feature: Name of the vector_timeless feature with LPIS data
    :type feature: str
    :param country: Name of the country
    :type country: str
    """
    def __init__(self, feature, country):
        self.feature = feature
        self.country = country
        self.mapping = None

        self._set_mapping()

    def _set_mapping(self):
        if self.country == 'Slovenia':
            self.mapping = get_slovenia_crop_geopedia_idx_to_crop_id_mapping()
        elif self.country == 'Austria':
            self.mapping = get_austria_crop_geopedia_idx_to_crop_id_mapping()
        elif self.country == 'Denmark':
            self.mapping = get_danish_crop_geopedia_idx_to_crop_id_mapping()

    def _fix_slovenian_lpis(self, eopatch):
        """
        See Task's docs for the explanation of what is done.
        """
        eopatch.vector_timeless[self.feature].rename(index=str, columns={"SIFRA_KMRS": "crop_geopedia_idx"},
                                                    inplace=True)
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='crop_geopedia_idx')
        eopatch.vector_timeless[self.feature].loc[eopatch.vector_timeless[self.feature]['SIFKMRS'] == '204_a',
                                                  'SIFKMRS'] = '1204'
        eopatch.vector_timeless[self.feature]['SIFKMRS'] = pd.to_numeric(eopatch.vector_timeless[self.feature]['SIFKMRS'])


    def _fix_austrian_lpis(self, eopatch):
        """
        See Task's docs for the explanation of what is done.
        """
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='SNAR_BEZEI')

    def _fix_danish_lpis(self, eopatch):
        """
        See Task's docs for the explanation of what is done.
        """
        eopatch.vector_timeless[self.feature].rename(index=str, columns={"CropName": "crop_geopedia_idx"}, inplace=True)
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='crop_geopedia_idx')
        eopatch.vector_timeless[self.feature]['crop_geopedia_idx'] = eopatch.vector_timeless[self.feature]['PreCropName']
        self.mapping.rename(index=str, columns={"Crop Name": "PreCrop Name"}, inplace=True)
        eopatch.vector_timeless[self.feature] = pd.merge(eopatch.vector_timeless[self.feature],
                                                         self.mapping,
                                                         on='crop_geopedia_idx')
        eopatch.vector_timeless[self.feature].drop(['crop_geopedia_idx', 'PreCropName'], axis=1, inplace=True)

    def execute(self, eopatch):
        if self.country == 'Slovenia':
            self._fix_slovenian_lpis(eopatch)
        elif self.country == 'Austria':
            self._fix_austrian_lpis(eopatch)
        elif self.country == 'Denmark':
            self._fix_danish_lpis(eopatch)

        return eopatch


class ValidDataFractionPredicate:
    """
    Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold

class Sen2CorValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a valid data mask.
    The valid data mask is post-processed (optional).

    The Sen2Cor's classification map is asumed to be found in eopatch.mask['SCL']
    """

    def __init__(self, valid_classes, erosion_radius=0, dilation_radius=0):
        self.valid = valid_classes
        self.erosion = erosion_radius
        self.dilation = dilation_radius

    def __call__(self, eopatch):
        sen2cor_valid = np.zeros_like(eopatch.mask['SCL'], dtype=np.bool)

        for valid in self.valid:
            sen2cor_valid = np.logical_or(sen2cor_valid, (eopatch.mask['SCL'] == valid))

        sen2cor_valid = sen2cor_valid.squeeze()
        if self.erosion:
            sen2cor_valid = np.logical_not(
                np.asarray([binary_erosion(np.logical_not(mask), disk(self.erosion)) for mask in sen2cor_valid],
                           dtype=np.bool))

        if self.dilation:
            sen2cor_valid = np.logical_not(
                np.asarray([binary_dilation(np.logical_not(mask), disk(self.dilation)) for mask in sen2cor_valid],
                           dtype=np.bool))

        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool), sen2cor_valid[..., np.newaxis])


class AddGeopediaVectorFeature(EOTask):
    """
    Add vector data from Geopedia.
    """

    def __init__(self, feature, layer, year_filter=None, drop_duplicates=False):
        self.feature_type, self.feature_name = next(self._parse_features(feature)())
        self.layer = layer
        self.drop_duplicates = drop_duplicates
        self.year_col_name = year_filter[0] if year_filter is not None else None
        self.year = year_filter[1] if year_filter is not None else None

    def execute(self, eopatch):
        # convert to 3857 CRS
        bbox_3857 = eopatch.bbox.transform(CRS.POP_WEB)

        # get iterator over features
        gpd_iter = GeopediaFeatureIterator(layer=self.layer, bbox=bbox_3857)

        features = list(gpd_iter)
        if len(features):
            gdf = gpd.GeoDataFrame.from_features(features)
            gdf.crs = CRS.WGS84.pyproj_crs()
            # convert back to EOPatch CRS
            gdf = gdf.to_crs(eopatch.bbox.crs.pyproj_crs())

            if self.year:
                # Filter by years
                gdf = gdf.loc[gdf[self.year_col_name].isin([self.year])]

            if self.drop_duplicates:
                sel = gdf.drop('geometry', axis=1)
                sel = sel.drop_duplicates()
                gdf = gdf.loc[sel.index]

            eopatch[self.feature_type][self.feature_name] = gdf

        return eopatch
