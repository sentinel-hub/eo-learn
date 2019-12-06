"""
Module for validating results obtained from any ML classifier

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pickle
import itertools
from abc import ABC, abstractmethod

import numpy as np
import seaborn as sns
import pandas as pd


class SGMLBaseValidator(ABC):
    """
    Abstract class for various validations of SGML image classifiers.

    All the work is performed in the validate method, where
    for each EOPatch the following actions are performed:
        - 1. execute WOWorkflow on EOPatch
        - 2. extract ground truth (reference) from the EOPatch
            - sets self.truth_masks
            - the class values in ground truth has to the same as
              in the provided dictionary
        - 3. count truth labeled pixels
            - sets self.pixel_truth_counts
        - 4. extract classification from the EOPatch
            - sets self.classification_masks
        - 6. count classified pixels
            - sets self.pixel_classification_counts
        - 7. collect results
            - sets pixel_truth_sum and pixel_classification_sum

    Parameters:
    -----------

    workflow: EOWorkflow that is executed on EOPatches (can be simply load EOPacth)

    class_dictionary: dictionary
        Dictionary of class names and class values.

    validation_dirs: list
        List of directories containing EOPatches of this validation sample
    """

    def __init__(self, class_dictionary):
        self.class_dictionary = class_dictionary

        self.n_validation_sets = 0

        self.truth_masks = None
        self.classification_masks = None
        self.pixel_truth_counts = None
        self.pixel_classification_counts = None

        self.pixel_truth_sum = None
        self.pixel_classification_sum = None

        self.truth_classes = None

        self.val_df = None

        super().__init__()

    @abstractmethod
    def _transform_truth(self, patch):
        """
        Transform and extract the truth mask form the EOPatch and store the transformed masks in self.truth_masks.

        Parameters:
        -----------

        patch: EOPatch containing ground truth
        """

    def reset_counters(self):
        """
        Resets all counters, truth and classification masks.
        """
        self.truth_masks = None
        self.classification_masks = None
        self.pixel_truth_counts = None
        self.pixel_classification_counts = None
        self.pixel_truth_sum = None
        self.pixel_classification_sum = None
        self.val_df = None
        self.n_validation_sets = 0

    def _count_truth_pixels(self):
        """
        Count the pixels belonging to each truth class
        """
        pixel_count = np.array([[np.nonzero(mask)[0].shape[0] for mask in masktype]
                                for masktype in self.truth_masks])

        pixel_count = np.moveaxis(pixel_count, 0, -1)

        if self.pixel_truth_counts is None:
            self.pixel_truth_counts = np.copy(pixel_count)
        else:
            self.pixel_truth_counts = np.concatenate((self.pixel_truth_counts, pixel_count))

    def _count_classified_pixels(self):
        """
        Count the pixels belonging to each classified class.
        """
        class_values = self.class_dictionary.values()

        classification_count = np.array([[[np.count_nonzero(prediction[np.nonzero(mask)] == class_val)
                                           for prediction, mask in zip(self.classification_masks, masktype)]
                                          for masktype in self.truth_masks]
                                         for class_val in class_values])

        classification_count = np.moveaxis(classification_count, 0, -1)
        classification_count = np.moveaxis(classification_count, 0, -2)

        if self.pixel_classification_counts is None:
            self.pixel_classification_counts = np.copy(classification_count)
        else:
            self.pixel_classification_counts = np.concatenate((self.pixel_classification_counts, classification_count))

    @abstractmethod
    def _classify(self, patch):
        """
        Extract classification from the EOPatch.

        The classification results should be collected in self.classification_masks
        """

    def add_validation_patch(self, patch):
        """
        Extracts ground truth and classification results from the EOPatch and
        aggregates the results.
        """
        # 2. Convert 8-bit mask
        self._transform_truth(patch)

        # 3. Count truth labeled pixels
        self._count_truth_pixels()

        # 5. Perform classification
        self._classify(patch)

        # 6. Count pixel classified as class i
        self._count_classified_pixels()

        self.n_validation_sets = self.n_validation_sets + 1

    def validate(self):
        """
        Aggregate the results from all EOPatches.
        """
        self.pixel_truth_sum = np.sum(self.pixel_truth_counts, axis=0)
        self.pixel_classification_sum = np.sum(self.pixel_classification_counts, axis=0)

    def save(self, filename):
        """
        Save validator object to pickle.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    def pandas_df(self):
        """
        Returns pandas DataFrame containing pixel counts for all truth classes,
        classified classes (for each truth class), and file name of the input
        EODataSet.

        The data frame thus contains

        N =  self.n_validation_sets rows

        and

        M = len(self.truth_classes) + len(self.truth_classes) * len (self.class_dictionary) + 1 columns
        """

        if self.val_df is not None:
            return self.val_df

        clf = self.pixel_classification_counts.reshape(self.pixel_classification_counts.shape[0],
                                                       self.pixel_classification_counts.shape[1] *
                                                       self.pixel_classification_counts.shape[2])

        combo = np.hstack((self.pixel_truth_counts, clf))

        columns = list(itertools.product(self.truth_classes, list(self.class_dictionary.keys())))
        columns = [(item[0] + '_as_' + item[1]).replace(" ", "_") for item in columns]
        truth_columns = ['truth_' + item.replace(" ", "_") for item in self.truth_classes]

        self.val_df = pd.DataFrame(combo, columns=truth_columns + columns)

        return self.val_df

    def to_csv(self, filename):
        """
        Writes validation results using pandas to csv file.
        """
        self.pandas_df().to_csv(filename)

    def confusion_matrix(self):
        """
        Returns the normalised confusion matrix
        """
        confusion_matrix = self.pixel_classification_sum.astype(np.float)
        confusion_matrix = np.divide(confusion_matrix.T, self.pixel_truth_sum.T).T

        return confusion_matrix * 100.0

    def plot_confusion_matrix(self, normalised=True):
        """
        Plots the confusion matrix.
        """
        conf_matrix = self.confusion_matrix()

        if normalised:
            sns.heatmap(conf_matrix,
                        annot=True, annot_kws={"size": 12}, fmt='2.1f', cmap='YlGnBu', vmin=0.0,
                        vmax=100.0,
                        xticklabels=list(self.class_dictionary.keys()),
                        yticklabels=self.truth_classes)
        else:
            sns.heatmap(self.pixel_classification_counts,
                        annot=True, annot_kws={"size": 12}, fmt='2.1f', cmap='YlGnBu', vmin=0.0,
                        vmax=np.max(self.pixel_classification_counts),
                        xticklabels=list(self.class_dictionary.keys()),
                        yticklabels=self.truth_classes)

    def summary(self, scoring):
        """
        Prints out the summary of validation for giving scoring function.
        """

        if scoring == 'class_confusion':
            print('*' * 50)
            print('  Confusion Matrix ')
            print('x-axis: ' + ' | '.join(list(self.class_dictionary.keys())))
            print('y-axis: ' + ' | '.join(self.truth_classes))
            print(self.confusion_matrix())
