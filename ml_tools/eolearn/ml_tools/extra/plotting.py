"""
The module provides some utility functions for ML specific plotting

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import itertools
from typing import List, Optional, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap
except ImportError as exception:
    raise ImportError("This module requires an installation of matplotlib package") from exception


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    classes: List[str],
    normalize: bool = True,
    title: str = "Confusion matrix",
    cmap: Optional[Union[str, Colormap]] = plt.cm.Blues,
    xlabel: str = "Predicted label",
    ylabel: str = "True label",
) -> None:
    """Make a single confusion matrix plot."""
    if normalize:
        normalisation_factor = confusion_matrix.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps
        confusion_matrix = confusion_matrix.astype(float) / normalisation_factor

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = ".2f" if normalize else "d"
    threshold = confusion_matrix.max() / 2.0
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(
            j,
            i,
            format(confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            fontsize=12,
            color="white" if confusion_matrix[i, j] > threshold else "black",
        )

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
