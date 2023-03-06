"""
Tests for `EOPatch` visualizations

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import os

import numpy as np
import pytest
from matplotlib.pyplot import Axes

from eolearn.core import EOPatch, FeatureType
from eolearn.visualization import PlotConfig


@pytest.fixture(name="eopatch", scope="module")
def eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "example_data", "TestEOPatch")
    return EOPatch.load(path)


@pytest.mark.parametrize(
    "feature, params",
    [
        ((FeatureType.DATA, "BANDS-S2-L1C"), {"rgb": [3, 2, 1]}),
        ((FeatureType.DATA, "BANDS-S2-L1C"), {"times": [7, 14, 67], "channels": slice(4, 8)}),
        ((FeatureType.MASK, "CLM"), {}),
        ((FeatureType.MASK, "IS_VALID"), {}),
        ((FeatureType.SCALAR, "CLOUD_COVERAGE"), {"times": [5, 8, 13, 21]}),
        ((FeatureType.SCALAR, "CLOUD_COVERAGE"), {"times": slice(20, 25)}),
        ((FeatureType.LABEL, "IS_CLOUDLESS"), {"channels": [0]}),
        ((FeatureType.LABEL, "RANDOM_DIGIT"), {}),
        ((FeatureType.VECTOR, "CLM_VECTOR"), {}),
        ((FeatureType.VECTOR, "CLM_VECTOR"), {"channels": slice(10, 20)}),
        ((FeatureType.VECTOR, "CLM_VECTOR"), {"channels": [0, 5, 8]}),
        ((FeatureType.DATA_TIMELESS, "DEM"), {}),
        ((FeatureType.MASK_TIMELESS, "RANDOM_UINT8"), {}),
        ((FeatureType.SCALAR_TIMELESS, "LULC_PERCENTAGE"), {}),
        ((FeatureType.LABEL_TIMELESS, "LULC_COUNTS"), {}),
        ((FeatureType.LABEL_TIMELESS, "LULC_COUNTS"), {"channels": [0]}),
        ((FeatureType.VECTOR_TIMELESS, "LULC"), {}),
        (FeatureType.BBOX, {}),
    ],
)
@pytest.mark.sh_integration  # python 3.7 dose not support matpotlib 3.6
def test_eopatch_plot(eopatch, feature, params):
    """A simple test of EOPatch plotting for different features."""
    # We reduce width and height otherwise running matplotlib.pyplot.subplots in combination with pytest would
    # kill the Python kernel.
    config = PlotConfig(subplot_width=1, subplot_height=1)

    axes = eopatch.plot(feature, config=config, **params)
    axes = axes.flatten()

    assert isinstance(axes, np.ndarray)
    for item in axes:
        assert isinstance(item, Axes)
