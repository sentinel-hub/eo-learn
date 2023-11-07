"""
A collection of bands extraction EOTasks

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np

from eolearn.core import MapFeatureTask
from eolearn.core.types import Feature


class NormalizedDifferenceIndexTask(MapFeatureTask):
    """The task calculates a Normalized Difference Index (NDI) between two bands A and B as:

    :math:`NDI = \\dfrac{A-B+c}{A+B+c}`,

    where c is provided as the *acorvi_constant* argument. For the reasoning behind using the acorvi_constant in the
    equation, check the article `Using NDVI with atmospherically corrected data
    <http://www.cesbio.ups-tlse.fr/multitemp/?p=12746>`_.
    """

    def __init__(
        self,
        input_feature: Feature,
        output_feature: Feature,
        bands: tuple[int, int],
        acorvi_constant: float = 0,
        undefined_value: float = np.nan,
    ):
        """
        :param input_feature: A source feature from which to take the bands.
        :param output_feature: An output feature to which to write the NDI.
        :param bands: A list of bands from which to calculate the NDI.
        :param acorvi_constant: A constant to be used in the NDI calculation. It is set to 0 by default.
        :param undefined_value: A value to override any calculation result that is not a finite value (e.g.: inf, nan).
        """
        super().__init__(input_feature, output_feature)

        if not isinstance(bands, (list, tuple)) or len(bands) != 2 or not all(isinstance(x, int) for x in bands):
            raise ValueError("bands argument should be a list or tuple of two integers!")

        self.band_a, self.band_b = bands
        self.undefined_value = undefined_value
        self.acorvi_constant = acorvi_constant

    def map_method(self, feature: np.ndarray) -> np.ndarray:
        """
        :param feature: An eopatch on which to calculate the NDI.
        """
        band_a, band_b = feature[..., self.band_a], feature[..., self.band_b]

        with np.errstate(divide="ignore", invalid="ignore"):
            ndi = (band_a - band_b + self.acorvi_constant) / (band_a + band_b + self.acorvi_constant)

        ndi[~np.isfinite(ndi)] = self.undefined_value

        return ndi[..., np.newaxis]
