"""
A collection of bands extraction EOTasks

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import numpy as np

from eolearn.core import MapFeatureTask


class EuclideanNormTask(MapFeatureTask):
    """ The task calculates the Euclidean Norm:

        :math:`Norm = \\sqrt{\\sum_{i} B_i^2}`

    where :math:`B_i` are the individual bands within a user-specified feature array.
    """

    def __init__(self, input_feature, output_feature, bands=None):
        """
        :param input_feature: A source feature from which to take the subset of bands.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature to which to write the euclidean norm.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param bands: A list of bands from which to extract the euclidean norm. If None, all bands are taken.
        :type bands: list
        """
        super().__init__(input_feature, output_feature)
        self.bands = bands

    def map_method(self, feature):
        """
        :param feature: An eopatch on which to calculate the euclidean norm.
        :type feature: numpy.array
        """
        array = feature if not self.bands else feature[..., self.bands]
        return np.sqrt(np.sum(array**2, axis=-1))[..., np.newaxis]


class NormalizedDifferenceIndexTask(MapFeatureTask):
    """ The task calculates a Normalized Difference Index (NDI) between two bands A and B as:

        :math:`NDI = \\dfrac{A-B+c}{A+B+c}`,

        where c is provided as the *acorvi_constant* argument. For the reasoning behind using the acorvi_constant in the
        equation, check the article `Using NDVI with atmospherically corrected data
        <http://www.cesbio.ups-tlse.fr/multitemp/?p=12746>`_.
    """

    def __init__(self, input_feature, output_feature, bands, acorvi_constant=0, undefined_value=np.nan):
        """
        :param input_feature: A source feature from which to take the bands.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_feature: An output feature to which to write the NDI.
        :type output_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param bands: A list of bands from which to calculate the NDI.
        :type bands: list
        :param acorvi_constant: A constant to be used in the NDI calculation. It is set to 0 by default.
        :type acorvi_constant: float
        :param undefined_value: A value to override any calculation result that is not a finite value (e.g.: inf, nan).
        """
        super().__init__(input_feature, output_feature)

        if not isinstance(bands, (list, tuple)) or len(bands) != 2 or not all(isinstance(x, int) for x in bands):
            raise ValueError('bands argument should be a list or tuple of two integers!')

        self.band_a, self.band_b = bands
        self.undefined_value = undefined_value
        self.acorvi_constant = acorvi_constant

    def map_method(self, feature):
        """
        :param feature: An eopatch on which to calculate the NDI.
        :type feature: numpy.array
        """
        band_a, band_b = feature[..., self.band_a], feature[..., self.band_b]

        with np.errstate(divide='ignore'):
            ndi = (band_a - band_b + self.acorvi_constant) / (band_a + band_b + self.acorvi_constant)

        ndi[~np.isfinite(ndi)] = self.undefined_value

        return ndi[..., np.newaxis]
