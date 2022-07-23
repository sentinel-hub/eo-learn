"""
An integration with `thunder-registration` package. Note that due to unmaintained dependencies this might not work
for Python `>=3.10`.

To use tasks from this module you have to install THUNDER package extension:

.. code-block::

    pip install eo-learn-coregistration[THUNDER]

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Domagoj Korais, Matic Lubej, Žiga Lukšič (Sinergise)
Copyright (c) 2017-2022 Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import logging

import numpy as np

try:
    import registration
except ImportError as exception:
    raise ImportError("This module requires an installation of thunder-registration package") from exception

from ..coregistration import RegistrationTask

LOGGER = logging.getLogger(__name__)


class ThunderRegistrationTask(RegistrationTask):
    """Registration task implementing a translational registration using the `thunder-registration` package"""

    def register(self, src, trg, trg_mask=None, src_mask=None):
        """Implementation of pair-wise registration using thunder-registration

        For more information on the model estimation, refer to https://github.com/thunder-project/thunder-registration
        This function takes two 2D single channel images and estimates a 2D translation that best aligns the pair. The
        estimation is done by maximising the correlation of the Fourier transforms of the images. Once, the translation
        is estimated, it is applied to the (multichannel) image to warp and, possibly, ot hte ground-truth. Different
        interpolations schemes could be more suitable for images and ground-truth values (or masks).

        :param src: 2D single channel source moving image
        :param trg: 2D single channel target reference image
        :param src_mask: Mask of source image. Not used in this method.
        :param trg_mask: Mask of target image. Not used in this method.
        :return: Estimated 2D transformation matrix of shape 2x3
        """
        # Initialise instance of CrossCorr object
        ccreg = registration.CrossCorr()
        # padding_value = 0
        # Compute translation between a pair of images
        model = ccreg.fit(src, reference=trg)
        # Get translation as an array
        translation = [-x for x in model.toarray().tolist()[0]]
        # Fill in transformation matrix
        warp_matrix = np.eye(2, 3)
        warp_matrix[0, 2] = translation[1]
        warp_matrix[1, 2] = translation[0]
        # Return transformation matrix
        return warp_matrix

    def get_params(self):
        LOGGER.info("%s: This registration does not require parameters", self.__class__.__name__)

    def check_params(self):
        pass
