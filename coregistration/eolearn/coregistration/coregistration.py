"""
This module implements the co-registration transformers.

Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
import copy

from abc import ABC, abstractmethod
from enum import Enum
import registration
import cv2
import numpy as np

from eolearn.core import EOTask, FeatureType

from .coregistration_utilities import ransac, EstimateEulerTransformModel

LOGGER = logging.getLogger(__name__)

MAX_TRANSLATION = 20
MAX_ROTATION = np.pi / 9


class InterpolationType(Enum):
    """ Types of interpolation, available are NEAREST, LINEAR and CUBIC
    """
    NEAREST = 0
    LINEAR = 1
    CUBIC = 3


class RegistrationTask(EOTask, ABC):
    """ Abstract class for multi-temporal image co-registration

        The task uses a temporal stack of images of the same location (i.e. a temporal-spatial feature in `EOPatch`).
        Starting from the latest frame and proceeding backwards it calculates a transformation between two temporally
        adjacent images. The transformation is used to correct the earlier image to best fit the later. The reason for
        such reversed order is that the latest frames are supposed to be less affected by orthorectificational
        inaccuracies.

        Each transformation is calculated using only a single channel of the images. If feature which contains masks of
        valid pixels is specified it is used during the calculation. At the end the transformations are applied to each
        of the specified features. Any additional registration parameters can be passed on to registration method class.

        Parameters:

        :param registration_feature: A feature which will be used for co-registration,
                                        e.g. feature=(FeatureType.DATA, 'bands'). By default this feature is of type
                                        FeatureType.DATA therefore also only feature name can be given e.g.
                                        feature='bands'
        :type registration_feature: (FeatureType, str) or str
        :param channel: Index of `feature`'s channel to be used in co-registration
        :type channel: int
        :param valid_mask_feature: Feature containing a mask of valid pixels for `registration_feature`. By default no
                                    mask is set. It can be set to e.g. valid_mask_feature=(FeatureType.MASK, 'IS_DATA')
                                    or valid_mask_feature='IS_DATA' if the feature is of type FeatureType.MASK
        :type valid_mask_feature: str or (FeatureType, str) or None
        :param apply_to_features: A collection of features to which co-registration will be applied to. By default this
                                    is only `registration_feature` and `valid_mask_feature` if given. Note that each
                                    feature must have same temporal dimension as `registration_feature`.
        :type apply_to_features: dict(FeatureType: set(str) or dict(str: str))
        :param interpolation_type: Type of interpolation used. Default is `InterpolationType.CUBIC`
        :type interpolation_type: InterpolationType
        :param params: Any other registration setting which will be passed to registration method
        :type params: object
    """
    def __init__(self, registration_feature, channel=0, valid_mask_feature=None, apply_to_features=...,
                 interpolation_type=InterpolationType.CUBIC, **params):
        self.registration_feature = self._parse_features(registration_feature, default_feature_type=FeatureType.DATA)

        self.channel = channel

        self.valid_mask_feature = None if valid_mask_feature is None else \
            self._parse_features(valid_mask_feature, default_feature_type=FeatureType.MASK)

        if apply_to_features is ...:
            apply_to_features = [next(self.registration_feature())]
            if valid_mask_feature:
                apply_to_features.append(next(self.valid_mask_feature()))
        self.apply_to_features = self._parse_features(apply_to_features)

        self.interpolation_type = interpolation_type
        self.params = params

    @abstractmethod
    def register(self, src, trg, trg_mask=None, src_mask=None):
        """ Method for registration

        :param src: src
        :param trg: trg
        :param trg_mask: trg_mask
        :param src_mask: src_mask
        """
        raise NotImplementedError

    @abstractmethod
    def check_params(self):
        """ Method to validate registration parameters """
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        """ Method to print out registration parameters used """
        raise NotImplementedError

    @staticmethod
    def _get_interpolation_flag(interpolation_type):
        try:
            return {
                InterpolationType.CUBIC: cv2.INTER_CUBIC,
                InterpolationType.NEAREST: cv2.INTER_NEAREST,
                InterpolationType.LINEAR: cv2.INTER_LINEAR
            }[interpolation_type]
        except KeyError:
            raise ValueError("Unsupported interpolation method specified")

    def execute(self, eopatch):
        """ Method that estimates registrations and warps EOPatch objects
        """
        self.check_params()
        self.get_params()

        new_eopatch = copy.deepcopy(eopatch)

        f_type, f_name = next(self.registration_feature(eopatch))
        sliced_data = copy.deepcopy(eopatch[f_type][f_name][..., self.channel])
        time_frames = sliced_data.shape[0]

        iflag = self._get_interpolation_flag(self.interpolation_type)

        for idx in range(time_frames - 1, 0, -1):  # Pair-wise registration starting from the most recent frame

            src_mask, trg_mask = None, None
            if self.valid_mask_feature is not None:
                f_type, f_name = next(self.valid_mask_feature(eopatch))
                src_mask = new_eopatch[f_type][f_name][idx - 1]
                trg_mask = new_eopatch[f_type][f_name][idx]

            # Estimate transformation
            warp_matrix = self.register(sliced_data[idx - 1], sliced_data[idx], src_mask=src_mask, trg_mask=trg_mask)

            # Check amount of deformation
            rflag = self.is_registration_suspicious(warp_matrix)

            # Flag suspicious registrations and set them to the identity
            if rflag:
                LOGGER.warning("{:s} warning in pair-wise reg {:d} to {:d}".format(self.__class__.__name__, idx - 1,
                                                                                   idx))
                warp_matrix = np.eye(2, 3)

            # Transform and update sliced_data
            sliced_data[idx - 1] = self.warp(warp_matrix, sliced_data[idx - 1], iflag)

            # Apply tranformation to every given feature
            for feature_type, feature_name in self.apply_to_features(eopatch):
                new_eopatch[feature_type][feature_name][idx - 1] = \
                    self.warp(warp_matrix, new_eopatch[feature_type][feature_name][idx - 1], iflag)

        return new_eopatch

    def warp(self, warp_matrix, img, iflag=cv2.INTER_NEAREST):
        """ Function to warp input image given an estimated 2D linear transformation

        :param warp_matrix: Linear 2x3 matrix to use to linearly warp the input images
        :type warp_matrix: ndarray
        :param img: Image to be warped with estimated transformation
        :type img: ndarray
        :param iflag: Interpolation flag, specified interpolation using during resampling of warped image
        :type iflag: cv2.INTER_*
        :return: Warped image using the linear matrix
        """

        height, width = img.shape[:2]
        warped_img = np.zeros_like(img, dtype=img.dtype)

        # Check if image to warp is 2D or 3D. If 3D need to loop over channels
        if (self.interpolation_type == InterpolationType.LINEAR) or img.ndim == 2:
            warped_img = cv2.warpAffine(img.astype(np.float32), warp_matrix, (width, height),
                                        flags=iflag).astype(img.dtype)

        elif img.ndim == 3:
            for idx in range(img.shape[-1]):
                warped_img[..., idx] = cv2.warpAffine(img[..., idx].astype(np.float32), warp_matrix, (width, height),
                                                      flags=iflag).astype(img.dtype)
        else:
            raise ValueError('Image has incorrect number of dimensions: {}'.format(img.ndim))

        return warped_img

    @staticmethod
    def is_registration_suspicious(warp_matrix):
        """ Static method that check if estimated linear transformation could be unplausible

        This function checks whether the norm of the estimated translation or the rotation angle exceed predefined
        values. For the translation, a maximum translation radius of 20 pixels is flagged, while larger rotations than
        20 degrees are flagged.

        :param warp_matrix: Input linear transformation matrix
        :type warp_matrix: ndarray
        :return: 0 if registration doesn't exceed threshold, 1 otherwise
        """
        if warp_matrix is None:
            return 1

        cos_theta = np.trace(warp_matrix[:2, :2]) / 2
        rot_angle = np.arccos(cos_theta)
        transl_norm = np.linalg.norm(warp_matrix[:, 2])
        return 1 if int((rot_angle > MAX_ROTATION) or (transl_norm > MAX_TRANSLATION)) else 0


class ThunderRegistration(RegistrationTask):
    """ Registration task implementing a translational registration using the thunder-registration package
    """

    def register(self, src, trg, trg_mask=None, src_mask=None):
        """ Implementation of pair-wise registration using thunder-registration

        For more information on the model estimation, refer to https://github.com/thunder-project/thunder-registration
        This function takes two 2D single channel images and estimates a 2D translation that best aligns the pair. The
        estimation is done by maximising the correlation of the Fourier transforms of the images. Once, the translation
        is estimated, it is applied to the (multi-channel) image to warp and, possibly, ot hte ground-truth. Different
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
        # Compute translation between pair of images
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
        LOGGER.info("{:s}:This registration does not require parameters".format(self.__class__.__name__))

    def check_params(self):
        pass


class ECCRegistration(RegistrationTask):
    """ Registration task implementing an intensity-based method from OpenCV
    """

    def get_params(self):
        LOGGER.info("{:s}:Params for this registration are:".format(self.__class__.__name__))
        LOGGER.info("\t\t\t\tMaxIters: {:d}".format(self.params['MaxIters']))
        LOGGER.info("\t\t\t\tgaussFiltSize: {:d}".format(self.params['gaussFiltSize']))

    def check_params(self):
        if not isinstance(self.params.get('MaxIters'), int):
            LOGGER.info("{:s}:MaxIters set to 200".format(self.__class__.__name__))
            self.params['MaxIters'] = 200
        if not isinstance(self.params.get('gaussFilterSize'), int):
            LOGGER.info("{:s}:gaussFilterSize set to 1".format(self.__class__.__name__))
            self.params['gaussFiltSize'] = 1

    def register(self, src, trg, trg_mask=None, src_mask=None):
        """ Implementation of pair-wise registration and warping using Enhanced Correlation Coefficient

        This function estimates an Euclidean transformation (x,y translation + rotation) using the intensities of the
        pair of images to be registered. The similarity metric is a modification of the cross-correlation metric, which
        is invariant to distortions in contrast and brightness.

        :param src: 2D single channel source moving image
        :param trg: 2D single channel target reference image
        :param trg_mask: Mask of target image. Not used in this method.
        :param src_mask: Mask of source image. Not used in this method.
        :return: Estimated 2D transformation matrix of shape 2x3
        """
        # Parameters of registration
        warp_mode = cv2.MOTION_EUCLIDEAN
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.params['MaxIters'], termination_eps)
        # Initialise warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        _, warp_matrix = cv2.findTransformECC(src.astype(np.float32),
                                              trg.astype(np.float32),
                                              warp_matrix,
                                              warp_mode,
                                              criteria,
                                              None,
                                              self.params['gaussFiltSize'])
        return warp_matrix


class PointBasedRegistration(RegistrationTask):
    """ Registration class implementing a point-based registration from OpenCV contrib package
    """
    def get_params(self):
        LOGGER.info("{:s}:Params for this registration are:".format(self.__class__.__name__))
        LOGGER.info("\t\t\t\tModel: {:s}".format(self.params['Model']))
        LOGGER.info("\t\t\t\tDescriptor: {:s}".format(self.params['Descriptor']))
        LOGGER.info("\t\t\t\tMaxIters: {:d}".format(self.params['MaxIters']))
        LOGGER.info("\t\t\t\tRANSACThreshold: {:.2f}".format(self.params['RANSACThreshold']))

    def check_params(self):
        if not (self.params.get('Model') in ['Euler', 'PartialAffine', 'Homography']):
            LOGGER.info("{:s}:Model set to Euler".format(self.__class__.__name__))
            self.params['Model'] = 'Euler'
        if not (self.params.get('Descriptor') in ['SIFT', 'SURF']):
            LOGGER.info("{:s}:Descriptor set to SIFT".format(self.__class__.__name__))
            self.params['Descriptor'] = 'SIFT'
        if not isinstance(self.params.get('MaxIters'), int):
            LOGGER.info("{:s}:RANSAC MaxIters set to 1000".format(self.__class__.__name__))
            self.params['MaxIters'] = 1000
        if not isinstance(self.params.get('RANSACThreshold'), float):
            LOGGER.info("{:s}:RANSAC threshold set to 7.0".format(self.__class__.__name__))
            self.params['RANSACThreshold'] = 7.0

    def register(self, src, trg, trg_mask=None, src_mask=None):
        """ Implementation of pair-wise registration and warping using point-based matching

        This function estimates a number of transforms (Euler, PartialAffine and Homography) using point-based matching.
        Features descriptor are first extracted from the pair of images using either SIFT or SURF descriptors. A
        brute-force point-matching algorithm estimates matching points and a transformation is computed. All
        transformations use RANSAC to robustly fit a tranform to the matching points. However, the feature extraction
        and point matching estimation can be very poor and unstable. In those cases, an identity transform is used
        to warp the images instead.

        :param src: 2D single channel source moving image
        :param trg: 2D single channel target reference image
        :param trg_mask: Mask of target image. Not used in this method.
        :param src_mask: Mask of source image. Not used in this method.
        :return: Estimated 2D transformation matrix of shape 2x3
        """
        # Initialise matrix and failed registrations flag
        warp_matrix = None
        # Initiate point detector
        ptdt = cv2.xfeatures2d.SIFT_create() if self.params['Descriptor'] == 'SIFT' else cv2.xfeatures2d.SURF_create()
        # create BFMatcher object
        bf_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = ptdt.detectAndCompute(self.rescale_image(src), None)
        kp2, des2 = ptdt.detectAndCompute(self.rescale_image(trg), None)
        # Match descriptors if any are found
        if des1 is not None and des2 is not None:
            matches = bf_matcher.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.asarray([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 2)
            trg_pts = np.asarray([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 2)
            # Parse model and estimate matrix
            if self.params['Model'] == 'PartialAffine':
                warp_matrix = cv2.estimateRigidTransform(src_pts, trg_pts, fullAffine=False)
            elif self.params['Model'] == 'Euler':
                model = EstimateEulerTransformModel(src_pts, trg_pts)
                warp_matrix = ransac(src_pts.shape[0], model, 3, self.params['MaxIters'], 1, 5)
            elif self.params['Model'] == 'Homography':
                warp_matrix, _ = cv2.findHomography(src_pts, trg_pts, cv2.RANSAC,
                                                    ransacReprojThreshold=self.params['RANSACThreshold'],
                                                    maxIters=self.params['MaxIters'])
                if warp_matrix is not None:
                    warp_matrix = warp_matrix[:2, :]
        return warp_matrix

    @staticmethod
    def rescale_image(image):
        """ Normalise and scale image in 0-255 range """
        s2_min_value, s2_max_value = 0, 1
        out_min_value, out_max_value = 0, 255
        # Clamp values in 0-1 range
        image[image > s2_max_value] = s2_max_value
        image[image < s2_min_value] = s2_min_value
        # Rescale to uint8 range
        out_image = out_max_value + (image-s2_min_value)*(out_max_value-out_min_value)/(s2_max_value-s2_min_value)
        return out_image.astype(np.uint8)
