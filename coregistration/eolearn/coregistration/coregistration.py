"""
This module implements the co-registration transformers.
"""
# pylint: disable=invalid-name

import logging
import numpy as np
import cv2

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from registration import CrossCorr

from eolearn.core import EOTask

from .coregistration_utilities import ransac, EstimateEulerTransformModel

LOGGER = logging.getLogger(__name__)

MAX_TRANSLATION = 20
MAX_ROTATION = np.pi / 9


class Interpolation(Enum):
    NEAREST = 0
    LINEAR = 1
    CUBIC = 3


class RegistrationTask(EOTask):
    """ Task that implements a temporal co-registration of an input eopatch """

    def __init__(self, registration):
        """ Registration to be performed

        :param registration: Registration instance. Could be any of registration implemented in `registration` module
        """
        self.registration = registration

    def get_registration(self):
        """ Return the registration object """
        return self.registration

    def execute(self, eopatches):
        """ Perform registration and return transformed EOPatch

        :param eopatches: Input eopatch
        """
        return self.registration.execute(eopatches)


class Registration(ABC):
    """ Abstract class for registration

        Abstract for Registration methods. Four registration methods are implemented, namely ThunderRegistration,
        ECCRegistration, PointBasedRegistration and ElastixRegistration. Each registration estimates a series of
        pair-wise transformation that map one source image to a target image. The pair-wise registrations start form the
        latest frame proceeding backwards time-wise. This is because the latest frames are supposed to be least affected
        by orthorectification inaccuracies. Each pair-wise registration uses a single channel to estimate the
        transformation. The constructor takes two mandatory arguments specifying the attribute and field of array to
        use for registration, and three optional arguments specifying the index of the channel to use to
        estimate the registration, a dictionary specifying the parameters of the registration and an interpolation
        method used to generate the final warped images. If a ground-truth mask is present, it is warped using a nearest
        neighbour interpolation to not alter the encoding.

        :param attr_type: FeatureType specifying the attribute to use for transformation estimation
        :type attr_type: FeatureType
        :param field_name: String specifying the name of the dictionary field of hte attribute
        :type field_name: string
        :param valid_mask: String specifying the name of hte valid mask field ot be used to mask pixels during
                           registration. This field is supposed to be stored in the `mask` attribute
        :type valid_mask: string
        :param channel: Index of channel to be used in estimating transformation
        :type channel: int
        :param params: Dictionary of registration settings. Varies depending on registration method
        :type params: dict
        :param interpolation: Enum type specifying interpolation method. Allowed types are NEAREST, LINEAR and CUBIC
        :type interpolation: Interpolation enum
    """

    def __init__(self, attr_type, field_name, valid_mask=None, channel=0, params=None,
                 interpolation=Interpolation.CUBIC):
        self.attr_type = attr_type
        self.field_name = field_name
        self.channel = channel
        self.params = params
        self.interpolation = interpolation
        self.valid_data_name = valid_mask

    @abstractmethod
    def register(self, src, trg, trg_mask=None, src_mask=None):
        raise NotImplementedError

    @abstractmethod
    def check_params(self):
        """ Method to validate registration parameters """
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        """ Method to print out registration parameters used """
        raise NotImplementedError

    # def execute(self, eopatches):
    #     """ Method that performs registration on tuples """
    #     return tuple(map(self.do_execute, eopatches))

    def execute(self, eopatch):
        """ Method that estimates registrations and warps EOPatch objects """
        # Check if params are given correctly
        self.check_params()
        self.get_params()
        # Copy EOPatch and replace registered fields
        eopatch_new = deepcopy(eopatch)
        # Extract channel for registration
        sliced_data = deepcopy(eopatch[self.attr_type.value][self.field_name][..., self.channel])
        # Number of timeframes
        dn = sliced_data.shape[0]
        # Resolve interpolation
        if self.interpolation == Interpolation.CUBIC:
            iflag = cv2.INTER_CUBIC
        elif self.interpolation == Interpolation.NEAREST:
            iflag = cv2.INTER_NEAREST
        elif self.interpolation == Interpolation.LINEAR:
            iflag = cv2.INTER_LINEAR
        else:
            raise ValueError("Unsupported interpolation method specified")
        # Pair-wise registration starting from the most recent frame
        for idx in range(dn - 1, 0, -1):
            src_mask, trg_mask = None, None
            if (self.valid_data_name is not None) and (self.valid_data_name in eopatch_new.mask.keys()):
                src_mask = eopatch_new.mask[self.valid_data_name][idx-1]
                trg_mask = eopatch_new.mask[self.valid_data_name][idx]

            # Estimate transformation
            warp_matrix = self.register(sliced_data[idx-1], sliced_data[idx], src_mask=src_mask, trg_mask=trg_mask)

            # Check amount of deformation
            rflag = self.is_registration_suspicious(warp_matrix)

            # Flag suspicious registrations and set them to the identity
            if rflag:
                LOGGER.warning("{:s} warning in pair-wise reg {:d} to {:d}".format(self.__class__.__name__,
                                                                                   idx - 1, idx))
                warp_matrix = np.eye(2, 3)

            # Transform and update sliced_data
            sliced_data[idx-1] = self.warp(warp_matrix, sliced_data[idx-1], iflag)

            # Warp corresponding image in every field of data
            for data_key in eopatch_new.data.keys():
                eopatch_new.data[data_key][idx-1] = self.warp(warp_matrix, eopatch_new.data[data_key][idx-1], iflag)

            # Warp corresponding image in every field of mask
            for mask_key in eopatch_new.mask.keys():
                eopatch_new.mask[mask_key][idx-1] = self.warp(warp_matrix, eopatch_new.mask[mask_key][idx-1], iflag)

            del warp_matrix

        return eopatch_new

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

        h, w = img.shape[:2]
        warped_img = np.zeros_like(img, dtype=img.dtype)

        # Check if image to warp is 2D or 3D. If 3D need to loop over channels
        if (self.interpolation == Interpolation.LINEAR) or img.ndim == 2:
            warped_img = cv2.warpAffine(img.astype(np.float32), warp_matrix, (w, h), flags=iflag).astype(img.dtype)

        elif img.ndim == 3:
            for idx in range(img.shape[-1]):
                warped_img[..., idx] = cv2.warpAffine(img[..., idx].astype(np.float32),
                                                      warp_matrix, (w, h),
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


class ThunderRegistration(Registration):
    """ Registration class implementing a translational registration using the thunder-registration package """
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
        ccreg = CrossCorr()
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


class ECCRegistration(Registration):
    """ Registration class implementing an intensity-based method within opencv """
    def get_params(self):
        LOGGER.info("{:s}:Params for this registration are:".format(self.__class__.__name__))
        LOGGER.info("\t\t\t\tMaxIters: {:d}".format(self.params['MaxIters']))

    def check_params(self):
        if (self.params is None) or (not isinstance(self.params.get('MaxIters'), int)):
            self.params = dict(MaxIters=200)

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
                                              warp_matrix, warp_mode, criteria)
        return warp_matrix


class PointBasedRegistration(Registration):
    """ Registration class implementing a point-based registration using opencv-contrib """
    def get_params(self):
        LOGGER.info("{:s}:Params for this registration are:".format(self.__class__.__name__))
        LOGGER.info("\t\t\t\tModel: {:s}".format(self.params['Model']))
        LOGGER.info("\t\t\t\tDescriptor: {:s}".format(self.params['Descriptor']))
        LOGGER.info("\t\t\t\tMaxIters: {:d}".format(self.params['MaxIters']))
        LOGGER.info("\t\t\t\tRANSACThreshold: {:.2f}".format(self.params['RANSACThreshold']))

    def check_params(self):
        if self.params is None:
            self.params = dict(Model='Euler', Descriptor='SIFT', MaxIters=1000, RANSACThreshold=7.0)
        else:
            if not (self.params.get('Model') in ['Euler', 'PartialAffine', 'Homography']):
                LOGGER.info("{:s}:Model set to Euler".format(self.__class__.__name__))
                self.params['Model'] = 'Euler'
            if not (self.params.get('Descriptor') in ['SIFT', 'SURF']):
                LOGGER.info("{:s}:Descriptor set to SIFT".format(self.__class__.__name__))
                self.params['Descriptor'] = 'SIFT'
            if (self.params.get('MaxIters') is None) or (not isinstance(self.params.get('MaxIters'), int)):
                LOGGER.info("{:s}:RANSAC MaxIters set to 1000".format(self.__class__.__name__))
                self.params['MaxIters'] = 1000
            if (self.params.get('RANSACThreshold') is None) or (not isinstance(self.params.get('RANSACThreshold'),
                                                                               float)):
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
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = ptdt.detectAndCompute(self.rescale_image(src), None)
        kp2, des2 = ptdt.detectAndCompute(self.rescale_image(trg), None)
        # Match descriptors if any are found
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
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
