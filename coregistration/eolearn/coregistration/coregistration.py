"""
This module implements the co-registration transformers.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.exceptions import EORuntimeWarning
from eolearn.core.types import FeaturesSpecification

LOGGER = logging.getLogger(__name__)


class ECCRegistrationTask(EOTask):
    """Multi-temporal image co-registration using OpenCV Enhanced Cross-Correlation method

    The task uses a temporal stack of images of the same location (i.e. a temporal-spatial feature in `EOPatch`)
    and a reference timeless feature to estimate a transformation that aligns each frame of the temporal stack
    to the reference feature.


    Each transformation is calculated using only a single channel of the images. If feature which contains masks of
    valid pixels is specified it is used during the estimation of the transformation. The estimated transformations
    are applied to each of the specified features.
    """

    def __init__(
        self,
        registration_feature: Tuple[FeatureType, str],
        reference_feature: Tuple[FeatureType, str],
        channel: int,
        valid_mask_feature: Optional[Tuple[FeatureType, str]] = None,
        apply_to_features: FeaturesSpecification = ...,
        interpolation_mode: int = cv2.INTER_LINEAR,
        warp_mode: int = cv2.MOTION_TRANSLATION,
        max_iter: int = 100,
        gauss_kernel_size: int = 1,
        border_mode: int = cv2.BORDER_REPLICATE,
        border_value: float = 0,
        num_threads: int = 1,
        max_translation: float = 5.0,
    ):
        """
        :param registration_feature: Feature in EOPatch holding the multi-temporal stack to register to the
            reference. Needs to be of FeatureType.DATA.
        :param reference_feature: Feature in EOPatch used as reference frame for registration.
        :param channel: Defines the index of the stack and reference feature to use during registration.
        :param valid_mask_feature: Optional feature in EOPatch that defines which pixels should be used for
            registration.
        :param apply_to_features: List of temporal features in EOPatch to which applied the estimated
            transformation.
        :param interpolation_mode: Interpolation type used when transforming the stack of images.
        :param warp_mode: Defines the transformation model used to match the stack and the reference.
            Examples include TRANSLATION, RIGID_MOTION, AFFINE.
        :param max_iter: Maximum number of iterations used during optimization of algorithm.
        :param gauss_kernel_size: Size of Gaussian kernel used to smooth images prior to registration.
        :param border_mode: Defines the padding strategy when transforming the images with estimated
            transformation.
        :param border_value: Value used for padding when border mode is set to CONSTANT.
        :param num_threads: Number of threads used for optimization of the algorithm.
        :param max_translation: Estimated transformations are considered incorrect when the norm of the
            translation component is larger than this parameter.
        """
        self.registration_feature = self.parse_feature(registration_feature, allowed_feature_types=[FeatureType.DATA])
        self.reference_feature = self.parse_feature(
            reference_feature, allowed_feature_types=[FeatureType.DATA_TIMELESS]
        )
        self.channel = channel
        self.valid_mask_feature = (
            None
            if valid_mask_feature is None
            else self.parse_feature(valid_mask_feature, allowed_feature_types=[FeatureType.MASK])
        )
        self.apply_features_parser = self.get_feature_parser(
            apply_to_features, allowed_feature_types=[FeatureType.DATA, FeatureType.MASK]
        )
        self.warp_mode = warp_mode
        self.interpolation_mode = interpolation_mode
        self.max_iter = max_iter
        self.gauss_kernel_size = gauss_kernel_size
        self.border_mode = border_mode
        self.border_value = border_value
        self.num_threads = num_threads
        self.max_translation = max_translation

    def register(
        self,
        src: np.ndarray,
        trg: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        warp_mode: int = cv2.MOTION_TRANSLATION,
    ) -> np.ndarray:
        """Method that estimates the transformation between source and target image"""
        criteria = (cv2.TERM_CRITERIA_COUNT, self.max_iter, 0)
        warp_matrix_size = (3, 3) if warp_mode == cv2.MOTION_HOMOGRAPHY else (2, 3)
        warp_matrix = np.eye(*warp_matrix_size, dtype=np.float32)

        try:
            cv2.setNumThreads(self.num_threads)
            _, warp_matrix = cv2.findTransformECC(
                src.astype(np.float32),
                trg.astype(np.float32),
                warp_matrix,
                warp_mode,
                criteria,
                valid_mask,
                self.gauss_kernel_size,
            )
        except cv2.error as cv2err:
            warnings.warn(f"Could not calculate the warp matrix: {cv2err}", EORuntimeWarning)

        return warp_matrix

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Method that estimates registrations and warps EOPatch objects"""
        multi_temp_stack = eopatch[self.registration_feature][..., self.channel]
        time_frames = multi_temp_stack.shape[0]

        valid_mask = None
        if self.valid_mask_feature is not None:
            valid_mask = eopatch[self.valid_mask_feature].squeeze(axis=-1)
            valid_mask = valid_mask.astype(np.uint8)

        reference_image = eopatch[self.reference_feature][..., self.channel]

        new_eopatch = EOPatch(bbox=eopatch.bbox, timestamp=eopatch.timestamp)
        for feature_type, feature_name in self.apply_features_parser.get_features(eopatch):
            new_eopatch[feature_type][feature_name] = np.zeros_like(eopatch[feature_type][feature_name])

        warp_matrices = {}

        for idx in range(time_frames):
            valid_mask_ = None if valid_mask is None else valid_mask[idx]
            warp_matrix = self.register(
                reference_image, multi_temp_stack[idx], valid_mask=valid_mask_, warp_mode=self.warp_mode
            )

            if self.is_translation_large(warp_matrix):
                warp_matrix = np.eye(2, 3)

            warp_matrices[idx] = warp_matrix.tolist()

            # Apply transformation to every given feature
            for feature_type, feature_name in self.apply_features_parser.get_features(eopatch):
                new_eopatch[feature_type][feature_name][idx] = self.warp_feature(
                    eopatch[feature_type][feature_name][idx], warp_matrix
                )

        new_eopatch[FeatureType.META_INFO, "warp_matrices"] = warp_matrices
        return new_eopatch

    def warp(self, img: np.ndarray, warp_matrix: np.ndarray, shape: Tuple[int, int], flags: int) -> np.ndarray:
        """Transform the target image with the estimated transformation matrix"""
        if warp_matrix.shape == (3, 3):
            return cv2.warpPerspective(
                img.astype(np.float32),
                warp_matrix,
                shape,
                flags=flags,
                borderMode=self.border_mode,
                borderValue=self.border_value,
            )
        return cv2.warpAffine(
            img.astype(np.float32),
            warp_matrix,
            shape,
            flags=flags,
            borderMode=self.border_mode,
            borderValue=self.border_value,
        )

    def warp_feature(self, img: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
        """Function to warp input image given an estimated 2D linear transformation"""

        height, width = img.shape[:2]
        warped_img = np.zeros_like(img, dtype=np.float32)

        flags = self.interpolation_mode + cv2.WARP_INVERSE_MAP

        # Check if image to warp is 2D or 3D. If 3D need to loop over channels
        if img.ndim == 2:
            warped_img = self.warp(img, warp_matrix, (width, height), flags=flags)
        elif img.ndim == 3:
            for idx in range(img.shape[-1]):
                warped_img[..., idx] = self.warp(img[..., idx], warp_matrix, (width, height), flags=flags)
        else:
            raise ValueError(f"Image has incorrect number of dimensions: {img.ndim}. Correct number is either 2 or 3.")

        return warped_img.astype(img.dtype)

    def is_translation_large(self, warp_matrix: np.ndarray) -> bool:
        """Method that checks if estimated linear translation could be implausible.

        This function checks whether the norm of the estimated translation in pixels exceeds a predefined value.
        """
        return np.linalg.norm(warp_matrix[:, 2]).astype(float) > self.max_translation


def get_gradient(src: np.ndarray) -> np.ndarray:
    """Method which calculates and returns the gradients for the input image, which are
    better suited for co-registration
    """
    # Calculate the x and y gradients using Sobel operator
    src = src.astype(np.float32)
    grad_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad
