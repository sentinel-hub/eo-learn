"""
The module provides an EOTask for the computation of a T-Digest representation of an EOPatch.

Credits:
Copyright (c) 2023 Michael Engel

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from functools import partial
from itertools import product

import numpy as np
import tdigest as td
from typing_extensions import Literal

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import FeaturesSpecification


class TDigestTask(EOTask):
    """
    An EOTask to compute the T-Digest representation of a chosen feature of an EOPatch.
    It integrates the [T-Digest algorithm by Ted Dunning](https://arxiv.org/abs/1902.04023) to efficiently \
        compute quantiles of the underlying dataset into eo-learn.
    The output features of the tasks may be merged to compute a representation of the complete dataset.
    That enables quantile based normalisation or statistical analysis of datasets larger than RAM in EO.
    """

    def __init__(
        self,
        in_feature: FeaturesSpecification,
        out_feature: FeaturesSpecification,
        mode: Literal["standard", "timewise", "monthly", "total"] = "standard",
        pixelwise: bool = False,
    ):
        """
        :param in_feature: The input feature to compute the T-Digest representation for.
        :param out_feature: The output feature where to save the T-Digest representation of the chosen feature.
        :param mode: The mode to apply to the timestamps and bands.
        - The 'standard' mode computes the T-Digest representation \
            for each band accumulating timestamps.
        - The 'timewise' mode computes the T-Digest representation \
            for each band and timestamp of the chosen feature.
        - The 'monthly' mode computes the T-Digest representation \
            for each band accumulating the timestamps per month.
        - The 'total' mode computes the total T-Digest representation \
            of the whole feature accumulating all timestamps, bands and pixels \
                - cannot be used with pixelwise=True.
        :param pixelwise: Decider whether to compute the T-Digest representation accumulating pixels or per pixel. \
            Cannot be used with mode='total'.
        """
        # set mode
        self.mode = mode

        # check pixelwise parameter
        self.pixelwise = pixelwise

        if self.pixelwise and self.mode == "total":
            raise ValueError("Total mode does not support pixelwise=True.")

        # check input and output features
        self.in_feature = self.parse_features(
            in_feature, allowed_feature_types=partial(_get_allowed_ftypes, ifin=True, mode=mode, pixelwise=pixelwise)
        )
        self.out_feature = self.parse_features(
            out_feature, allowed_feature_types=partial(_get_allowed_ftypes, ifin=False, mode=mode, pixelwise=pixelwise)
        )

        if len(self.in_feature) != len(self.out_feature):
            raise ValueError(
                f"The number of input ({len(self.in_feature)}) and output features ({len(self.out_feature)}) must"
                " match."
            )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        Execute method that computes the TDigest of the chosen features.

        :param eopatch: EOPatch which the chosen input feature already exists
        """

        if self.mode == "standard":
            for in_feature, out_feature in zip(self.in_feature, self.out_feature):
                shape = np.array(eopatch[in_feature].shape)
                if self.pixelwise:
                    eopatch[out_feature] = np.empty(shape[-3:], dtype=object)
                    for i, j, k in product(range(shape[-3]), range(shape[-2]), range(shape[-1])):
                        eopatch[out_feature][i, j, k] = td.TDigest()
                        eopatch[out_feature][i, j, k].batch_update(eopatch[in_feature][..., i, j, k].flatten())

                else:
                    eopatch[out_feature] = np.empty(shape[-1], dtype=object)
                    for k in range(shape[-1]):
                        eopatch[out_feature][k] = td.TDigest()
                        eopatch[out_feature][k].batch_update(eopatch[in_feature][..., k].flatten())

        elif self.mode == "timewise":
            for in_feature, out_feature in zip(self.in_feature, self.out_feature):
                shape = np.array(eopatch[in_feature].shape)
                if self.pixelwise:
                    eopatch[out_feature] = np.empty(shape, dtype=object)
                    for time_, i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2]), range(shape[3])):
                        eopatch[out_feature][time_, i, j, k] = td.TDigest()
                        eopatch[out_feature][time_, i, j, k].batch_update(eopatch[in_feature][time_, i, j, k].flatten())

                else:
                    eopatch[out_feature] = np.empty(shape[[0, -1]], dtype=object)
                    for k in range(shape[-1]):
                        eopatch[out_feature][time_, k] = td.TDigest()
                        eopatch[out_feature][time_, k].batch_update(eopatch[in_feature][time_, ..., k].flatten())

        elif self.mode == "monthly":
            midx = []
            for month_ in range(12):
                midx.append(np.array([timestamp.month == month_ + 1 for timestamp in eopatch["timestamp"]]))

            for in_feature, out_feature in zip(self.in_feature, self.out_feature):
                shape = np.array(eopatch[in_feature].shape)
                if self.pixelwise:
                    eopatch[out_feature] = np.empty([12, *shape[1:]], dtype=object)
                    for month_, i, j, k in product(range(12), range(shape[1]), range(shape[2]), range(shape[3])):
                        eopatch[out_feature][month_, i, j, k] = td.TDigest()
                        eopatch[out_feature][month_, i, j, k].batch_update(
                            eopatch[in_feature][midx[month_], i, j, k].flatten()
                        )

                else:
                    eopatch[out_feature] = np.empty([12, shape[-1]], dtype=object)
                    for month_, k in product(range(12), range(shape[-1])):
                        eopatch[out_feature][month_, k] = td.TDigest()
                        eopatch[out_feature][month_, k].batch_update(
                            eopatch[in_feature][midx[month_], ..., k].flatten()
                        )

        elif self.mode == "total":
            for in_feature, out_feature in zip(self.in_feature, self.out_feature):
                eopatch[out_feature] = td.TDigest()
                eopatch[out_feature].batch_update(eopatch[in_feature].flatten())

        else:
            raise RuntimeError(f"TDigestTask: mode {self.mode} not implemented!")

        return eopatch


def _get_allowed_ftypes(featuretype, ifin=True, mode="standard", pixelwise=False):
    if mode == "standard":
        allowed_in_types = [
            FeatureType.DATA,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK,
            FeatureType.MASK_TIMELESS,
        ]
        allowed_out_types = [FeatureType.DATA_TIMELESS] if pixelwise else [FeatureType.SCALAR_TIMELESS]

    elif mode == "timewise" or mode == "monthly":
        allowed_in_types = [FeatureType.DATA, FeatureType.MASK]
        allowed_out_types = [FeatureType.DATA] if pixelwise else [FeatureType.SCALAR]

    elif mode == "total":
        allowed_in_types = ...
        allowed_out_types = [FeatureType.SCALAR_TIMELESS]

    if ifin:
        return True if allowed_in_types is Ellipsis else featuretype in allowed_in_types
    else:
        return True if allowed_out_types is Ellipsis else featuretype in allowed_out_types
