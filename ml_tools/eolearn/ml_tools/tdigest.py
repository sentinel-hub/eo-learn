"""
The module provides an EOTask for the computation of a T-Digest representation of an EOPatch.
Requires installation of `eolearn.ml_tools[TDIGEST]`.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from functools import partial
from itertools import product
from typing import Any, Callable, Generator, Iterable, Literal, Union

import numpy as np
import tdigest as td

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import FeatureSpec, FeaturesSpecification

ModeTypes = Union[Literal["standard", "timewise", "monthly", "total"], Callable]


class TDigestTask(EOTask):
    """
    An EOTask to compute the T-Digest representation of a chosen feature of an EOPatch.
    It integrates the [T-Digest algorithm by Ted Dunning](https://arxiv.org/abs/1902.04023) to efficiently compute
    quantiles of the underlying dataset into eo-learn.

    The output features of the tasks may be merged to compute a representation of the complete dataset.
    That enables quantile based normalisation or statistical analysis of datasets larger than RAM in EO.
    """

    def __init__(
        self,
        in_feature: FeaturesSpecification,
        out_feature: FeaturesSpecification,
        mode: Literal["standard", "timewise", "monthly", "total"] | Callable = "standard",
        pixelwise: bool = False,
        filternan: bool = False,
    ):
        """
        :param in_feature: The input feature to compute the T-Digest representation for.
        :param out_feature: The output feature where to save the T-Digest representation of the chosen feature.
        :param mode: The mode to apply to the timestamps and bands.
            * `'standard'` computes the T-Digest representation for each band accumulating timestamps.
            * `'timewise'` computes the T-Digest representation for each band and timestamp of the chosen feature.
            * `'monthly'` computes the T-Digest representation for each band accumulating the timestamps per month.
            * | `'total'` computes the total T-Digest representation of the whole feature accumulating all timestamps,
              | bands and pixels. Cannot be used with `pixelwise=True`.
            * | Callable computes the T-Digest representation defined by the processing function given as mode. Receives
              | the input_array of the feature, timestamps, shape and pixelwise and filternan keywords as an input.
        :param pixelwise: Decider whether to compute the T-Digest representation accumulating pixels or per pixel.
            Cannot be used with `mode='total'`.
        :param filternan: Decider whether to filter out nan-values before computing the T-Digest.
        """

        self.mode = mode

        self.pixelwise = pixelwise

        self.filternan = filternan

        if self.pixelwise and self.mode == "total":
            raise ValueError("Total mode does not support pixelwise=True.")

        self.in_feature = self.parse_features(in_feature, allowed_feature_types=partial(_is_input_ftype, mode=mode))
        self.out_feature = self.parse_features(
            out_feature, allowed_feature_types=partial(_is_output_ftype, mode=mode, pixelwise=pixelwise)
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

        for in_feature_, out_feature_, shape in _looper(
            in_feature=self.in_feature, out_feature=self.out_feature, eopatch=eopatch
        ):
            processing_func = self.mode if callable(self.mode) else _processing_function[self.mode]
            eopatch[out_feature_] = processing_func(
                input_array=eopatch[in_feature_],
                timestamps=eopatch.timestamps,
                shape=shape,
                pixelwise=self.pixelwise,
                filternan=self.filternan,
            )

        return eopatch


# auxiliary
def _is_input_ftype(feature_type: FeatureType, mode: ModeTypes) -> bool:
    if mode == "standard":
        return feature_type.is_image()
    if mode in ("timewise", "monthly"):
        return feature_type in [FeatureType.DATA, FeatureType.MASK]
    return True


def _is_output_ftype(feature_type: FeatureType, mode: ModeTypes, pixelwise: bool) -> bool:
    if callable(mode):
        return True

    if mode == "standard":
        return feature_type == (FeatureType.DATA_TIMELESS if pixelwise else FeatureType.SCALAR_TIMELESS)

    if mode in ("timewise", "monthly"):
        return feature_type == (FeatureType.DATA if pixelwise else FeatureType.SCALAR)

    return feature_type == FeatureType.SCALAR_TIMELESS


def _looper(
    in_feature: list[FeatureSpec], out_feature: list[FeatureSpec], eopatch: EOPatch
) -> Generator[tuple[FeatureSpec, FeatureSpec, np.ndarray], None, None]:
    for in_feature_, out_feature_ in zip(in_feature, out_feature):
        shape = np.array(eopatch[in_feature_].shape)
        yield in_feature_, out_feature_, shape


def _process_standard(
    input_array: np.ndarray, shape: np.ndarray, pixelwise: bool, filternan: bool, **_: Any
) -> np.ndarray:
    if pixelwise:
        array = np.empty(shape[-3:], dtype=object)
        for i, j, k in product(range(shape[-3]), range(shape[-2]), range(shape[-1])):
            array[i, j, k] = _get_tdigest(input_array[..., i, j, k], filternan)

    else:
        array = np.empty(shape[-1], dtype=object)
        for k in range(shape[-1]):
            array[k] = _get_tdigest(input_array[..., k], filternan)

    return array


def _process_timewise(
    input_array: np.ndarray, shape: np.ndarray, pixelwise: bool, filternan: bool, **_: Any
) -> np.ndarray:
    if pixelwise:
        array = np.empty(shape, dtype=object)
        for time_, i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2]), range(shape[3])):
            array[time_, i, j, k] = _get_tdigest(input_array[time_, i, j, k], filternan)

    else:
        array = np.empty(shape[[0, -1]], dtype=object)
        for time_, k in product(range(shape[0]), range(shape[-1])):
            array[time_, k] = _get_tdigest(input_array[time_, ..., k], filternan)

    return array


def _process_monthly(
    input_array: np.ndarray, timestamps: Iterable, shape: np.ndarray, pixelwise: bool, filternan: bool, **_: Any
) -> np.ndarray:
    midx = []
    for month_ in range(12):
        midx.append(np.array([timestamp.month == month_ + 1 for timestamp in timestamps]))

    if pixelwise:
        array = np.empty([12, *shape[1:]], dtype=object)
        for month_, i, j, k in product(range(12), range(shape[1]), range(shape[2]), range(shape[3])):
            array[month_, i, j, k] = _get_tdigest(input_array[midx[month_], i, j, k], filternan)

    else:
        array = np.empty([12, shape[-1]], dtype=object)
        for month_, k in product(range(12), range(shape[-1])):
            array[month_, k] = _get_tdigest(input_array[midx[month_], ..., k], filternan)

    return array


def _process_total(input_array: np.ndarray, filternan: bool, **_: Any) -> np.ndarray:
    return _get_tdigest(input_array, filternan)


_processing_function: dict[str, Callable] = {
    "standard": _process_standard,
    "timewise": _process_timewise,
    "monthly": _process_monthly,
    "total": _process_total,
}


def _get_tdigest(values: np.ndarray, filternan: bool) -> td.TDigest:
    result = td.TDigest()
    values_ = values.flatten()
    result.batch_update(values_[~np.isnan(values_)] if filternan else values_)
    return result
