"""
The eodata module provides core objects for handling remote sensing multi-temporal data (such as satellite imagery).

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sentinelhub import CRS, BBox

from ..constants import FeatureType
from ..eodata import EOPatch
from ..utils.parsing import FeatureParser

DEFAULT_BBOX = BBox((0, 0, 100, 100), crs=CRS("EPSG:32633"))


@dataclass
class PatchGeneratorConfig:
    """Dataclass containing a more complex setup of the PatchGenerator class."""

    num_timestamps: int = 5
    timestamps_range: Tuple[dt.datetime, dt.datetime] = (dt.datetime(2019, 1, 1), dt.datetime(2019, 12, 31))
    timestamps: List[dt.datetime] = field(init=False, repr=False)

    max_integer_value: int = 256
    raster_shape: Tuple[int, int] = (98, 151)
    depth_range: Tuple[int, int] = (1, 3)

    def __post_init__(self):
        self.timestamps = list(pd.date_range(*self.timestamps_range, periods=self.num_timestamps).to_pydatetime())


def generate_eopatch(
    features: Optional[List[Tuple[FeatureType, str]]] = None,
    bbox: BBox = DEFAULT_BBOX,
    timestamps: Optional[List[dt.datetime]] = None,
    seed: int = 42,
    config: Optional[PatchGeneratorConfig] = None,
):
    """A class for generating EOPatches with dummy data."""
    config = config if config is not None else PatchGeneratorConfig()
    supported_feature_types = [FeatureType.DATA, FeatureType.MASK, FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS]
    parsed_features = FeatureParser(features or [], supported_feature_types).get_features()
    rng = np.random.default_rng(seed)

    timestamps = timestamps if timestamps is not None else config.timestamps
    patch = EOPatch(bbox=bbox, timestamps=timestamps)

    # fill eopatch with random data
    # note: the patch generation functionality could be extended by generating extra random features
    for ftype, fname in parsed_features:
        shape = _get_feature_shape(rng, ftype, timestamps, config)
        patch[(ftype, fname)] = _generate_feature_data(rng, ftype, shape, config)
    return patch


def _generate_feature_data(
    rng: np.random.Generator, ftype: FeatureType, shape: Tuple[int, ...], config: PatchGeneratorConfig
) -> np.ndarray:
    if ftype.is_discrete():
        return rng.integers(config.max_integer_value, size=shape)
    return rng.normal(size=shape)


def _get_feature_shape(
    rng: np.random.Generator, ftype: FeatureType, timestamps: List[dt.datetime], config: PatchGeneratorConfig
) -> Tuple[int, ...]:
    time, height, width, depth = len(timestamps), *config.raster_shape, rng.integers(*config.depth_range)
    return (time, height, width, depth) if ftype.is_temporal() else (height, width, depth)
