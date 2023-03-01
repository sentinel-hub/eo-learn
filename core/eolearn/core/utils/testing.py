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
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from sentinelhub import CRS, BBox

from ..constants import FeatureType
from ..eodata import EOPatch
from ..utils.parsing import FeatureParser


@dataclass
class PatchGeneratorConfig:
    """Dataclass containing a more complex setup of the PatchGenerator class."""

    timestamps_periods: int = 15
    timestamps_range: Tuple[str, str] = ("2019-01-01", "2019-12-31")

    max_integer_value: int = 256
    raster_shape: Tuple[int, int] = (300, 400)
    depth_range: Tuple[int, int] = (1, 5)

    num_random_features: int = 5
    allowed_feature_types: Set[FeatureType] = field(default_factory=set)

    bbox: BBox = field(init=False, repr=False)
    timestamps: List[dt.datetime] = field(init=False, repr=False)

    def __post_init__(self):
        self.bbox = BBox((0, 0, *self.raster_shape), crs=CRS.UTM_33N)
        self.timestamps = list(pd.date_range(*self.timestamps_range, periods=self.timestamps_periods).to_pydatetime())
        if not self.allowed_feature_types:
            self.allowed_feature_types = {ftype for ftype in FeatureType if ftype.is_raster()}


def patch_generator(
    features: Optional[List[Tuple[FeatureType, str]]] = None,
    bbox: Optional[BBox] = None,
    timestamps: Optional[List[dt.datetime]] = None,
    seed: int = 42,
    config: Optional[PatchGeneratorConfig] = None,
):
    """A class for generating EOPatches with dummy data."""
    config = config if config is not None else PatchGeneratorConfig()
    parsed_features = FeatureParser(features or [], config.allowed_feature_types).get_features()
    rng = np.random.default_rng(seed)

    bbox = bbox if bbox is not None else config.bbox
    timestamps = timestamps if timestamps is not None else config.timestamps
    patch = EOPatch(bbox=bbox, timestamps=timestamps)

    # generate random features
    feature_types = [ftype for ftype, _ in parsed_features]
    feature_types.extend(rng.choice(np.array(config.allowed_feature_types), config.num_random_features))
    for ftype in set(feature_types):
        parsed_features.extend([(ftype, f"{ftype.value}{idx}") for idx in range(feature_types.count(ftype) + 1)])

    # fill eopatch with random data
    for ftype, fname in parsed_features:
        shape = [len(timestamps)] if ftype.is_temporal() else []  # timestamps
        shape.extend(config.raster_shape if ftype.is_spatial() else [])  # height, width
        shape.append(rng.integers(*config.depth_range))  # depth

        data = rng.integers(config.max_integer_value, size=shape) if ftype.is_discrete() else rng.normal(size=shape)
        patch[(ftype, fname)] = data

    return patch
