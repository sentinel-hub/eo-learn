"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pytest

from eolearn.core import FeatureType


@pytest.mark.parametrize(
    "old_ftype, new_ftype",
    [
        (FeatureType.TIMESTAMP, FeatureType.TIMESTAMPS),
        (FeatureType["TIMESTAMP"], FeatureType["TIMESTAMPS"]),
        (FeatureType("timestamp"), FeatureType("timestamps")),
    ],
    ids=["attribute access", "name access", "value access"],
)
def test_timestamp_featuretype(old_ftype, new_ftype) -> None:
    assert old_ftype is new_ftype
