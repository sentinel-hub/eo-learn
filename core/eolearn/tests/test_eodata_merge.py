"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
import datetime as dt

import numpy as np
import pytest
from geopandas import GeoDataFrame

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureType, merge_eopatches
from eolearn.core.constants import TIMESTAMP_COLUMN
from eolearn.core.eodata_io import FeatureIO
from eolearn.core.exceptions import EORuntimeWarning

DUMMY_BBOX = BBox((1, 2, 3, 4), CRS.WGS84)


def test_time_dependent_merge():
    all_timestamps = [dt.datetime(2020, month, 1) for month in range(1, 7)]
    eop1 = EOPatch(
        bbox=DUMMY_BBOX,
        data={"bands": np.ones((3, 4, 5, 2))},
        timestamps=[all_timestamps[0], all_timestamps[5], all_timestamps[4]],
    )
    eop2 = EOPatch(
        bbox=DUMMY_BBOX,
        data={"bands": np.ones((5, 4, 5, 2))},
        timestamps=[all_timestamps[3], all_timestamps[1], all_timestamps[2], all_timestamps[4], all_timestamps[3]],
    )

    eop = merge_eopatches(eop1, eop2)
    expected_eop = EOPatch(bbox=DUMMY_BBOX, data={"bands": np.ones((6, 4, 5, 2))}, timestamps=all_timestamps)
    assert eop == expected_eop

    eop = merge_eopatches(eop1, eop2, time_dependent_op="concatenate")
    expected_eop = EOPatch(
        bbox=DUMMY_BBOX,
        data={"bands": np.ones((8, 4, 5, 2))},
        timestamps=all_timestamps[:4] + [all_timestamps[3], all_timestamps[4]] + all_timestamps[4:],
    )
    assert eop == expected_eop

    eop1.data["bands"][1, ...] = 6
    eop1.data["bands"][-1, ...] = 3
    eop2.data["bands"][0, ...] = 5
    eop2.data["bands"][1, ...] = 4

    with pytest.raises(ValueError):
        merge_eopatches(eop1, eop2)

    eop = merge_eopatches(eop1, eop2, time_dependent_op="mean")
    expected_eop = EOPatch(bbox=DUMMY_BBOX, data={"bands": np.ones((6, 4, 5, 2))}, timestamps=all_timestamps)
    expected_eop.data["bands"][1, ...] = 4
    expected_eop.data["bands"][3, ...] = 3
    expected_eop.data["bands"][4, ...] = 2
    expected_eop.data["bands"][5, ...] = 6
    assert eop == expected_eop


def test_time_dependent_merge_with_missing_features():
    timestamps = [dt.datetime(2020, month, 1) for month in range(1, 7)]
    eop1 = EOPatch(
        bbox=DUMMY_BBOX,
        data={"bands": np.ones((6, 4, 5, 2))},
        label={"label": np.ones((6, 7), dtype=np.uint8)},
        timestamps=timestamps,
    )
    eop2 = EOPatch(bbox=DUMMY_BBOX, timestamps=timestamps[:4])

    eop = merge_eopatches(eop1, eop2)
    assert eop == eop1

    eop = merge_eopatches(eop2, eop1, eop1, eop2, time_dependent_op="min")
    assert eop == eop1

    eop = merge_eopatches(eop1)
    assert eop == eop1


def test_failed_time_dependent_merge():
    eop1 = EOPatch(bbox=DUMMY_BBOX, data={"bands": np.ones((6, 4, 5, 2))})
    with pytest.raises(ValueError):
        merge_eopatches(
            eop1,
        )
    eop2 = EOPatch(bbox=DUMMY_BBOX, data={"bands": np.ones((1, 4, 5, 2))}, timestamps=[dt.datetime(2020, 1, 1)])
    with pytest.raises(ValueError):
        merge_eopatches(eop2, eop1)


def test_timeless_merge():
    eop1 = EOPatch(
        bbox=DUMMY_BBOX,
        mask_timeless={"mask": np.ones((3, 4, 5), dtype=np.int16), "mask1": np.ones((5, 4, 3), dtype=np.int16)},
    )
    eop2 = EOPatch(
        bbox=DUMMY_BBOX,
        mask_timeless={"mask": 4 * np.ones((3, 4, 5), dtype=np.int16), "mask2": np.ones((4, 5, 3), dtype=np.int16)},
    )

    with pytest.raises(ValueError):
        merge_eopatches(eop1, eop2)

    eop = merge_eopatches(eop1, eop2, timeless_op="concatenate")
    expected_eop = EOPatch(
        bbox=DUMMY_BBOX,
        mask_timeless={
            "mask": np.ones((3, 4, 10), dtype=np.int16),
            "mask1": eop1.mask_timeless["mask1"],
            "mask2": eop2.mask_timeless["mask2"],
        },
    )
    expected_eop.mask_timeless["mask"][..., 5:] = 4
    assert eop == expected_eop

    eop = merge_eopatches(eop1, eop2, eop2, timeless_op="min")
    expected_eop = EOPatch(
        bbox=DUMMY_BBOX,
        mask_timeless={
            "mask": eop1.mask_timeless["mask"],
            "mask1": eop1.mask_timeless["mask1"],
            "mask2": eop2.mask_timeless["mask2"],
        },
    )
    assert eop == expected_eop


def test_vector_merge():
    bbox = BBox((1, 2, 3, 4), CRS.WGS84)
    dummy_gdf = GeoDataFrame(
        {
            "values": [1, 2],
            TIMESTAMP_COLUMN: [dt.datetime(2017, 1, 1, 10, 4, 7), dt.datetime(2017, 1, 4, 10, 14, 5)],
            "geometry": [bbox.geometry, bbox.geometry],
        },
        crs=bbox.crs.pyproj_crs(),
    )

    eop1 = EOPatch(bbox=bbox, vector_timeless={"vectors": dummy_gdf})

    assert eop1 == merge_eopatches(eop1, eop1)

    eop2 = eop1.__deepcopy__()
    eop2.vector_timeless["vectors"].crs = CRS.POP_WEB.pyproj_crs()
    with pytest.raises(ValueError):
        merge_eopatches(eop1, eop2)


def test_meta_info_merge():
    eop1 = EOPatch(bbox=DUMMY_BBOX, meta_info={"a": 1, "b": 2})
    eop2 = EOPatch(bbox=DUMMY_BBOX, meta_info={"a": 1, "c": 5})

    eop = merge_eopatches(eop1, eop2)
    expected_eop = EOPatch(bbox=DUMMY_BBOX, meta_info={"a": 1, "b": 2, "c": 5})
    assert eop == expected_eop

    eop2.meta_info["a"] = 3
    with pytest.warns(EORuntimeWarning):
        eop = merge_eopatches(eop1, eop2)
    assert eop == expected_eop


def test_bbox_merge():
    eop1 = EOPatch(bbox=BBox((1, 2, 3, 4), CRS.WGS84))
    eop2 = EOPatch(bbox=BBox((1, 2, 3, 4), CRS.POP_WEB))

    eop = merge_eopatches(eop1, eop1)
    assert eop == eop1

    with pytest.raises(ValueError):
        merge_eopatches(eop1, eop2)


def test_lazy_loading(test_eopatch_path):
    eop1 = EOPatch.load(test_eopatch_path, lazy_loading=True)
    eop2 = EOPatch.load(test_eopatch_path, lazy_loading=True)

    eop = merge_eopatches(eop1, eop2, features=[(FeatureType.MASK, ...)])
    assert isinstance(eop.mask.get("CLM"), np.ndarray)
    assert isinstance(eop1.mask.get("CLM"), np.ndarray)
    assert isinstance(eop1.mask_timeless._get_unloaded("LULC"), FeatureIO)  # noqa: SLF001
