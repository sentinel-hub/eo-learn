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
import datetime

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from geopandas import GeoSeries, GeoDataFrame

from sentinelhub import BBox, CRS
from eolearn.core import EOPatch, FeatureType, FeatureTypeSet
from eolearn.core.eodata_io import FeatureIO


def test_numpy_feature_types():
    eop = EOPatch()

    data_examples = []
    for size in range(6):
        for dtype in [np.float32, np.float64, float, np.uint8, np.int64, bool]:
            data_examples.append(np.zeros((2,) * size, dtype=dtype))

    for feature_type in FeatureTypeSet.RASTER_TYPES:
        valid_count = 0

        for data in data_examples:
            try:
                eop[feature_type]["TEST"] = data
                valid_count += 1
            except ValueError:
                pass

        expected_count = 3 if feature_type.is_discrete() else 6
        assert valid_count == expected_count, f"Feature type {feature_type} should take only a specific type of data"


def test_vector_feature_types():
    eop = EOPatch()

    invalid_entries = [{}, [], 0, None]

    for feature_type in FeatureTypeSet.VECTOR_TYPES:
        for entry in invalid_entries:
            with pytest.raises(ValueError):
                # Invalid entry for feature_type should raise an error
                eop[feature_type]["TEST"] = entry

    crs_test = CRS.WGS84.pyproj_crs()
    geo_test = GeoSeries([BBox((1, 2, 3, 4), crs=CRS.WGS84).geometry], crs=crs_test)

    eop.vector_timeless["TEST"] = geo_test
    assert isinstance(eop.vector_timeless["TEST"], GeoDataFrame), "GeoSeries should be parsed into GeoDataFrame"
    assert hasattr(eop.vector_timeless["TEST"], "geometry"), "Feature should have geometry attribute"
    assert eop.vector_timeless["TEST"].crs == crs_test, "GeoDataFrame should still contain the crs"

    with pytest.raises(ValueError):
        # Should fail because there is no TIMESTAMP column
        eop.vector["TEST"] = geo_test


@pytest.mark.parametrize(
    "invalid_entry", [0, list(range(4)), tuple(range(5)), {}, set(), [1, 2, 4, 3, 4326, 3], "BBox"]
)
def test_bbox_feature_type(invalid_entry):
    eop = EOPatch()
    with pytest.raises((TypeError, ValueError)):
        # Invalid bbox entry should raise an error
        eop.bbox = invalid_entry


@pytest.mark.parametrize(
    "valid_entry", [["2018-01-01", "15.2.1992"], (datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.date(2017, 1, 11))]
)
def test_timestamp_valid_feature_type(valid_entry):
    eop = EOPatch()
    eop.timestamp = valid_entry


@pytest.mark.parametrize(
    "invalid_entry",
    [
        [datetime.datetime(2017, 1, 1, 10, 4, 7), None, datetime.datetime(2017, 1, 11, 10, 3, 51)],
        "something",
        datetime.datetime(2017, 1, 1, 10, 4, 7),
    ],
)
def test_timestamp_invalid_feature_type(invalid_entry):
    eop = EOPatch()
    with pytest.raises((ValueError, TypeError)) as e:
        eop.timestamp = invalid_entry


def test_invalid_characters():
    eop = EOPatch()
    with pytest.raises(ValueError):
        eop.data_timeless["mask.npy"] = np.arange(3 * 3 * 2).reshape(3, 3, 2)


def test_repr(test_eopatch_path):
    test_eopatch = EOPatch.load(test_eopatch_path)
    repr_str = repr(test_eopatch)
    assert repr_str.startswith("EOPatch(") and repr_str.endswith(")")
    assert len(repr_str) > 100

    assert repr(EOPatch()) == "EOPatch()"


def test_repr_no_crs(test_eopatch):
    test_eopatch.vector_timeless["LULC"].crs = None
    repr_str = test_eopatch.__repr__()
    assert (
        isinstance(repr_str, str) and len(repr_str) > 100
    ), "EOPatch __repr__ must return non-empty string even in case of missing crs"


def test_add_feature():
    bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)

    eop = EOPatch()
    eop.data["bands"] = bands

    assert np.array_equal(eop.data["bands"], bands), "Data numpy array not stored"


def test_simplified_feature_operations():
    bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    feature = FeatureType.DATA, "TEST-BANDS"
    eop = EOPatch()

    eop[feature] = bands
    assert np.array_equal(eop[feature], bands), "Data numpy array not stored"


def test_delete_feature():
    bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    zeros = np.zeros_like(bands, dtype=float)
    ones = np.ones_like(bands, dtype=int)
    twos = np.ones_like(bands, dtype=int) * 2
    threes = np.ones((3, 3, 1), dtype=np.uint8) * 3
    arranged = np.arange(3 * 3 * 1, dtype=np.uint16).reshape(3, 3, 1)

    eop = EOPatch(
        data={"bands": bands, "zeros": zeros},
        mask={"ones": ones, "twos": twos},
        mask_timeless={"threes": threes, "arranged": arranged},
    )

    test_cases = [
        [FeatureType.DATA, "zeros", "bands", bands],
        [FeatureType.MASK, "ones", "twos", twos],
        [FeatureType.MASK_TIMELESS, "threes", "arranged", arranged],
    ]
    for feature_type, deleted, remaining, unchanged in test_cases:
        del eop[(feature_type, deleted)]
        assert deleted not in eop[feature_type], f"`({feature_type}, {deleted})` not deleted"
        assert_array_equal(
            eop[feature_type][remaining],
            unchanged,
            err_msg=f"`({feature_type}, {remaining})` changed or wrongly deleted",
        )

    with pytest.raises(KeyError):
        del eop[(FeatureType.DATA, "not_here")]


def test_shallow_copy(test_eopatch):
    eopatch_copy = test_eopatch.copy()
    assert test_eopatch == eopatch_copy
    assert test_eopatch is not eopatch_copy

    eopatch_copy.mask["CLM"] += 1
    assert test_eopatch == eopatch_copy
    assert test_eopatch.mask["CLM"] is eopatch_copy.mask["CLM"]

    eopatch_copy.timestamp.pop()
    assert test_eopatch != eopatch_copy


def test_deep_copy(test_eopatch):
    eopatch_copy = test_eopatch.copy(deep=True)
    assert test_eopatch == eopatch_copy
    assert test_eopatch is not eopatch_copy

    eopatch_copy.mask["CLM"] += 1
    assert test_eopatch != eopatch_copy


@pytest.mark.parametrize("features", (..., [(FeatureType.MASK, "CLM")]))
def test_copy_lazy_loaded_patch(test_eopatch_path, features):
    original_eopatch = EOPatch.load(test_eopatch_path, lazy_loading=True)
    copied_eopatch = original_eopatch.copy(features=features)

    value1 = original_eopatch.mask.__getitem__("CLM", load=False)
    assert isinstance(value1, FeatureIO)
    value2 = copied_eopatch.mask.__getitem__("CLM", load=False)
    assert isinstance(value2, FeatureIO)
    assert value1 is value2

    mask1 = original_eopatch.mask["CLM"]
    assert copied_eopatch.mask.__getitem__("CLM", load=False).loaded_value is not None
    mask2 = copied_eopatch.mask["CLM"]
    assert isinstance(mask1, np.ndarray)
    assert mask1 is mask2

    original_eopatch = EOPatch.load(test_eopatch_path, lazy_loading=True)
    copied_eopatch = original_eopatch.copy(features=features, deep=True)

    value1 = original_eopatch.mask.__getitem__("CLM", load=False)
    assert isinstance(value1, FeatureIO)
    value2 = copied_eopatch.mask.__getitem__("CLM", load=False)
    assert isinstance(value2, FeatureIO)
    assert value1 is not value2
    mask1 = original_eopatch.mask["CLM"]
    assert copied_eopatch.mask.__getitem__("CLM", load=False).loaded_value is None
    mask2 = copied_eopatch.mask["CLM"]
    assert np.array_equal(mask1, mask2) and mask1 is not mask2


def test_copy_features(test_eopatch):
    feature = FeatureType.MASK, "CLM"
    eopatch_copy = test_eopatch.copy(features=[feature])
    assert test_eopatch != eopatch_copy
    assert eopatch_copy[feature] is test_eopatch[feature]
    assert eopatch_copy.timestamp == []


@pytest.mark.parametrize(
    "ftype, fname",
    [
        [FeatureType.DATA, "BANDS-S2-L1C"],
        [FeatureType.MASK, "CLM"],
        [FeatureType.BBOX, ...],
        [FeatureType.TIMESTAMP, None],
    ],
)
def test_contains(ftype, fname, test_eopatch):
    assert ftype in test_eopatch
    assert (ftype, fname) in test_eopatch

    if ftype.has_dict():
        del test_eopatch[ftype, fname]
    else:
        test_eopatch[ftype] = None if ftype is FeatureType.BBOX else []

    assert ftype, fname not in test_eopatch


def test_equals():
    eop1 = EOPatch(data={"bands": np.arange(2 * 3 * 3 * 2, dtype=np.float32).reshape(2, 3, 3, 2)})
    eop2 = EOPatch(data={"bands": np.arange(2 * 3 * 3 * 2, dtype=np.float32).reshape(2, 3, 3, 2)})
    assert eop1 == eop2
    assert eop1.data == eop2.data

    eop1.data["bands"][1, ...] = np.nan
    assert eop1 != eop2
    assert eop1.data != eop2.data

    eop2.data["bands"][1, ...] = np.nan
    assert eop1 == eop2

    eop1.data["bands"] = np.reshape(eop1.data["bands"], (2, 3, 2, 3))
    assert eop1 != eop2

    eop2.data["bands"] = np.reshape(eop2.data["bands"], (2, 3, 2, 3))
    eop1.data["bands"] = eop1.data["bands"].astype(np.float16)
    assert eop1 != eop2

    del eop1.data["bands"]
    del eop2.data["bands"]
    assert eop1 == eop2

    eop1.data_timeless["dem"] = np.arange(3 * 3 * 2).reshape(3, 3, 2)
    assert eop1 != eop2


def test_timestamp_consolidation():
    # 10 frames
    timestamps = [
        datetime.datetime(2017, 1, 1, 10, 4, 7),
        datetime.datetime(2017, 1, 4, 10, 14, 5),
        datetime.datetime(2017, 1, 11, 10, 3, 51),
        datetime.datetime(2017, 1, 14, 10, 13, 46),
        datetime.datetime(2017, 1, 24, 10, 14, 7),
        datetime.datetime(2017, 2, 10, 10, 1, 32),
        datetime.datetime(2017, 2, 20, 10, 6, 35),
        datetime.datetime(2017, 3, 2, 10, 0, 20),
        datetime.datetime(2017, 3, 12, 10, 7, 6),
        datetime.datetime(2017, 3, 15, 10, 12, 14),
    ]

    data = np.random.rand(10, 100, 100, 3)
    mask = np.random.randint(0, 2, (10, 100, 100, 1))
    mask_timeless = np.random.randint(10, 20, (100, 100, 1))
    scalar = np.random.rand(10, 1)

    eop = EOPatch(
        timestamp=timestamps,
        data={"DATA": data},
        mask={"MASK": mask},
        scalar={"SCALAR": scalar},
        mask_timeless={"MASK_TIMELESS": mask_timeless},
    )

    good_timestamps = timestamps.copy()
    del good_timestamps[0]
    del good_timestamps[-1]
    good_timestamps.append(datetime.datetime(2017, 12, 1))

    removed_frames = eop.consolidate_timestamps(good_timestamps)

    assert good_timestamps[:-1] == eop.timestamp
    assert len(removed_frames) == 2
    assert timestamps[0] in removed_frames
    assert timestamps[-1] in removed_frames
    assert np.array_equal(data[1:-1, ...], eop.data["DATA"])
    assert np.array_equal(mask[1:-1, ...], eop.mask["MASK"])
    assert np.array_equal(scalar[1:-1, ...], eop.scalar["SCALAR"])
    assert np.array_equal(mask_timeless, eop.mask_timeless["MASK_TIMELESS"])
