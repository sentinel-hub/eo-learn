"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import datetime

import pytest
import numpy as np
from geopandas import GeoSeries, GeoDataFrame

from sentinelhub import BBox, CRS
from eolearn.core import EOPatch, FeatureType, FeatureTypeSet


def test_loading_valid(test_eopatch_path):
    eop = EOPatch.load(test_eopatch_path)

    repr_str = eop.__repr__()
    assert isinstance(repr_str, str) and len(repr_str) > 0, 'EOPatch __repr__ must return non-empty string'


def test_numpy_feature_types():
    eop = EOPatch()

    data_examples = []
    for size in range(6):
        for dtype in [np.float32, np.float64, float, np.uint8, np.int64, bool]:
            data_examples.append(np.zeros((2, ) * size, dtype=dtype))

    for feature_type in FeatureTypeSet.RASTER_TYPES:
        valid_count = 0

        for data in data_examples:
            try:
                eop[feature_type]['TEST'] = data
                valid_count += 1
            except ValueError:
                pass

        assert valid_count == 6, f'Feature type {feature_type} should take only a specific type of data'


def test_vector_feature_types():
    eop = EOPatch()

    invalid_entries = [
        {}, [], 0, None
    ]

    for feature_type in FeatureTypeSet.VECTOR_TYPES:
        for entry in invalid_entries:
            with pytest.raises(ValueError):
                # Invalid entry for feature_type should raise an error
                eop[feature_type]['TEST'] = entry

    crs_test = CRS.WGS84.pyproj_crs()
    geo_test = GeoSeries([BBox((1, 2, 3, 4), crs=CRS.WGS84).geometry], crs=crs_test)

    eop.vector_timeless['TEST'] = geo_test
    assert isinstance(eop.vector_timeless['TEST'], GeoDataFrame), 'GeoSeries should be parsed into GeoDataFrame'
    assert hasattr(eop.vector_timeless['TEST'], 'geometry'), 'Feature should have geometry attribute'
    assert eop.vector_timeless['TEST'].crs == crs_test, 'GeoDataFrame should still contain the crs'

    with pytest.raises(ValueError):
        # Should fail because there is no TIMESTAMP column
        eop.vector['TEST'] = geo_test


def test_bbox_feature_type():
    eop = EOPatch()
    invalid_entries = [
        0, list(range(4)), tuple(range(5)), {}, set(), [1, 2, 4, 3, 4326, 3], 'BBox'
    ]

    for entry in invalid_entries:
        with pytest.raises((TypeError, ValueError)):
            # Invalid bbox entry should raise an error
            eop.bbox = entry


def test_timestamp_feature_type():
    eop = EOPatch()
    invalid_entries = [
        [datetime.datetime(2017, 1, 1, 10, 4, 7), None, datetime.datetime(2017, 1, 11, 10, 3, 51)],
        'something',
        datetime.datetime(2017, 1, 1, 10, 4, 7)
    ]

    valid_entries = [
        ['2018-01-01', '15.2.1992'],
        (datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.date(2017, 1, 11))
    ]

    for entry in invalid_entries:
        with pytest.raises((ValueError, TypeError)) as e:
            # Invalid timestamp entry should raise an error
            eop.timestamp = entry

    for entry in valid_entries:
        eop.timestamp = entry


def test_invalid_characters():
    eopatch = EOPatch()

    with pytest.raises(ValueError):
        eopatch.data_timeless['mask.npy'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)


def test_repr_no_crs(test_eopatch):
    test_eopatch.vector_timeless['LULC'].crs = None
    repr_str = test_eopatch.__repr__()
    assert isinstance(repr_str, str) and len(repr_str) > 0, \
        'EOPatch __repr__ must return non-empty string even in case of missing crs'


def test_add_feature():
    bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

    eop = EOPatch()
    eop.data['bands'] = bands

    assert np.array_equal(eop.data['bands'], bands), 'Data numpy array not stored'


def test_simplified_feature_operations():
    bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
    feature = FeatureType.DATA, 'TEST-BANDS'
    eop = EOPatch()

    eop[feature] = bands
    assert np.array_equal(eop[feature], bands), 'Data numpy array not stored'


def test_rename_feature():
    bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)

    eop = EOPatch()
    eop.data['bands'] = bands

    eop.rename_feature(FeatureType.DATA, 'bands', 'new_bands')

    assert 'new_bands' in eop.data


def test_rename_feature_missing():
    bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)

    eop = EOPatch()
    eop.data['bands'] = bands

    with pytest.raises(BaseException):
        # Should fail because there is no `missing_bands` feature in the EOPatch.
        eop.rename_feature(FeatureType.DATA, 'missing_bands', 'new_bands')


def test_get_feature():
    bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

    eop = EOPatch()
    eop.data['bands'] = bands

    eop_bands = eop.get_feature(FeatureType.DATA, 'bands')

    assert np.array_equal(eop_bands, bands), 'Data numpy array not returned properly'


def test_remove_feature():
    bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
    names = ['bands1', 'bands2', 'bands3']

    eop = EOPatch()
    eop.add_feature(FeatureType.DATA, names[0], bands)
    eop.data[names[1]] = bands
    eop[FeatureType.DATA][names[2]] = bands

    for feature_name in names:
        assert feature_name in eop.data, f'Feature {feature_name} was not added to EOPatch'
        assert np.array_equal(eop.data[feature_name], bands), f'Data of feature {feature_name} is incorrect'

    eop.remove_feature(FeatureType.DATA, names[0])
    del eop.data[names[1]]
    del eop[FeatureType.DATA][names[2]]
    for feature_name in names:
        assert not (feature_name in eop.data), f'Feature {feature_name} should be deleted from EOPatch'


def test_concatenate():
    eop1 = EOPatch()
    bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
    eop1.data['bands'] = bands1

    eop2 = EOPatch()
    bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
    eop2.data['bands'] = bands2

    eop = EOPatch.concatenate(eop1, eop2)
    assert np.array_equal(eop.data['bands'], np.concatenate((bands1, bands2), axis=0))


def test_concatenate_different_key():
    eop1 = EOPatch()
    bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
    eop1.data['bands'] = bands1

    eop2 = EOPatch()
    bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
    eop2.data['measurements'] = bands2

    eop = EOPatch.concatenate(eop1, eop2)
    assert 'bands' in eop.data and 'measurements' in eop.data, 'Failed to concatenate different features'


def test_concatenate_timeless():
    eop1 = EOPatch()
    mask1 = np.arange(3*3*2).reshape(3, 3, 2)
    eop1.data_timeless['mask1'] = mask1
    eop1.data_timeless['mask'] = 5 * mask1

    eop2 = EOPatch()
    mask2 = np.arange(3*3*2).reshape(3, 3, 2)
    eop2.data_timeless['mask2'] = mask2
    eop2.data_timeless['mask'] = 5 * mask1  # add mask1 to eop2

    eop = EOPatch.concatenate(eop1, eop2)

    for name in ['mask', 'mask1', 'mask2']:
        assert name in eop.data_timeless
    assert np.array_equal(eop.data_timeless['mask'], 5 * mask1), 'Data with same values should stay the same'


def test_concatenate_missmatched_timeless():
    mask = np.arange(3*3*2).reshape(3, 3, 2)

    eop1 = EOPatch()
    eop1.data_timeless['mask'] = mask
    eop1.data_timeless['nask'] = 3 * mask

    eop2 = EOPatch()
    eop2.data_timeless['mask'] = mask
    eop2.data_timeless['nask'] = 5 * mask

    with pytest.raises(ValueError):
        _ = EOPatch.concatenate(eop1, eop2)


def test_equals():
    eop1 = EOPatch(data={'bands': np.arange(2 * 3 * 3 * 2, dtype=np.float32).reshape(2, 3, 3, 2)})
    eop2 = EOPatch(data={'bands': np.arange(2 * 3 * 3 * 2, dtype=np.float32).reshape(2, 3, 3, 2)})
    assert eop1 == eop2

    eop1.data['bands'][1, ...] = np.nan
    assert eop1 != eop2

    eop2.data['bands'][1, ...] = np.nan
    assert eop1 == eop2

    eop1.data['bands'] = np.reshape(eop1.data['bands'], (2, 3, 2, 3))
    assert eop1 != eop2

    eop2.data['bands'] = np.reshape(eop2.data['bands'], (2, 3, 2, 3))
    eop1.data['bands'] = eop1.data['bands'].astype(np.float16)
    assert eop1 != eop2

    del eop1.data['bands']
    del eop2.data['bands']
    assert eop1 == eop2

    eop1.data_timeless['dem'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)

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
        datetime.datetime(2017, 3, 15, 10, 12, 14)
        ]

    data = np.random.rand(10, 100, 100, 3)
    mask = np.random.randint(0, 2, (10, 100, 100, 1))
    mask_timeless = np.random.randint(10, 20, (100, 100, 1))
    scalar = np.random.rand(10, 1)

    eop = EOPatch(
        timestamp=timestamps,
        data={'DATA': data},
        mask={'MASK': mask},
        scalar={'SCALAR': scalar},
        mask_timeless={'MASK_TIMELESS': mask_timeless}
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
    assert np.array_equal(data[1:-1, ...], eop.data['DATA'])
    assert np.array_equal(mask[1:-1, ...], eop.mask['MASK'])
    assert np.array_equal(scalar[1:-1, ...], eop.scalar['SCALAR'])
    assert np.array_equal(mask_timeless, eop.mask_timeless['MASK_TIMELESS'])
