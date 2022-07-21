"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import uuid
from typing import Sequence, Union, cast

import geopandas as gpd
import numpy as np
from geopandas.testing import assert_geodataframe_equal


def deep_eq(fst_obj: object, snd_obj: object) -> bool:
    """Compares whether fst_obj and snd_obj are deeply equal.

    In case when both fst_obj and snd_obj are of type np.ndarray or either np.memmap, they are compared using
    np.array_equal(fst_obj, snd_obj). Otherwise, when they are lists or tuples, they are compared for length and then
    deep_eq is applied component-wise. When they are dict, they are compared for key set equality, and then deep_eq is
    applied value-wise. For all other data types that are not list, tuple, dict, or np.ndarray, the method falls back
    to the __eq__ method.

    Because np.ndarray is not a hashable object, it is impossible to form a set of numpy arrays, hence deep_eq works
    correctly.

    :param fst_obj: First object compared
    :param snd_obj: Second object compared
    :return: `True` if objects are deeply equal, `False` otherwise
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(fst_obj, type(snd_obj)):
        return False

    if isinstance(fst_obj, np.ndarray):
        snd_obj = cast(np.ndarray, snd_obj)
        if fst_obj.dtype != snd_obj.dtype:
            return False
        fst_nan_mask = np.isnan(fst_obj)
        snd_nan_mask = np.isnan(snd_obj)
        return np.array_equal(fst_obj[~fst_nan_mask], snd_obj[~snd_nan_mask]) and np.array_equal(
            fst_nan_mask, snd_nan_mask
        )

    if isinstance(fst_obj, gpd.GeoDataFrame):
        try:
            # We allow differences in index types and in dtypes of columns
            assert_geodataframe_equal(fst_obj, snd_obj, check_index_type=False, check_dtype=False)
            return True
        except AssertionError:
            return False

    if isinstance(fst_obj, (tuple, list)):
        snd_obj = cast(Sequence, snd_obj)

        return len(fst_obj) == len(snd_obj) and all(map(deep_eq, fst_obj, snd_obj))

    if isinstance(fst_obj, dict):
        snd_obj = cast(dict, snd_obj)

        if fst_obj.keys() != snd_obj.keys():
            return False

        return all(deep_eq(fst_obj[key], snd_obj[key]) for key in fst_obj)

    return fst_obj == snd_obj


def generate_uid(prefix: str) -> str:
    """Generates a (sufficiently) unique ID starting with the `prefix`.

    The ID is composed of the prefix, a hexadecimal string obtained from the current time and a random hexadecimal
    string. This makes the uid sufficiently unique.
    """
    time_uid = uuid.uuid1(node=0).hex[:-12]
    random_uid = uuid.uuid4().hex[:12]
    return f"{prefix}-{time_uid}-{random_uid}"


def is_discrete_type(number_type: Union[np.dtype, type]) -> bool:
    """Checks if a given `numpy` type is a discrete numerical type."""
    return np.issubdtype(number_type, np.integer) or np.issubdtype(number_type, bool)
