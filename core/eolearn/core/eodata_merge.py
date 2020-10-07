"""
A module implementing EOPatch merging utility

Credits:
Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Grega Milčinski, Matic Lubej, Devis Peresutti (Sinergise)
Copyright (c) 2017-2020 Nejc Vesel, Jovan Višnjić, Anže Zupanc (Sinergise)
Copyright (c) 2018-2020 William Ouellette

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import pandas as pd

from .constants import FeatureType
from .utilities import FeatureParser, deep_eq


def merge_eopatch(*eopatches, features=..., time_dependent_op='concatenate', timeless_op=None):
    """ Concatenate an existing eopatch in the filesystem and a new eopatch chronologically
    """
    _check_allowed_operations(time_dependent_op, timeless_op)

    eopatch_content = {}
    for eopatch in eopatches:
        for ftype, fname in FeatureParser(features)(eopatch):
            feat = (ftype, fname)
            if ftype.is_time_dependent() and ftype not in [FeatureType.TIMESTAMP, FeatureType.VECTOR]:
                if feat not in eopatch_content:
                    eopatch_content[feat] = [eopatch[feat]]
                else:
                    if eopatch[feat].shape[1:] == eopatch_content[feat][-1].shape[1:]:
                        eopatch_content[feat].append(eopatch[feat])
                    else:
                        raise ValueError(f'The arrays have mismatching n x m x b shapes for {feat}.')
            elif ftype is FeatureType.VECTOR:
                if feat not in eopatch_content:
                    eopatch_content[feat] = [eopatch[feat]]
                else:
                    eopatch_content[feat].append(eopatch[feat])
            elif ftype.is_timeless() and ftype != FeatureType.VECTOR_TIMELESS:
                if feat not in eopatch_content.keys():
                    eopatch_content[feat] = [eopatch[feat]]
                else:
                    if eopatch[feat].shape == eopatch_content[feat][-1].shape:
                        if deep_eq(eopatch[feat], eopatch_content[feat][-1]) and timeless_op:
                            raise ValueError(f'Two identical timeless arrays were found for {feat}.')
                        eopatch_content[feat].append(eopatch[feat])
                    else:
                        raise ValueError(f'The arrays have mismatching n x m x b shapes for {feat}.')

    return _perform_merge_operation(eopatches, eopatch_content, time_dependent_op, timeless_op)


def _check_allowed_operations(time_dependent_op, timeless_op):
    """ Checks that no the passed operations for time-dependent and timeless feature merging are allowed.
    """
    allowed_timeless_op = [None, 'mean', 'max', 'min', 'median']
    allowed_time_dependent_op = ['concatenate', 'mean', 'max', 'min', 'median']

    if timeless_op not in allowed_timeless_op:
        raise ValueError(f'timeless_op "{timeless_op}" is invalid, must be one of {allowed_timeless_op}')

    if time_dependent_op not in allowed_time_dependent_op:
        raise ValueError(f'time_dependent_op "{time_dependent_op}" is invalid, must be one of '
                         f'{allowed_time_dependent_op}')


def _perform_merge_operation(eopatches, eopatch_content, time_dependent_op, timeless_op):
    """Performs the merging of duplicate timestamps of time-dependent features and of timeless features.
    """
    ops = {'mean': np.nanmean,
           'median': np.nanmedian,
           'min': np.nanmin,
           'max': np.nanmax
           }

    all_timestamps = [tstamp for eop in eopatches for tstamp in eop.timestamp]

    timestamp_list = [eop.timestamp for eop in eopatches]
    masks = [np.isin(time_post, list(set(time_post).difference(set(time_pre))))
             for time_pre, time_post in zip(timestamp_list[:-1], timestamp_list[1:])]
    unique_timestamps = eopatches[0].timestamp + [tstamp for eop, mask in zip(eopatches[1:], masks)
                                                  for tstamp, to_keep in zip(eop.timestamp, mask) if to_keep]

    unique_indices = sorted(range(len(unique_timestamps)), key=lambda k: unique_timestamps[k])

    eopatch = eopatches[0].__copy__()
    eopatch.timestamp = sorted(unique_timestamps)

    for feat, arrays in eopatch_content.items():
        ftype, _ = feat
        if ftype.is_time_dependent() and ftype != FeatureType.VECTOR and unique_indices:
            eopatch[feat] = np.concatenate(arrays, axis=0)
            for idx, timestamp in zip(unique_indices, eopatch.timestamp):
                array = eopatch[feat][[i for i, t in enumerate(all_timestamps) if t == timestamp]]
                _check_duplicate_timestamp(array, feat)
                eopatch[feat][idx] = ops[time_dependent_op](array, axis=0) if time_dependent_op in ops else array[0]
            eopatch[feat] = eopatch[feat][unique_indices]
        elif ftype.is_timeless():
            eopatch[feat] = ops[timeless_op](arrays, axis=0) if timeless_op in ops else arrays[0]
        elif ftype == FeatureType.VECTOR:
            eopatch[feat] = pd.concat([array for array in arrays if not array.empty]).sort_values('TIMESTAMP')\
                .drop_duplicates("TIMESTAMP")

    return eopatch


def _check_duplicate_timestamp(array, feature):
    """ Checks that no duplicate timestamps with different values exist
    """
    if any(not deep_eq(array[x], array[y]) for i, x in enumerate(range(array.shape[0]))
           for j, y in enumerate(range(array.shape[0])) if i != j):
        raise ValueError(f'Two identical timestamps with different values were found for {feature}.')
