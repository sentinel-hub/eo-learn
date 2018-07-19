"""
The utilities module is a collection of methods used across the eolearn package, such as checking whether two objects
are deeply equal, padding of an image, etc.
"""

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def get_common_timestamps(source, target):
    """
    Return indices of timestamps from source that are also found in target.

    :param source: timestamps from source
    :type source: list of datetime objects
    :param target: timestamps from target
    :type target: list of datetime objects
    :return: indices of timestamps from source that are also found in target
    :rtype: list of ints
    """
    remove_from_source = set(source).difference(target)
    remove_from_source_idxs = [source.index(rm_date) for rm_date in remove_from_source]
    return [idx for idx, _ in enumerate(source) if idx not in remove_from_source_idxs]


def deep_eq(fst_obj, snd_obj):
    """
    Compares whether fst_obj and snd_obj are deeply equal. In case when both fst_obj and snd_obj are of type np.ndarray,
    they are compared using np.array_equal(fst_obj, snd_obj).

    Otherwise, when they are lists or tuples, they are compared for length and then
    deep_eq is applied component-wise.

    When they are dict, they are compared for key set equality, and then deep_eq is applied value-wise.

    For all other data types that are not list, tuple, dict, or np.ndarray, the method falls back to the
    good old __eq__.

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
        return np.array_equal(fst_obj, snd_obj)

    if isinstance(fst_obj, (list, tuple)):
        if len(fst_obj) != len(snd_obj):
            return False

        for element_fst, element_snd in zip(fst_obj, snd_obj):
            if not deep_eq(element_fst, element_snd):
                return False
        return True

    if isinstance(fst_obj, dict):
        if fst_obj.keys() != snd_obj.keys():
            return False

        for key in fst_obj:
            if not deep_eq(fst_obj[key], snd_obj[key]):
                return False
        return True

    return fst_obj == snd_obj


def negate_mask(mask):
    """
    Returns the negated mask. If elements of input mask have 0 and non-zero values, then
    the returned matrix will have all elements 0 (1) where the original one has non-zero (0).

    Parameters:
    -----------

    mask: array, any shape
        Input mask

    Returns:
    -----------

    array of same shape and dtype=int8 as input array
    """
    res = np.ones(mask.shape, dtype=np.int8)
    res[mask > 0] = 0

    return res


def constant_pad(X, multiple_of, up_down_rule='even', left_right_rule='even', pad_value=0):
    """
    Function pads an image of shape (rows, columns, channels) with zeros so that the shape
    becomes (rows + padded_rows, columns + padded_columns, channels), where

    padded_rows = (int(rows/multiple_of[0]) + 1)*multiple_of[0] - rows

    Same rule is applied to columns.


    Parameters:
    -----------

    X: array of shape (rows, columns, channels) or (rows, columns)

    multiple_of: tuple (rows, columns)
        make X' rows and columns multiple of this tuple

    up_down_rule: string, (even, up, down)
        Add padded rows evenly to the top/bottom of the image, or up (top) / down (bottom) only

    up_down_rule: string, (even, left, right)
        Add padded columns evenly to the left/right of the image, or left / right only

    pad_value: int,
        Value to be assigned to padded rows and columns
    """
    # pylint: disable=invalid-name
    shape = X.shape

    row_padding, col_padding = 0, 0

    if shape[0] % multiple_of[0]:
        row_padding = (int(shape[0] / multiple_of[0]) + 1) * multiple_of[0] - shape[0]

    if shape[1] % multiple_of[1]:
        col_padding = (int(shape[1] / multiple_of[1]) + 1) * multiple_of[1] - shape[1]

    row_padding_up, row_padding_down, col_padding_left, col_padding_right = 0, 0, 0, 0

    if row_padding > 0:
        if up_down_rule == 'up':
            row_padding_up = row_padding
        elif up_down_rule == 'down':
            row_padding_down = row_padding
        elif up_down_rule == 'even':
            row_padding_up = int(row_padding / 2)
            row_padding_down = row_padding_up + (row_padding % 2)
        else:
            raise ValueError('Padding rule for rows not supported. Choose beteen even, down or up!')

    if col_padding > 0:
        if left_right_rule == 'left':
            col_padding_left = col_padding
        elif left_right_rule == 'right':
            col_padding_right = col_padding
        elif left_right_rule == 'even':
            col_padding_left = int(col_padding / 2)
            col_padding_right = col_padding_left + (col_padding % 2)
        else:
            raise ValueError('Padding rule for columns not supported. Choose beteen even, left or right!')

    return np.lib.pad(X, ((row_padding_up, row_padding_down), (col_padding_left, col_padding_right)),
                      'constant', constant_values=((pad_value, pad_value), (pad_value, pad_value)))
