#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import numpy as np

from histopathology.utils.array_utils import is_sorted_1d, search_1d


def _test_search_1d(array: np.ndarray, value: float, ascending: bool) -> int:
    index = search_1d(array, value, ascending)

    length = len(array)
    assert 0 <= index <= length

    if ascending:
        if index > 0:
            assert array[index - 1] < value
        if index < length:
            assert value <= array[index]
    else:
        if index > 0:
            assert value < array[index - 1]
        if index < length:
            assert array[index] <= value

    return index


def test_search_1d() -> None:
    length = 5
    array = np.arange(length, dtype=float)

    expected_exact_index = length // 2
    expected_intermediate_index = expected_exact_index
    expected_lower_index = 0
    expected_higher_index = length

    exact_value = array[expected_exact_index]
    if length > 1:
        intermediate_value = (array[expected_intermediate_index - 1] + array[expected_intermediate_index]) / 2
    else:
        intermediate_value = array[0] - 0.5
    lower_value = array.min() - 1
    higher_value = array.max() + 1

    index = _test_search_1d(array, exact_value, ascending=True)
    assert index == expected_exact_index
    assert array[index] == exact_value

    index = _test_search_1d(array, intermediate_value, ascending=True)
    assert index == expected_intermediate_index

    index = _test_search_1d(array, lower_value, ascending=True)
    assert index == expected_lower_index

    index = _test_search_1d(array, higher_value, ascending=True)
    assert index == expected_higher_index

    # Descending search on ascending array
    index = _test_search_1d(array, intermediate_value, ascending=False)
    assert index == expected_higher_index

    # Descending array
    reversed_array = array[::-1]

    index = _test_search_1d(reversed_array, exact_value, ascending=False)
    assert index == length - expected_exact_index - 1
    assert reversed_array[index] == exact_value

    index = _test_search_1d(reversed_array, intermediate_value, ascending=False)
    assert index == length - expected_intermediate_index

    index = _test_search_1d(reversed_array, lower_value, ascending=False)
    assert index == length - expected_lower_index

    index = _test_search_1d(reversed_array, higher_value, ascending=False)
    assert index == length - expected_higher_index

    # Ascending search on descending array
    index = _test_search_1d(reversed_array, intermediate_value, ascending=True)
    assert index == expected_lower_index


def test_is_sorted() -> None:
    length = 5
    ascending_array = np.arange(length)
    descending_array = ascending_array[::-1]
    unsorted_array = ascending_array % (length // 2)

    assert is_sorted_1d(ascending_array, ascending=True)
    assert not is_sorted_1d(ascending_array, ascending=False)

    assert not is_sorted_1d(descending_array, ascending=True)
    assert is_sorted_1d(descending_array, ascending=False)

    assert not is_sorted_1d(unsorted_array, ascending=True)
    assert not is_sorted_1d(unsorted_array, ascending=False)
