import numpy as np

from histopathology.utils.op_points import _searchsorted


def _test_searchsorted(array: np.ndarray, value: float, ascending: bool) -> int:
    index = _searchsorted(array, value, ascending)

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


def test_searchsorted() -> None:
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

    index = _test_searchsorted(array, exact_value, ascending=True)
    assert index == expected_exact_index
    assert array[index] == exact_value

    index = _test_searchsorted(array, intermediate_value, ascending=True)
    assert index == expected_intermediate_index

    index = _test_searchsorted(array, lower_value, ascending=True)
    assert index == expected_lower_index

    index = _test_searchsorted(array, higher_value, ascending=True)
    assert index == expected_higher_index

    # Descending search on ascending array
    index = _test_searchsorted(array, intermediate_value, ascending=False)
    assert index == expected_higher_index

    # Descending array
    reversed_array = array[::-1]

    index = _test_searchsorted(reversed_array, exact_value, ascending=False)
    assert index == length - expected_exact_index - 1
    assert reversed_array[index] == exact_value

    index = _test_searchsorted(reversed_array, intermediate_value, ascending=False)
    assert index == length - expected_intermediate_index

    index = _test_searchsorted(reversed_array, lower_value, ascending=False)
    assert index == length - expected_lower_index

    index = _test_searchsorted(reversed_array, higher_value, ascending=False)
    assert index == length - expected_higher_index

    # Ascending search on descending array
    index = _test_searchsorted(reversed_array, intermediate_value, ascending=True)
    assert index == expected_lower_index
