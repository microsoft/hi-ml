import numpy as np
import pytest

from histopathology.utils.op_points import ConfusionMatrix, _is_sorted, _searchsorted


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


def test_is_sorted() -> None:
    length = 5
    ascending_array = np.arange(length)
    descending_array = ascending_array[::-1]
    unsorted_array = ascending_array % (length // 2)

    assert _is_sorted(ascending_array, ascending=True)
    assert not _is_sorted(ascending_array, ascending=False)

    assert not _is_sorted(descending_array, ascending=True)
    assert _is_sorted(descending_array, ascending=False)

    assert not _is_sorted(unsorted_array, ascending=True)
    assert not _is_sorted(unsorted_array, ascending=False)


@pytest.fixture
def conf_matrix() -> ConfusionMatrix:
    num_total = 100
    true_labels = np.random.randint(2, size=num_total)
    num_positives = true_labels.sum()
    num_negatives = num_total - num_positives
    pred_scores = np.zeros(num_total)
    pred_scores[true_labels == 1] = np.random.randn(num_positives) + 1
    pred_scores[true_labels == 0] = np.random.randn(num_negatives) - 1
    return ConfusionMatrix.from_labels_and_scores(true_labels, pred_scores)


def test_confusion_matrix_levels(conf_matrix: ConfusionMatrix) -> None:
    level = 0.9
    cm_sen90 = conf_matrix.at_sensitivity(level)
    assert abs(cm_sen90.sensitivity - level) < 1 / conf_matrix.num_positives

    cm_spc90 = conf_matrix.at_specificity(level)
    assert abs(cm_spc90.specificity - level) < 1 / conf_matrix.num_negatives


NON_INDEXABLE_ATTRS = [
    'num_total',
    'num_positives',
    'num_negatives',
    'prevalence',
]
INDEXABLE_ATTRS = [
    'true_positives',
    'false_positives',
    'true_negatives',
    'false_negatives',
    'pred_positives',
    'pred_negatives',
    'sensitivity',
    'specificity',
    'pos_pred_value',
    'neg_pred_value',
]


def test_confusion_matrix_int_indexing(conf_matrix: ConfusionMatrix) -> None:
    arbitrary_index = 42
    indexed_conf_matrix = conf_matrix[arbitrary_index]
    for attr in NON_INDEXABLE_ATTRS:
        assert getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)
    for attr in INDEXABLE_ATTRS:
        assert getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)[arbitrary_index]


def test_confusion_matrix_slice_indexing(conf_matrix: ConfusionMatrix) -> None:
    arbitrary_slice = slice(24, 42, 2)
    indexed_conf_matrix = conf_matrix[arbitrary_slice]
    for attr in NON_INDEXABLE_ATTRS:
        assert getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)
    for attr in INDEXABLE_ATTRS:
        assert np.all(getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)[arbitrary_slice])
