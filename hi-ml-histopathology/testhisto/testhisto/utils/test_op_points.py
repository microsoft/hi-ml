import numpy as np
import pytest

from histopathology.utils.op_points import ConfusionMatrix


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
