import numpy as np
from pathlib import Path

from health.utils.reports import reports as report_util
from health.utils.reports import classification_report as c_report_util

TEST_DATA_FOLDER = Path(__file__).parent.parent / "test_data"


def test_get_correct_and_misclassified_examples() -> None:
    test_metrics_file = TEST_DATA_FOLDER / "test_metrics_classification.csv"
    val_metrics_file = TEST_DATA_FOLDER / "val_metrics_classification.csv"

    results = c_report_util.get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_file,
                                                                   test_metrics_csv=test_metrics_file)

    assert results is not None  # for mypy

    true_positives = [item[c_report_util.LoggingColumns.Patient.value] for _, item in results.true_positives.iterrows()]
    assert all([i in true_positives for i in [3, 4, 5]])

    true_negatives = [item[c_report_util.LoggingColumns.Patient.value] for _, item in results.true_negatives.iterrows()]
    assert all([i in true_negatives for i in [6, 7, 8]])

    false_positives = [item[c_report_util.LoggingColumns.Patient.value] for _, item in
                       results.false_positives.iterrows()]
    assert all([i in false_positives for i in [9, 10, 11]])

    false_negatives = [item[c_report_util.LoggingColumns.Patient.value] for _, item in
                       results.false_negatives.iterrows()]
    assert all([i in false_negatives for i in [0, 1, 2]])


def test_get_k_best_and_worst_performing(test_metrics_file, val_metrics_file) -> None:

    results = c_report_util.get_k_best_and_worst_performing(val_metrics_csv=val_metrics_file,
                                                            test_metrics_csv=test_metrics_file,
                                                            k=2)
    assert results is not None  # for mypy

    best_true_positives = [item[report_util.LoggingColumns.Patient.value] for _, item in
                           results.true_positives.iterrows()]
    assert best_true_positives == [5, 4]

    best_true_negatives = [item[report_util.LoggingColumns.Patient.value] for _, item in
                           results.true_negatives.iterrows()]
    assert best_true_negatives == [6, 7]

    worst_false_positives = [item[report_util.LoggingColumns.Patient.value] for _, item in
                             results.false_positives.iterrows()]
    assert worst_false_positives == [11, 10]

    worst_false_negatives = [item[report_util.LoggingColumns.Patient.value] for _, item in
                             results.false_negatives.iterrows()]
    assert worst_false_negatives == [0, 1]


def test_add_table_of_best_and_worst_performing(tmp_path: Path):
    test_metrics_file = TEST_DATA_FOLDER / "test_metrics_classification.csv"
    val_metrics_file = TEST_DATA_FOLDER / "val_metrics_classification.csv"
    num_best_and_worst = 5
    prediction_target = "Default"

    report = report_util.initialize_report(output_folder=tmp_path)
    c_report_util.add_table_of_best_and_worst_performing(val_metrics_file, test_metrics_file, num_best_and_worst,
                                                         prediction_target, report)
    report_path = report.save_report()
    assert report_path.exists()
