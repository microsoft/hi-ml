import numpy as np
from pathlib import Path

from health.utils.reports import reports as report_util
from health.utils.reports import classification_report as c_report_util

TEST_DATA_FOLDER = Path(__file__).parent.parent / "test_data"


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
