import sys

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from health.utils import reports as report_util


@pytest.mark.parametrize("args, attr_name, attr_val", [
    (["", "--output_folder", "report_folder"], "output_folder", "report_folder"),
    ([""], "output_folder", "reports"),
    # ([], "", "")
])
def test_create_args(args, attr_name, attr_val):
    with patch.object(sys, 'argv', args):
        args = report_util.parse_arguments(sys.argv[1:])
        assert hasattr(args, attr_name)
        assert getattr(args, attr_name) == attr_val


def test_add_table_from_dataframe(tmp_path: Path):
    df = pd.DataFrame({
        "X": [1, 2, 3],
        "Y": ["a", "b", "c"]
    })
    report_title = "My Report"
    report_output_folder = tmp_path / "reports"
    report = report_util.initialize_report(report_title=report_title, output_folder= report_output_folder)

    report.add_table_from_dataframe(df)

    report_path = report.save_report()
    stripped_title = report_title.lower().replace(" ", "_")
    expected_report_path = report_output_folder / stripped_title / (stripped_title + ".pdf")

    assert report_path == expected_report_path
    assert expected_report_path.exists()
    assert expected_report_path.stat().st_size > 0


@pytest.mark.parametrize("string_to_test, font_size, expected_width", [
    # ("abc", 10, 30),
    ("16/02/2015", 5, 50)
])
def test_get_string_width_(string_to_test: str, font_size: int, expected_width: int):
    report = report_util.Report()
    report.set_font_size(font_size)
    assert report.get_string_width_(string_to_test) == expected_width