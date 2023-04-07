#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from health_azure.utils import create_aml_run_object
from health_ml.experiment_config import ExperimentConfig
from health_ml.training_runner import TrainingRunner
from health_ml.configs.hello_world import HelloWorld
from health_ml.utils.regression_test_utils import (
    CONTENTS_MISMATCH,
    FILE_FORMAT_ERROR,
    MISSING_FILE,
    REGRESSION_TEST_OUTPUT_FOLDER,
    REGRESSION_TEST_AZUREML_FOLDER,
    REGRESSION_TEST_AZUREML_PARENT_FOLDER,
    REGRESSION_TEST_METRICS_FILENAME,
    TEXT_FILE_SUFFIXES,
    _compare_metric_values,
    _compare_metrics_list,
    _is_nested_dict,
    _load_json_dict,
    compare_dictionaries,
    compare_files,
    compare_folder_contents,
    compare_folders_and_run_outputs,
    compare_metrics_dictionaries,
    compare_metrics_files,
)
from testazure.utils_testazure import DEFAULT_WORKSPACE, experiment_for_unittests


def create_folder_and_write_text(file: Path, text: str) -> None:
    """
    Writes the given text to a file. The folders in which the file lives are created too, unless they exist already.
    Writing the text keeps the line separators as-is (no translation).
    """
    file.parent.mkdir(exist_ok=True, parents=True)
    with file.open(mode="w", newline="") as f:
        f.write(text)


def test_regression_test() -> None:
    """
    Test that the file comparison for regression tests is actually called in the workflow.
    """
    container = HelloWorld()
    container.regression_test_folder = Path(str(uuid.uuid4().hex))
    runner = TrainingRunner(container=container, experiment_config=ExperimentConfig())
    runner.setup()
    with pytest.raises(ValueError) as ex:
        runner.run()
    assert "Folder with expected files does not exist" in str(ex)


@pytest.mark.parametrize("file_extension", TEXT_FILE_SUFFIXES)
def test_compare_files_text(tmp_path: Path, file_extension: str) -> None:
    """
    Checks the basic code to compare the contents of two text files.
    :param test_output_dirs:
    :param file_extension: The extension of the file to create.
    """
    with mock.patch("json.loads"):
        expected = tmp_path / f"expected{file_extension}"
        actual = tmp_path / "actual.does_not_matter"
        # Make sure that we test different line endings - the files should still match
        create_folder_and_write_text(expected, "Line1\r\nLine2")
        create_folder_and_write_text(actual, "Line1\nLine2")
        assert compare_files(expected=expected, actual=actual) == ""
        actual.write_text("does_not_match")
        assert compare_files(expected=expected, actual=actual) == CONTENTS_MISMATCH


def test_compare_files_json(tmp_path: Path) -> None:
    tmp_json_dir = tmp_path / "json_files"
    tmp_json_dir.mkdir()
    dict1 = {"a": [1.0, 2.0, 3.0], "b": 4}
    dict2 = {"a": [1.0, 2.0, 3.0], "b": 4}
    tmp_json_path1 = tmp_json_dir / "tmp_json1.json"
    with open(tmp_json_path1, "w+") as f_path:
        json.dump(dict1, f_path)
    tmp_json_path2 = tmp_json_dir / "tmp_json2.json"
    with open(tmp_json_path2, "w+") as f_path:
        json.dump(dict2, f_path)
    assert compare_files(expected=tmp_json_path1, actual=tmp_json_path2) == ""

    # now test the case where the dictionaries are different
    dict3 = {"c": [5.0, 6.0]}
    tmp_json_path3 = tmp_json_dir / "tmp_json2.json"
    with open(tmp_json_path3, "w+") as f_path:
        json.dump(dict3, f_path)
    err = compare_files(expected=tmp_json_path1, actual=tmp_json_path3)
    assert err == CONTENTS_MISMATCH


def test_compare_files_csv(tmp_path: Path) -> None:
    expected = tmp_path / "expected.csv"
    actual = tmp_path / "actual.does_not_matter"
    expected.write_text(
        """foo,bar
1.0,10.0"""
    )
    actual.write_text(
        """foo,bar
1.0001,10.001"""
    )
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=1e-2) == ""
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=1e-3) == ""
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=2e-4) == ""
    assert compare_files(expected=expected, actual=actual, csv_relative_tolerance=9e-5) == CONTENTS_MISMATCH


def test_compare_files_empty_csv(tmp_path: Path) -> None:
    """
    If either of the two CSV files is empty, it should not raise an error, but exit gracefully.
    """
    expected = tmp_path / "expected.csv"
    actual = tmp_path / "actual.csv"
    valid_csv = """foo,bar
    1.0,10.0"""
    empty_csv = ""
    for expected_contents, actual_contents in [(empty_csv, empty_csv), (valid_csv, empty_csv), (empty_csv, valid_csv)]:
        expected.write_text(expected_contents)
        actual.write_text(actual_contents)
        assert compare_files(expected=expected, actual=actual) == FILE_FORMAT_ERROR
    expected.write_text(valid_csv)
    actual.write_text(valid_csv)
    assert compare_files(expected=expected, actual=actual) == ""


@pytest.mark.parametrize("file_extension", [".png", ".whatever"])
def test_compare_files_binary(tmp_path: Path, file_extension: str) -> None:
    """
    Checks the comparison of files that are not recognized as text files, for example images.
    :param tmp_path: A folder for temporary files
    :param file_extension: The extension of the file to create.
    """
    expected = tmp_path / f"expected{file_extension}"
    actual = tmp_path / "actual.does_not_matter"
    data1 = bytes([1, 2, 3])
    data2 = bytes([4, 5, 6])
    expected.write_bytes(data1)
    actual.write_bytes(data1)
    assert compare_files(expected=expected, actual=actual) == ""
    actual.write_bytes(data2)
    assert compare_files(expected=expected, actual=actual) == CONTENTS_MISMATCH


def test_compare_folder(tmp_path: Path) -> None:
    """
    Test the comparison of folders that we use for regression tests.
    """
    # Create a test of expected and actual files on the fly.
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    matching = "matching.txt"
    missing = "missing.txt"
    ignored = "ignored.txt"
    # Comparison should cover at least .csv and .txt files
    mismatch = "mismatch.csv"
    extra = "extra.txt"
    subfolder = Path("folder")
    # This file exists in both expected and actual, should not raise any alerts because it contents matches
    # apart from linebreaks
    create_folder_and_write_text(expected / subfolder / matching, "Line1\r\nLine2")
    create_folder_and_write_text(actual / subfolder / matching, "Line1\nLine2")
    # This file only exists in the expected results, and should create an error saying that it is missing
    # from the actual results
    (expected / subfolder / missing).write_text("missing")
    # This file exists only in the actual results, and not the expected results, and so should not create an error.
    (actual / extra).write_text("extra")
    # This file exists in both actual and expected, but has different contents, hence should create an error
    (expected / subfolder / mismatch).write_text("contents1")
    (actual / subfolder / mismatch).write_text("contents2")

    messages = compare_folder_contents(expected_folder=expected, actual_folder=actual)
    all_messages = " ".join(messages)
    # No issues expected
    assert matching not in all_messages
    assert extra not in all_messages
    assert ignored not in all_messages
    # Folders should be skipped in the comparison
    assert f"{MISSING_FILE}: {subfolder}" not in messages
    assert f"{MISSING_FILE}: {subfolder}/{missing}" in messages
    assert f"{CONTENTS_MISMATCH}: {subfolder}/{mismatch}" in messages


@pytest.mark.fast
def test_compare_folder_metrics(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Compare folders that contain regression test metrics. When the metrics don't match, only the metric
    comparison should trigger, but not the json file comparison."""
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    # This file exists in both expected and actual, should not raise any alerts because it contents matches
    # apart from linebreaks
    create_folder_and_write_text(expected / REGRESSION_TEST_METRICS_FILENAME, '{"metric1": 123}')
    create_folder_and_write_text(actual / REGRESSION_TEST_METRICS_FILENAME, '{"metric1": 456}')

    with caplog.at_level(logging.WARNING):
        messages = compare_folder_contents(expected_folder=expected, actual_folder=actual)
    assert len(messages) == 1
    assert messages[0] == "Mismatch for 1 out of 1 metrics: regression_metrics.json"
    assert len(caplog.messages) == 2
    assert caplog.messages[0] == "Metric 'metric1': Expected 123 but got 456 (allowed tolerance 0.0)"
    assert caplog.messages[1] == "File regression_metrics.json: Mismatch for 1 out of 1 metrics"


def test_compare_plain_outputs(tmp_path: Path) -> None:
    """
    Test if we can compare that a set of files from the job outputs.
    """
    expected_root = tmp_path / "expected"
    expected = expected_root / REGRESSION_TEST_OUTPUT_FOLDER
    actual = tmp_path / "my_output"
    for folder in [expected, actual]:
        file1 = folder / "output.txt"
        create_folder_and_write_text(file1, "Something")
    # First comparison should pass
    compare_folders_and_run_outputs(expected=expected_root, actual=actual)
    # Now add a file to the set of expected files that does not exist in the run: comparison should now fail
    no_such_file = "no_such_file.txt"
    file2 = expected / no_such_file
    create_folder_and_write_text(file2, "foo")
    with pytest.raises(ValueError) as ex:
        compare_folders_and_run_outputs(expected=expected_root, actual=actual)
    message = ex.value.args[0].splitlines()
    assert f"{MISSING_FILE}: {no_such_file}" in message


def test_compare_folder_against_run(tmp_path: Path) -> None:
    """
    Test if we can compare that a set of files exists in an AML run.
    """
    upload_to_run_and_compare(
        regression_test_subfolder=REGRESSION_TEST_AZUREML_FOLDER,
        run_to_mock="RUN_CONTEXT",
        tmp_path=tmp_path,
    )


def test_compare_folder_against_run_parent(tmp_path: Path) -> None:
    """
    Test if we can compare that a set of files exists in an AML run.
    """
    upload_to_run_and_compare(
        regression_test_subfolder=REGRESSION_TEST_AZUREML_PARENT_FOLDER,
        run_to_mock="PARENT_RUN_CONTEXT",
        tmp_path=tmp_path,
    )


def upload_to_run_and_compare(regression_test_subfolder: str, run_to_mock: str, tmp_path: Path) -> None:
    """Creates a set of files in an AzureML run, and checks if the comparison tools accept/fail
    in the right way.

    :param regression_test_subfolder: The subfolder of the regression test results where the files
    should be created (either REGRESSION_TEST_AZUREML_FOLDER or REGRESSION_TEST_AZUREML_PARENT_FOLDER)
    :param run_to_mock: either RUN_CONTEXT or PARENT_RUN_CONTEXT
    :param tmp_path: A temporary folder to use
    """
    file_contents = "some file contents"
    file_name = "contents.txt"
    regression_test_folder = tmp_path / "expected"
    run = create_aml_run_object(
        workspace=DEFAULT_WORKSPACE.workspace,
        experiment_name=experiment_for_unittests(),
        run_name="upload_to_run_and_compare",
    )
    # Upload a single file to the newly created run. When comparing the run output files,
    # and seeing this in the set of files that are expected to exist on the run, this should pass.
    file1 = tmp_path / file_name
    create_folder_and_write_text(file1, file_contents)
    run.upload_file(file_name, str(file1))
    run.flush()
    file1_expected = regression_test_folder / regression_test_subfolder / file_name
    create_folder_and_write_text(file1_expected, file_contents)

    with mock.patch("health_ml.utils.regression_test_utils." + run_to_mock, run):
        # First comparison only on the single file should pass. Value passed for the 'actual' argument is irrelevant.
        compare_folders_and_run_outputs(expected=regression_test_folder, actual=Path.cwd())
        # Now add a file to the set of expected files that does not exist in the run: comparison should now fail
        no_such_file = "no_such_file.txt"
        file2_expected = regression_test_folder / regression_test_subfolder / no_such_file
        create_folder_and_write_text(file2_expected, "foo")
        with pytest.raises(ValueError) as ex:
            compare_folders_and_run_outputs(expected=regression_test_folder, actual=Path.cwd())
        message = ex.value.args[0].splitlines()
        assert f"{MISSING_FILE}: {no_such_file}" in message
    # Now run the same comparison that failed previously, without mocking. This should now
    # realize that the present run is an offline run, and skip the comparison
    compare_folders_and_run_outputs(expected=regression_test_folder, actual=Path.cwd())
    # Mark the run as complete otherwise will show as running in AML
    run.flush()
    run.complete()


@pytest.mark.fast
@pytest.mark.parametrize(
    "dict1, dict2, should_pass, expected_warnings",
    [
        ({"a": [1.0, 2.0, 3.0], "b": 4}, {"a": [1.0, 2.0, 3.0], "b": 4}, True, ""),
        (
            {"a": [1.0, 2.0, 3.0], "b": 4},
            {"c": [1.0, 2.0, 3.0]},
            False,
            ["Key a is expected but not found in actual", "Key b is expected but not found in actual"],
        ),
        ({"c": "hello"}, {"c": "hello"}, True, ""),
        ({"d": {"a": [1, 2, 3]}}, {"d": {"a": [1, 2, 3]}}, True, ""),
        ({"0": {"a": 0.1, "b": 0.5}}, {"0": {"a": 0.1}}, False, "Key b is expected but not found in actual"),
        ({"0": {"a": 0.1}}, {"0": {"a": 0.1, "b": 0.5}}, True, ""),
    ],
)
def test_compare_dictionaries(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    should_pass: bool,
    expected_warnings: List[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        compare_dictionaries(dict1, dict2)
        if should_pass:
            assert len(caplog.text) == 0
        else:
            for expected_warning in expected_warnings:
                assert expected_warning in caplog.text
    caplog.clear()


@pytest.mark.fast
@pytest.mark.parametrize(
    "expected",
    [
        {"a": 1},
        {"a": [1.0]},
        {"a": [1.0, 2.0]},
    ],
)
def test_compare_metrics_dictionaries_matches(expected: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    """Test simple cases where a metrics dictionary matches its expected value"""
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, expected) == ""
    assert len(caplog.records) == 0


@pytest.mark.fast
def test_compare_metrics_dictionaries_numeric(caplog: pytest.LogCaptureFixture) -> None:
    """Test comparing metrics dictionaries with numeric values, matching and not matching"""
    expected = {"a": 1.0}
    tol = 1e-3
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, {"a": 1 + 0.9 * tol}, tolerance=tol) == ""
    assert len(caplog.records) == 0

    with caplog.at_level(logging.WARNING):
        assert (
            compare_metrics_dictionaries(expected, {"a": 1 + 1.1 * tol}, tolerance=tol)
            == "Mismatch for 1 out of 1 metrics"
        )
    assert len(caplog.records) == 1
    assert caplog.messages[0] == "Metric 'a': Expected 1.0 but got 1.0011 (allowed tolerance 0.001)"


@pytest.mark.fast
def test_compare_metrics_dictionaries_list(caplog: pytest.LogCaptureFixture) -> None:
    """Test comparing metrics dictionaries with numeric values, matching and not matching"""
    expected = {"a": 1.0}
    tol = 1e-3
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, {"a": 1 + 0.9 * tol}, tolerance=tol) == ""
    assert len(caplog.records) == 0

    with caplog.at_level(logging.WARNING):
        assert (
            compare_metrics_dictionaries(expected, {"a": 1 + 1.1 * tol}, tolerance=tol)
            == "Mismatch for 1 out of 1 metrics"
        )
    assert len(caplog.records) == 1
    assert caplog.messages[0] == "Metric 'a': Expected 1.0 but got 1.0011 (allowed tolerance 0.001)"


@pytest.mark.fast
def test_compare_metrics_dictionaries_invalid_expected() -> None:
    """Test for metrics dictionaries where an invalid expected value is passed"""
    expected = {"a": False}
    with pytest.raises(ValueError, match="Expected value has type bool which is not handled."):
        compare_metrics_dictionaries(expected, {})


@pytest.mark.fast
def test_compare_metrics_dictionaries_invalid_actual(caplog: pytest.LogCaptureFixture) -> None:
    """Test for metrics dictionaries where an invalid actual value is passed"""
    expected = {"a": 1.0}
    actual = {"a": False}
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, actual) == "Mismatch for 1 out of 1 metrics"
    assert len(caplog.records) == 1
    assert caplog.messages[0] == "Metric 'a': Actual value has type bool which is not handled."


@pytest.mark.fast
def test_compare_metrics_dictionaries_missing(caplog: pytest.LogCaptureFixture) -> None:
    """Test for metrics dictionaries where no data is present for an expected metric"""
    expected = {"a": 1.0}
    actual: Dict[str, Any] = {}
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, actual) == "Mismatch for 1 out of 1 metrics"
    assert len(caplog.records) == 1
    assert caplog.messages[0] == "Metric 'a': No data found in actual metrics."


@pytest.mark.fast
def test_compare_metrics_dictionaries_type_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    """Test for metrics dictionaries where the types of expected and actual value don't match"""
    expected = {"a": 1.0}
    actual = {"a": "foo"}
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, actual) == "Mismatch for 1 out of 1 metrics"
    assert len(caplog.records) == 1
    assert caplog.messages[0] == "Metric 'a': Actual value has type str but we expected float."


@pytest.mark.fast
def test_compare_metrics_dictionaries_lists(caplog: pytest.LogCaptureFixture) -> None:
    """Test for metrics dictionaries with lists"""
    expected = {"a": [1.0, 2.0]}
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, expected) == ""
    assert len(caplog.records) == 0

    tol = 1e-3
    expected = {"a": [1.0, 1.0]}
    actual = {"a": [1.0 + 0.9 * tol, 1.0 + 1.1 * tol]}
    with caplog.at_level(logging.WARNING):
        assert compare_metrics_dictionaries(expected, actual, tolerance=tol) == "Mismatch for 1 out of 1 metrics"
    assert len(caplog.records) == 1
    assert caplog.messages[0] == "Metric 'a': Index 1: Expected 1.0 but got 1.0011 (allowed tolerance 0.001)"


@pytest.mark.fast
@pytest.mark.parametrize(
    "expected, actual, tol, expected_result",
    [
        (1.0, 1.0, 1e-3, ""),
        (1.0, 1.01, 1e-3, "Expected 1.0 but got 1.01 (allowed tolerance 0.001)"),
        (100.0, 100.09, 1e-3, ""),
        (100.0, 101, 1e-3, "Expected 100.0 but got 101 (allowed tolerance 0.001)"),
        ("foo", 1.0, 1e-3, "Don't know how to handle expected value of type str"),
        (1.0, "bar", 1e-3, "Expected a numeric value, but got type str"),
    ],
)
def test_compare_metric_values(expected: Any, actual: Any, tol: float, expected_result: str) -> None:
    result = _compare_metric_values(expected, actual, tol)
    assert result == expected_result


@pytest.mark.fast
@pytest.mark.parametrize(
    "expected, actual, tol, expected_result",
    [
        ([], [], 1e-3, []),
        ([], [1], 1e-3, ["Expected list of length 0 but got 1"]),
        ([1], [1], 1e-3, []),
        ([1], [1.1], 1e-3, ["Index 0: Expected 1 but got 1.1 (allowed tolerance 0.001)"]),
    ],
)
def test_compare_metric_lists(expected: Any, actual: Any, tol: float, expected_result: List) -> None:
    result = _compare_metrics_list(expected, actual, tol)
    assert result == expected_result


def _write_to_json_dict(tmp_path: Path, data: Any) -> Path:
    """Write an object to a JSON file and return the path to the file."""
    random_file_name = str(uuid.uuid4()) + ".json"
    json_file = tmp_path / random_file_name
    json_file.write_text(json.dumps(data))
    return json_file


@pytest.mark.fast
def test_load_json(tmp_path: Path) -> None:
    """Test loading JSON dictionaries from a file"""
    json_file = tmp_path / "test.json"
    data = {"a": 1.0}
    json_file.write_text(json.dumps(data))
    from_file = _load_json_dict(json_file)
    assert from_file == data

    # Write data that is not a dictionary
    invalid_data = [1, 2, 3]
    json_file.write_text(json.dumps(invalid_data))
    with pytest.raises(ValueError, match="to contain a JSON dictionary, but got list"):
        _load_json_dict(json_file)


@pytest.mark.fast
def test_is_nested_dict() -> None:
    """Test if nested dictionaries are correctly handled."""

    assert _is_nested_dict({"a": {"b": 1}}, message="")
    assert _is_nested_dict({"a": {"b": 1}, "foo": {"bar": 2}}, message="")
    assert not _is_nested_dict({"a": 1, "foo": 2}, message="")
    assert not _is_nested_dict({}, message="")
    with pytest.raises(ValueError, match="Prefix: Metrics file has inconsistent type"):
        _is_nested_dict({"a": 1, "b": {}}, message="Prefix")


def _test_compare_metrics_files(
    tmp_path: Path,
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> str:
    expected_file = _write_to_json_dict(tmp_path, expected)
    actual_file = _write_to_json_dict(tmp_path, actual)
    return compare_metrics_files(expected_file, actual_file)


@pytest.mark.fast
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"a": 1.0},
        {"a": 1.0, "b": 2.0},
        {"1": {"b": 2.0}, "2": {"c": 3.0}},
    ],
)
def test_compare_metrics_files_matches(
    data: Any,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test for metrics files when the expected and actual metrics are the same"""
    with caplog.at_level(logging.WARNING):
        assert _test_compare_metrics_files(expected=data, actual=data, tmp_path=tmp_path) == ""
    assert len(caplog.messages) == 0


def _compare_and_check(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    expected_result: str,
    messages: List[str],
) -> None:
    with caplog.at_level(logging.WARNING):
        assert _test_compare_metrics_files(tmp_path, expected, actual) == expected_result
    assert len(caplog.messages) == len(messages), f"Expected {len(messages)} messages, but got {len(caplog.records)}"
    for index, message in enumerate(messages):
        assert caplog.messages[index] == message, f"Message mismatch at index {index}"


@pytest.mark.fast
def test_compare_metrics_files_metric(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test cases where metrics files don't match"""
    _compare_and_check(
        tmp_path,
        caplog,
        {"a": 1.0},
        {"a": 2.0},
        "Mismatch for 1 out of 1 metrics",
        ["Metric 'a': Expected 1.0 but got 2.0 (allowed tolerance 1e-05)"],
    )


@pytest.mark.fast
def test_compare_metrics_files_dict_nested(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test cases where metrics files don't match"""
    _compare_and_check(
        tmp_path,
        caplog,
        {"0": {"b": 1.0}},
        {"a": 2.0},
        "Expected a nested dictionary as the actual metrics, but got a flat dictionary",
        [],
    )


@pytest.mark.fast
def test_compare_metrics_files_missing_child(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test cases where metrics files don't match"""
    _compare_and_check(
        tmp_path,
        caplog,
        {"0": {"b": 1.0}},
        {"1": {"b": 1.0}},
        "Mismatches for 1 child runs",
        ["Child run '0': Missing from the actual metrics"],
    )


@pytest.mark.fast
def test_compare_metrics_files_child_mismtach(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test cases where metrics files don't match"""
    _compare_and_check(
        tmp_path,
        caplog,
        {"0": {"b": 1.0}},
        {"0": {"b": 2.0}},
        "Mismatches for 1 child runs",
        [
            "Metric 'b': Expected 1.0 but got 2.0 (allowed tolerance 1e-05)",
            "Child run '0': Mismatch for 1 out of 1 metrics",
        ],
    )


@pytest.mark.fast
def test_compare_metrics_files_invalid_data(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test cases where metrics are not in the expected format"""
    expected = ["a", "b"]
    actual = {"0": {"b": 2.0}}
    with caplog.at_level(logging.WARNING):
        result = _test_compare_metrics_files(tmp_path, expected, actual)  # type: ignore
    assert len(caplog.messages) == 0
    assert result.startswith("Error comparing metrics files")
    assert result.endswith("contain a JSON dictionary, but got list")
