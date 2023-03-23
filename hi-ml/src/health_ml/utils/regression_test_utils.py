#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import json
import logging
import os
import shutil
import tempfile
from math import isclose
from pathlib import Path
from typing import Any, Dict, List, Optional

from azureml.core import Run
import pandas as pd

from health_azure.utils import PARENT_RUN_CONTEXT, RUN_CONTEXT, is_running_in_azure_ml


REGRESSION_TEST_OUTPUT_FOLDER = "OUTPUT"
REGRESSION_TEST_AZUREML_FOLDER = "AZUREML_OUTPUT"
REGRESSION_TEST_AZUREML_PARENT_FOLDER = "AZUREML_PARENT_OUTPUT"
CONTENTS_MISMATCH = "Contents mismatch"
FILE_FORMAT_ERROR = "File format error"
MISSING_FILE = "Missing"
CSV_SUFFIX = ".csv"
TEXT_FILE_SUFFIXES = [".txt", ".json", ".html", ".md"]
REGRESSION_TEST_METRICS_FILENAME = "regression_metrics.json"


def compare_dictionaries(expected: Dict[str, Any], actual: Dict[str, Any], tolerance: float = 1e-5) -> None:
    """
    Function to give more clarity on the difference between two dictionaries.

    :param expected: The first dictionary to compare
    :param actual: The second dictionary to compare
    :param tolerance: The tolerance to allow when comparing numeric values, defaults to 1e-5
    """

    def _check_values_match(expected_v: Any, actual_v: Any, tolerance: float = 1e-5) -> None:
        if type(actual_v) in [float, int] and type(expected_v) in [float, int]:
            if not isclose(actual_v, expected_v, rel_tol=tolerance):
                logging.warning(f"Expected: {expected_v} does not match actual {actual_v}")
        else:
            if expected_v != actual_v:
                logging.warning(f"Expected: {expected_v} does not match actual {actual_v}")

    for expected_key, expected_val in expected.items():
        if expected_key not in actual:
            logging.warning(f"Key {expected_key} is expected but not found in actual dictionary: {actual}")
        else:
            actual_val = actual[expected_key]
            expected_type = type(expected_val)
            actual_type = type(actual_val)
            if expected_type is not actual_type:
                logging.warning(f"Actual value has type {actual_type} but we expected {expected_type}")
            if actual_type in [float, int]:
                _check_values_match(expected_val, actual_val, tolerance=tolerance)
            elif actual_type is dict:
                compare_dictionaries(expected_val, actual_val, tolerance=tolerance)
            elif actual_type in [list, set, str]:
                expected_len = len(expected_val)
                actual_len = len(actual_val)
                if expected_len != actual_len:
                    logging.warning(f"Expected value to have length {expected_len} but found {actual_len}")
                for expected_value, actual_value in zip(expected_val, actual_val):
                    _check_values_match(expected_value, actual_value, tolerance=tolerance)


def _compare_metric_values(expected: Any, actual: Any, tolerance: float = 1e-5) -> str:
    """Compares an expected and an actual value coming from a metrics dictionary. If the values are numeric
    and are close enough (within the given tolerance), an empty string is returned. If the actual
    and expected value are not matching, a string description of the discrepancy will be returned.

    :param expected: The expected value
    :param actual: The actual value
    :param tolerance: For numerical values, this is the relative tolerance allowed, defaults to 1e-5
    :return: An empty string if the expected and actual values match, otherwise a string describing the discrepancy.
    """
    if type(expected) in [float, int]:
        if type(actual) in [float, int]:
            if isclose(actual, expected, rel_tol=tolerance):
                return ""
            return f"Expected {expected} but got {actual} (allowed tolerance {tolerance})"
        return f"Expected a numeric value, but got type {type(actual).__name__}"
    return f"Don't know how to handle expected value of type {type(expected).__name__}"


def _compare_metrics_list(expected: List, actual: List, tolerance: float = 1e-5) -> List[str]:
    """Compares two lists of expected and actual values, coming from a metrics dictionary. The two lists are
    considered matching if they have the same length, and matching list elements are within the given relative
    tolerance.

    :param expected: A list of expected values.
    :param actual: A list with actual metric values.
    :param tolerance: For numerical values, this is the relative tolerance allowed, defaults to 1e-5
    :return: A list of strings describing the discrepancies between the two lists. If the lists match, the
        return value is an empty list.
    """
    messages = []
    expected_len = len(expected)
    actual_len = len(actual)
    if expected_len == actual_len:
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            mismatch = _compare_metric_values(expected_value, actual_value, tolerance=tolerance)
            if mismatch:
                messages.append(f"Index {index}: {mismatch}")
    else:
        messages.append(f"Expected list of length {expected_len} but got {actual_len}")
    return messages


def compare_metrics_dictionaries(expected: Dict[str, Any], actual: Dict[str, Any], tolerance: float = 1e-5) -> str:
    """
    Function to compare two dictionaries that are expected to contain metrics (scalars or lists of scalars) or strings.
    Discrepancies are logged via logging.warning. The function returns an empty string if the dictionaries are
    identical, otherwise a string giving a short summary of the discrepancies.

    :param expected: The first dictionary to compare
    :param actual: The second dictionary to compare
    :param tolerance: The tolerance to allow when comparing numeric values, defaults to 1e-5.
    :return: An empty string if the dictionaries match, otherwise a string describing the discrepancies.
    """

    allowed_types = [float, int, str, list]
    discrepancies = 0
    for key, expected_val in expected.items():
        messages = []
        expected_type = type(expected_val)
        if expected_type not in allowed_types:
            raise ValueError(f"Expected value has type {expected_type.__name__} which is not handled.")
        if key not in actual:
            messages.append("No data found in actual metrics.")
        else:
            actual_val = actual[key]
            actual_type = type(actual_val)
            if actual_type not in allowed_types:
                messages.append(f"Actual value has type {actual_type.__name__} which is not handled.")
            elif expected_type is not actual_type:
                messages.append(
                    f"Actual value has type {actual_type.__name__} but we expected " f"{expected_type.__name__}."
                )
            elif expected_type is list:
                messages.extend(_compare_metrics_list(expected_val, actual_val, tolerance=tolerance))
            else:
                mismatch = _compare_metric_values(expected_val, actual_val, tolerance=tolerance)
                if mismatch:
                    messages.append(mismatch)
        for message in messages[:5]:
            logging.warning(f"Metric '{key}': {message}")
        if len(messages) > 0:
            discrepancies += 1
    return f"Mismatch for {discrepancies} out of {len(expected)} metrics" if discrepancies > 0 else ""


def _load_json_dict(path: Path) -> Dict[str, Any]:
    """Loads the contents of a JSON file. If the contents is a dictionary, this dictionary is returned.

    :param path: The file to load.
    :raises ValueError: If the file does not contain a JSON dictionary.
    :return: The contents of the JSON file.
    """
    metrics_json = json.loads(path.read_text())
    if not isinstance(metrics_json, Dict):
        raise ValueError(
            f"Expected metrics file {path} to contain a JSON dictionary, but got {type(metrics_json).__name__}"
        )
    return metrics_json


def _is_nested_dict(d: Dict[str, Any], message: str) -> bool:
    """Checks if a dictionary contains only dictionaries as values. Returns `True` if the dictionary contains
    only dictionaries as values. Returns `False` if none of the dictionary values are dictionaries, or if the
    dictionary is empty.
    Raises an exception if the dictionary contains a mix of dictionaries and non-dictionaries.

    :param d: The dictionary to check.
    :param message: A message to include in the exception if the dictionary contains a mix of dictionaries.
    :raises ValueError: If the dictionary contains a mix of dictionaries and non-dictionaries.
    :return: True if the dictionary contains values, and all those are dictionaries in turn. False otherwise.
    """
    has_dict_value = [isinstance(v, Dict) for v in d.values()]
    if len(d) == 0:
        return False
    if all(has_dict_value):
        return True
    if any(has_dict_value):
        raise ValueError(
            f"{message}: Metrics file has inconsistent type. Either all or none of the dictionary"
            "values should be dictionaries"
        )
    return False


def compare_metrics_files(expected: Path, actual: Path, tolerance: float = 1e-5) -> str:
    """Reads two files that contain a JSON representation of expected and actual metrics. The two files are read
    and the metrics are compared using compare_metrics_dictionaries. The function returns an empty string if the
    metrics match, otherwise a string summarizing the discrepancies. Details about the discrepancies are logged
    to `logging.warning`.

    :param expected: A file that contains the expected metrics, as a JSON dictionary.
    :param actual: A file that contains the actual metrics, as a JSON dictionary.
    :param tolerance: The maximum allowed tolerance for comparing metrics, defaults to 1e-5
    :return: An empty string if the metrics match, otherwise a string summarizing the discrepancies.
    """
    try:
        expected_metrics = _load_json_dict(expected)
        actual_metrics = _load_json_dict(actual)
        if _is_nested_dict(expected_metrics, "Expected metrics"):
            # For runs with child runs, loop over the child run dictionaries and compare them.
            if not _is_nested_dict(actual_metrics, "Actual metrics"):
                return "Expected a nested dictionary as the actual metrics, but got a flat dictionary"
            messages = []
            for key, expected_child in expected_metrics.items():
                prefix = f"Child run '{key}'"
                if key in actual_metrics:
                    actual_child = actual_metrics[key]
                    message = compare_metrics_dictionaries(expected_child, actual_child, tolerance=tolerance)
                else:
                    message = "Missing from the actual metrics"
                if message:
                    full_message = f"{prefix}: {message}"
                    logging.warning(full_message)
                    messages.append(message)
            if len(messages) > 0:
                return f"Mismatches for {len(messages)} child runs"
            return ""
        else:
            # For simple runs without child runs: Compare the dictionaries.
            if _is_nested_dict(actual_metrics, "Actual metrics"):
                return "Expected a flat dictionary as the actual metrics, but got a nested dictionary"
            return compare_metrics_dictionaries(expected_metrics, actual_metrics, tolerance=tolerance)
    except Exception as e:
        # This handles cases where the files cannot be found, are not JSON, or have inconsistent information.
        return f"Error comparing metrics files: {e}"


def compare_files(expected: Path, actual: Path, csv_relative_tolerance: float = 0.0) -> str:
    """
    Compares two individual files for regression testing. It returns an empty string if the two files appear identical.
    If the files are not identical, a brief error message is returned. Details about the mismatch are logged via
    logging.warning. This method handles CSV files (which are treated and compared as dataframes) and text files (TXT,
    JSON, HTML, MD, which are all compared while ignoring linebreaks. All other extensions are treated as binary,
    and compared on a byte-by-byte basis.

    :param expected: A file that contains the expected contents. The type of comparison (text or binary) is chosen
    based on the extension of this file.
    :param actual: A file to be checked against the expected file.
    :param csv_relative_tolerance: When comparing CSV files, maximum allowed relative discrepancy.
    If 0.0, do not allow any discrepancy.
    :return: An empty string if the files appear identical, or otherwise a brief error message. If there is a
    mismatch, details about the mismatch are printed via logging.warning.
    """

    def print_lines(prefix: str, lines: List[str]) -> None:
        num_lines = len(lines)
        count = min(5, num_lines)
        logging.warning(f"{prefix} {num_lines} lines, first {count} of those:")
        logging.warning(os.linesep.join(lines[:count]))

    def try_read_csv(prefix: str, file: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(file)
        except Exception as ex:
            logging.warning(f"{prefix} file can't be read as CSV: {str(ex)}")
            return None

    def _load_json_from_text_lines(lines: List[str]) -> Dict[str, Any]:
        return json.loads('\n'.join(lines))

    if expected.suffix == CSV_SUFFIX:
        expected_df = try_read_csv("Expected", expected)
        actual_df = try_read_csv("Actual", actual)
        if expected_df is None or actual_df is None:
            return FILE_FORMAT_ERROR
        try:
            pd.testing.assert_frame_equal(actual_df, expected_df, rtol=csv_relative_tolerance)
        except Exception as ex:
            logging.warning(str(ex))
            return CONTENTS_MISMATCH
    elif expected.suffix in TEXT_FILE_SUFFIXES:
        # Compare line-by-line to avoid issues with line separators
        expected_lines = expected.read_text().splitlines()
        actual_lines = actual.read_text().splitlines()
        if expected_lines != actual_lines:
            print_lines("Expected", expected_lines)
            print_lines("Actual", actual_lines)
            # Add additional context for json file mismatches
            if expected.suffix == ".json":
                compare_dictionaries(
                    _load_json_from_text_lines(expected_lines),
                    _load_json_from_text_lines(actual_lines),
                    tolerance=csv_relative_tolerance,
                )
            return CONTENTS_MISMATCH
    else:
        expected_binary = expected.read_bytes()
        actual_binary = actual.read_bytes()
        if expected_binary != actual_binary:
            logging.warning(f"Expected {len(expected_binary)} bytes, actual {len(actual_binary)} bytes")
            return CONTENTS_MISMATCH
    return ""


def compare_folder_contents(
    expected_folder: Path,
    actual_folder: Optional[Path] = None,
    run: Optional[Run] = None,
    csv_relative_tolerance: float = 0.0,
) -> List[str]:
    """
    Compares a set of files in a folder, against files in either the other folder or files stored in the given
    AzureML run. Each file that is present in the ``expected`` folder must be also present in the ``actual`` folder
    (or the AzureML run), with exactly the same contents, in the same folder structure.
    For example, if there is a file ``<expected>/foo/bar/contents.txt``, then there must also be a file
    ``<actual>/foo/bar/contents.txt``. If ``actual_folder`` is provided, then this is used to compare files against the
    set file files in ``expected_folder``, irrespective of the value in ``run``. If ``run`` is provided, the files
    uploaded to the AzureML run are compared against files in ``expected_folder``. If neither ``run`` nor
    ``actual_folder`` are provided, a :exc:`ValueError` is raised.

    :param expected_folder: A folder with files that are expected to be present.
    :param actual_folder: The output folder with the actually produced files.
    :param run: An AzureML run
    :param csv_relative_tolerance: When comparing CSV files, use this as the maximum allowed relative discrepancy.
    If 0.0, do not allow any discrepancy.
    :return: A list of human readable error messages, with message and file path. If no errors are found, the list is
    empty.
    """
    messages = []
    if run and not is_running_in_azure_ml(run):
        logging.warning("Skipping file comparison because the given run context is an AzureML offline run")
        return []
    files_in_run: List[str] = run.get_file_names() if run else []
    temp_folder = Path(tempfile.mkdtemp()) if run else None
    for file in expected_folder.rglob("*"):
        # rglob also returns folders, skip those
        if file.is_dir():
            continue
        # All files stored in AzureML runs use Linux-style path
        file_relative = file.relative_to(expected_folder).as_posix()
        if actual_folder:
            actual_file = actual_folder / file_relative
        elif temp_folder is not None and run is not None:
            actual_file = temp_folder / file_relative
            if file_relative in files_in_run:
                run.download_file(name=str(file_relative), output_file_path=str(actual_file))
        else:
            raise ValueError("Either of the two arguments 'run' or 'actual_folder' must be provided")
        if file_relative == REGRESSION_TEST_METRICS_FILENAME:
            message = compare_metrics_files(expected=file, actual=actual_file, tolerance=csv_relative_tolerance)
        elif actual_file.exists():
            message = compare_files(expected=file, actual=actual_file, csv_relative_tolerance=csv_relative_tolerance)
        else:
            message = MISSING_FILE
        if message:
            messages.append(f"{message}: {file_relative}")
            logging.warning(f"File {file_relative}: {message}")
        else:
            logging.info(f"File {file_relative}: OK")
    if temp_folder:
        shutil.rmtree(temp_folder)
    return messages


def compare_folders_and_run_outputs(expected: Path, actual: Path, csv_relative_tolerance: float = 0.0) -> None:
    """
    Compares the actual set of run outputs in the ``actual`` folder against an expected set of files in the ``expected``
    folder. The ``expected`` folder can have two special subfolders AZUREML_OUTPUT and AZUREML_PARENT_OUTPUT, that
    contain files that are expected to be present in the AzureML run context of the present run (AZUREML_OUTPUT)
    or the run context of the parent run (AZUREML_PARENT_OUTPUT).
    If a file is missing, or does not have the expected contents, an exception is raised.

    :param expected: A folder with files that are expected to be present.
    :param actual: The output folder with the actually produced files.
    :param csv_relative_tolerance: When comparing CSV files, use this as the maximum allowed relative discrepancy.
    If 0.0, do not allow any discrepancy.
    """
    if not expected.is_dir():
        raise ValueError(f"Folder with expected files does not exist: {expected}")
    logging.debug(f"Current working directory: {Path.cwd()}")
    messages = []
    folders_to_check = [
        (REGRESSION_TEST_OUTPUT_FOLDER, "run output files", actual, None),
        (REGRESSION_TEST_AZUREML_FOLDER, "AzureML outputs in present run", None, RUN_CONTEXT),
        (REGRESSION_TEST_AZUREML_PARENT_FOLDER, "AzureML outputs in parent run", None, PARENT_RUN_CONTEXT),
    ]
    for subfolder, message_prefix, actual_folder, run_to_compare in folders_to_check:
        folder = expected / subfolder
        if folder.is_dir():
            logging.info(f"Comparing results in {folder} against {message_prefix}:")
            if actual_folder is None and run_to_compare is None:
                logging.info("No AzureML run to compare against. Skipping")
                continue
            new_messages = compare_folder_contents(
                folder, actual_folder=actual_folder, run=run_to_compare, csv_relative_tolerance=csv_relative_tolerance
            )
            if new_messages:
                messages.append(f"Issues in {message_prefix}:")
                messages.extend(new_messages)
        else:
            logging.info(f"Folder {subfolder} not found, skipping comparison against {message_prefix}")
    if messages:
        raise ValueError(
            f"Some expected files were missing or did not have the expected contents:{os.linesep}"
            f"{os.linesep.join(messages)}"
        )
