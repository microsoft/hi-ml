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
                raise ValueError(f"Expected: {expected_v} does not match actual {actual_v}")
            else:
                return
        else:
            if expected_v != actual_v:
                raise ValueError(f"Expected: {expected_v} does not match actual {actual_v}")
        return

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
                    tolerance=csv_relative_tolerance
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
        message = compare_files(expected=file, actual=actual_file,
                                csv_relative_tolerance=csv_relative_tolerance) if actual_file.exists() else MISSING_FILE
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
        (REGRESSION_TEST_AZUREML_PARENT_FOLDER, "AzureML outputs in parent run", None, PARENT_RUN_CONTEXT)
    ]
    for (subfolder, message_prefix, actual_folder, run_to_compare) in folders_to_check:
        folder = expected / subfolder
        if folder.is_dir():
            logging.info(f"Comparing results in {folder} against {message_prefix}:")
            if actual_folder is None and run_to_compare is None:
                logging.info("No AzureML run to compare against. Skipping")
                continue
            new_messages = compare_folder_contents(folder,
                                                   actual_folder=actual_folder,
                                                   run=run_to_compare,
                                                   csv_relative_tolerance=csv_relative_tolerance)
            if new_messages:
                messages.append(f"Issues in {message_prefix}:")
                messages.extend(new_messages)
        else:
            logging.info(f"Folder {subfolder} not found, skipping comparison against {message_prefix}")
    if messages:
        raise ValueError(f"Some expected files were missing or did not have the expected contents:{os.linesep}"
                         f"{os.linesep.join(messages)}")
