#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Union


def assert_file_exists(file_path: Path) -> None:
    """
    Checks if the given file exists.
    """
    assert file_path.exists(), f"File does not exist: {file_path}"


def assert_file_contains_string(full_file: Union[str, Path], expected: Any = None) -> None:
    """
    Checks if the given file contains an expected string
    :param full_file: The path to the file.
    :param expected: The expected contents of the file, as a string.
    """
    logging.info("Checking file {}".format(full_file))
    file_path = full_file if isinstance(full_file, Path) else Path(full_file)
    assert_file_exists(file_path)
    if expected is not None:
        assert expected.strip() in file_path.read_text()
