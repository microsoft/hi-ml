#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Optional

from health_azure.utils import PathOrString


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(__file__).resolve().parent.parent.parent
    return root / path if path else root


def full_test_data_path(prefix: str = "", suffix: str = "") -> Path:
    """
    Takes a relative path inside the testhiml/test_data folder, and returns its absolute path.

    :param prefix: An optional prefix to the path "test_data" that comes after the root directory
    :param suffix: An optional suffix to the path "test_data"
    :return: The absolute path
    """
    data_path = tests_root_directory()
    if prefix:
        data_path = data_path / prefix

    data_path = data_path / "test_data"
    if suffix:
        data_path = data_path / suffix

    return data_path
