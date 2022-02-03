#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import patch
import os
import pytest

from health_ml.utils import common_utils


@pytest.mark.parametrize("os_name, expected_val", [
    ("nt", True),
    ("None", False),
    ("posix", False),
    ("", False)])
def test_is_windows(os_name: str, expected_val: bool) -> None:
    with patch.object(os, "name", new=os_name):
        assert common_utils.is_windows() == expected_val


@pytest.mark.parametrize("os_name, expected_val", [
    ("nt", False),
    ("None", False),
    ("posix", True),
    ("", False)])
def test_is_linux(os_name: str, expected_val: bool) -> None:
    with patch.object(os, "name", new=os_name):
        assert common_utils.is_linux() == expected_val


def test_change_working_directory(tmp_path: Path) -> None:
    """
    Test that change_working_directory temporarily changes the current working directory, but that the context manager
    works to restore the original working directory
    """
    orig_cwd_str = str(Path.cwd())
    tmp_path_str = str(tmp_path)
    assert orig_cwd_str != tmp_path_str
    with common_utils.change_working_directory(tmp_path):
        assert str(Path.cwd()) == tmp_path_str
    # outside of the context, the original working directory should be restored
    assert str(Path.cwd()) == orig_cwd_str != tmp_path_str
