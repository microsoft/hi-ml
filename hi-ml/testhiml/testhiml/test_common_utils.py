#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import pytest

from health_ml.utils import common_utils
from health_ml.utils.common_utils import get_docker_memory_gb, get_memory_gb, is_linux


@pytest.mark.parametrize("os_name, expected_val", [("nt", True), ("None", False), ("posix", False), ("", False)])
def test_is_windows(os_name: str, expected_val: bool) -> None:
    with patch.object(os, "name", new=os_name):
        assert common_utils.is_windows() == expected_val


@pytest.mark.parametrize("os_name, expected_val", [("nt", False), ("None", False), ("posix", True), ("", False)])
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


@pytest.mark.skipif(not is_linux(), reason="Test only runs on Linux")
def test_available_memory(capsys: pytest.CaptureFixture) -> None:
    """
    Test that get_memory_gb returns a value greater than 0
    """
    # All tests run on Linux, so the result should be available

    result = get_memory_gb(verbose=False)
    assert result is not None
    assert len(result) == 4
    for val in result:
        assert isinstance(val, float)
        assert val > 0.0
    stdout: str = capsys.readouterr().out
    assert len(stdout) == 0


@pytest.mark.skipif(not is_linux(), reason="Test only runs on Linux")
def test_available_memory_prints(capsys: pytest.CaptureFixture) -> None:
    """
    Test that get_memory_gb prints the result of running 'free'
    """
    # All tests run on Linux, so the result should be available

    get_memory_gb(verbose=True)
    stdout: str = capsys.readouterr().out
    assert len(stdout.splitlines()) == 5


def test_available_memory_reads_correctly(capsys: pytest.CaptureFixture) -> None:
    """
    Test that get_memory_gb picks the right fields of the output of 'free'
    """
    free_output = """              total        used        free      shared  buff/cache   available
Mem:           9950        3316        5133           2        1500        6332
Swap:          3072          15        3056
Total:        13022        3331        8190
"""
    with patch("os.popen", return_value=StringIO(free_output)):
        result = get_memory_gb(verbose=True)
        assert result is not None
        assert len(result) == 4
        assert result == (9.717, 5.013, 12.717, 7.998)
        stdout: str = capsys.readouterr().out
        assert free_output in stdout


def test_docker_memory(capsys: pytest.CaptureFixture) -> None:
    """Test that docker_memory returns the expected values"""
    with patch.multiple(
        "health_ml.utils.common_utils",
        is_linux=MagicMock(return_value=True),
        _is_running_in_docker=MagicMock(return_value=True),
    ):
        expected_gb = 1.234
        with patch("pathlib.Path.read_text", return_value=str(int(expected_gb * 1024**3))):
            # Test that the function returns the expected value, and that it prints nothing when verbose=False
            result = get_docker_memory_gb(verbose=False)
            assert result == expected_gb
            stdout: str = capsys.readouterr().out
            assert len(stdout.splitlines()) == 0
            # Test that the function prints the expected value when verbose=True
            result = get_docker_memory_gb(verbose=True)
            assert result == expected_gb
            stdout = capsys.readouterr().out
            assert stdout.splitlines() == [f"Total Docker memory: {expected_gb} GB"]

        with patch("pathlib.Path.read_text", side_effect=ValueError):
            assert get_docker_memory_gb() is None


def test_docker_memory_outside_docker(capsys: pytest.CaptureFixture) -> None:
    """Test that docker_memory returns None if not running in Docker"""
    with patch.multiple(
        "health_ml.utils.common_utils",
        is_linux=MagicMock(return_value=True),
        _is_running_in_docker=MagicMock(return_value=False),
    ):
        assert get_docker_memory_gb() is None
        stdout: str = capsys.readouterr().out
        assert len(stdout.splitlines()) == 0
        # Test that the function prints the expected value when verbose=True
        assert get_docker_memory_gb(verbose=True) is None
        stdout = capsys.readouterr().out
        assert stdout.startswith("Unable to determine Docker memory")
