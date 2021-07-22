#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Tests for hi-ml.
"""

import logging
import os
import subprocess
import sys
from typing import Dict, Generator, List, Tuple

import pytest


logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)


@pytest.fixture
def check_hi_ml_import() -> Generator:
    """
    Check if hi-ml has already been imported as a package. If so, do nothing, otherwise,
    add "src" to the PYTHONPATH.
    """
    try:
        # pragma pylint: disable=import-outside-toplevel, unused-import
        from health.azure.aml import submit_to_azure_if_needed  # noqa
        # pragma pylint: enable=import-outside-toplevel, unused-import
        yield
    except ImportError:
        logging.info("using local src")
        path = sys.path

        # Add src for local version of hi-ml.
        sys.path.append('src')
        yield
        # Restore the path.
        sys.path = path


def spawn_and_monitor_subprocess(process: str, args: List[str], env: Dict[str, str]) -> Tuple[int, List[str]]:
    """
    Helper function to spawn and monitor subprocesses.
    :param process: The name or path of the process to spawn.
    :param args: The args to the process.
    :param env: The environment variables for the process (default is the environment variables of the parent).
    :return: Return code after the process has finished, and the list of lines that were written to stdout by the
    subprocess.
    """
    p = subprocess.Popen(
        [process] + args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )

    # Read and print all the lines that are printed by the subprocess
    stdout_lines = [line.decode('UTF-8').strip() for line in p.stdout]  # type: ignore

    return p.wait(), stdout_lines


# pylint: disable=redefined-outer-name
def test_submit_to_azure_if_needed(check_hi_ml_import: Generator) -> None:
    """
    Test that submit_to_azure_if_needed can be called.
    """
    # pragma pylint: disable=import-outside-toplevel, import-error
    from health.azure.aml import submit_to_azure_if_needed
    # pragma pylint: enable=import-outside-toplevel, import-error

    with pytest.raises(Exception) as ex:
        submit_to_azure_if_needed(
            workspace_config=None,
            workspace_config_path=None)
    assert "We could not find config.json in:" in str(ex)


def test_invoking_hello_world() -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML.
    """
    score_args = [
        "tests/health/azure/test_data/simple/hello_world.py",
        "hello_world"
    ]
    env = dict(os.environ.items())
    env['PYTHONPATH'] = "."

    code, stdout = spawn_and_monitor_subprocess(
        process=sys.executable,
        args=score_args,
        env=env)
    assert code == 1
    assert "We could not find config.json in:" in "\n".join(stdout)
