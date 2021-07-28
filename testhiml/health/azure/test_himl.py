#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml.
"""
import logging
import os
import pathlib
import pytest
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock
from uuid import uuid4

from _pytest.capture import CaptureFixture

from conftest import check_config_json
from health.azure.himl import (AzureRunInformation, RUN_RECOVERY_FILE, WORKSPACE_CONFIG_JSON,
                               submit_to_azure_if_needed)
from testhiml.health.azure.util import get_most_recent_run, repository_root


INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"
EXAMPLE_SCRIPT = "elevate_this.py"
ENVIRONMENT_FILE = "environment.yml"


logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)

here = pathlib.Path(__file__).parent.resolve()


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

    logging.info("~~~~~~~~~~~~~~")
    logging.info("\n".join(stdout_lines))
    logging.info("~~~~~~~~~~~~~~")

    return p.wait(), stdout_lines


@pytest.mark.fast
def test_submit_to_azure_if_needed_returns_immediately() -> None:
    """
    Test that submit_to_azure_if_needed can be called, and returns immediately.
    """
    with mock.patch("sys.argv", ["", "--azureml"]):
        with pytest.raises(Exception) as ex:
            submit_to_azure_if_needed(
                aml_workspace=None,
                workspace_config_path=None,
                entry_script=Path(__file__),
                compute_cluster_name="foo",
                conda_environment_file=Path("env.yaml"))
        assert "Cannot submit to AzureML without the snapshot_root_directory" in str(ex)
    with mock.patch("sys.argv", ["", "--azureml"]):
        with pytest.raises(Exception) as ex:
            submit_to_azure_if_needed(
                aml_workspace=None,
                workspace_config_path=None,
                entry_script=Path(__file__),
                compute_cluster_name="foo",
                conda_environment_file=Path("env.yaml"),
                snapshot_root_directory=Path(__file__).parent)
        assert "Cannot glean workspace config from parameters" in str(ex)
    with mock.patch("sys.argv", [""]):
        result = submit_to_azure_if_needed(
            entry_script=Path(__file__),
            compute_cluster_name="foo",
            conda_environment_file=Path("env.yml"),
            )
        assert isinstance(result, AzureRunInformation)
        assert not result.is_running_in_azure


# @pytest.mark.parametrize("local", [True, False])
# def test_submit_to_azure_if_needed_runs_hello_world(
#         local: bool,
#         tmp_path: Path) -> None:
def saved_for_later(
        local: bool,
        tmp_path: Path,
        capsys: CaptureFixture) -> None:
    """
    Test we can run a simple script, which prints out a given guid. We use one of the examples to do this so this unit
    test is also a test of that example.
    """
    message_guid = uuid4().hex
    snapshot_root = tmp_path / uuid4().hex
    repo_root = repository_root()
    shutil.copytree(src=repo_root / "src", dst=snapshot_root)
    example_root = snapshot_root / "health" / "azure" / "examples"
    shutil.copy(src=repo_root / WORKSPACE_CONFIG_JSON, dst=example_root / WORKSPACE_CONFIG_JSON)

    cmd = f"export PYTHONPATH={snapshot_root} "
    cmd = cmd + f"&& python {EXAMPLE_SCRIPT} "
    cmd = cmd + f"--message={message_guid} "
    if not local:
        cmd = cmd + "--azureml "

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        cwd=example_root)
    captured = result.stdout.decode('utf-8')

    if local:
        assert "Successfully queued new run" not in captured
        assert f"The message was: {message_guid}" in captured
        return

    assert "Successfully queued new run" in captured
    run = get_most_recent_run(run_recovery_file=example_root / RUN_RECOVERY_FILE)
    assert run.status in ["Finalizing", "Completed"]
    log_root = snapshot_root / "logs"
    log_root.mkdir(exist_ok=False)
    run.get_all_logs(destination=log_root)
    driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
    log_text = driver_log.read_text()
    assert f"The message was: {message_guid}" in log_text


def test_invoking_hello_world() -> None:
    """
    Test that invoking hello_world.py does not elevate itself to AzureML without any config.
    """
    score_args = [
        str(repository_root() / "testhiml/health/azure/test_data/simple/hello_world.py"),
        "--azureml",
        "--message=hello_world"
    ]
    env = dict(os.environ.items())
    try:
        # pragma pylint: disable=import-outside-toplevel, unused-import
        from health.azure.himl import submit_to_azure_if_needed  # noqa
        # pragma pylint: enable=import-outside-toplevel, unused-import
    except ImportError:
        logging.info("using local src")
        env['PYTHONPATH'] = "src"

    code, stdout = spawn_and_monitor_subprocess(
        process=sys.executable,
        args=score_args,
        env=env)
    assert code == 1
    assert "Cannot glean workspace config from parameters, and so not submitting to AzureML" in "\n".join(stdout)


def test_invoking_hello_world_config1() -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json.
    """
    score_args = [
        str(repository_root() / "testhiml/health/azure/test_data/simple/hello_world_config1.py"),
        "--azureml",
        "--message=hello_world"
    ]
    env = dict(os.environ.items())
    env["COMPUTE_CLUSTER_NAME"] = INEXPENSIVE_TESTING_CLUSTER_NAME
    try:
        # pragma pylint: disable=import-outside-toplevel, unused-import
        from health.azure.himl import submit_to_azure_if_needed  # noqa
        # pragma pylint: enable=import-outside-toplevel, unused-import
    except ImportError:
        logging.info("using local src")
        env['PYTHONPATH'] = "src"

    with check_config_json(Path("testhiml/health/azure/test_data/simple")):
        code, stdout = spawn_and_monitor_subprocess(
            process=sys.executable,
            args=score_args,
            env=env)
        assert code == 0
        assert "ActivityFailedException:" in "\n".join(stdout)
