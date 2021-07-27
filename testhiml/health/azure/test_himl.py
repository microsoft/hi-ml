#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml.
"""
import logging
import shutil
import subprocess
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest
from _pytest.capture import CaptureFixture

try:
    from health.azure.himl import submit_to_azure_if_needed  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.health.azure.himl import submit_to_azure_if_needed  # type: ignore

from health.azure.himl import AzureRunInformation, RUN_RECOVERY_FILE, WORKSPACE_CONFIG_JSON
from testhiml.health.azure.util import get_most_recent_run, repository_root

INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"
EXAMPLE_SCRIPT = "elevate_this.py"
ENVIRONMENT_FILE = "environment.yml"


logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)


@pytest.mark.fast
def test_submit_to_azure_if_needed_returns_immediately() -> None:
    """
    Test that submit_to_azure_if_needed can be called, and returns immediately.
    """
    with mock.patch("sys.argv", ["", "--azureml"]):
        with pytest.raises(Exception) as ex:
            submit_to_azure_if_needed(
                workspace_config_path=None,
                entry_script=Path(__file__),
                compute_cluster_name="foo",
                conda_environment_file=Path("env.yaml"))
        assert "Cannot submit to AzureML without the snapshot_root_directory" in str(ex)
    with mock.patch("sys.argv", ["", "--azureml"]):
        with pytest.raises(Exception) as ex:
            submit_to_azure_if_needed(
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


@pytest.mark.parametrize("local", [True, False])
def test_submit_to_azure_if_needed_runs_hello_world(
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
    # Check run.status
