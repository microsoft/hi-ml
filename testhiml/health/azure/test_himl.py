#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml.
"""
import logging
import shutil
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

import health.azure.examples.elevate_this as elevate_this
from health.azure.himl import AzureRunInformation
from testhiml.health.azure.utils import repository_root

INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"

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
                workspace_config=None,
                workspace_config_path=None,
                entry_script=Path(__file__),
                compute_cluster_name="foo",
                conda_environment_file=Path("env.yaml"))
        assert "Cannot submit to AzureML without the snapshot_root_directory" in str(ex)
    with mock.patch("sys.argv", ["", "--azureml"]):
        with pytest.raises(Exception) as ex:
            submit_to_azure_if_needed(
                workspace_config=None,
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
    # workspace = default_aml_workspace()
    snapshot_root = tmp_path / uuid4().hex
    snapshot_root.mkdir(exist_ok=False)
    repo_root = repository_root()
    shutil.copy(src=repo_root / "src", dst=snapshot_root)
    example_root = snapshot_root / "health" / "azure" / "examples"
    shutil.copy(src=repo_root / "config.json", dst=example_root / "config.json")
    conda_env_file = example_root / "environment.yml"
    # entry_script = example_root / "elevate_this.py"
    sys_args = [
        "",
        f"--message={message_guid}",
        f"--workspace_config_path={example_root / 'config.json'}",
        f"--compute_cluster_name={INEXPENSIVE_TESTING_CLUSTER_NAME}",
        f"--conda_env={conda_env_file}"
    ]
    if not local:
        sys_args.append("--azureml")
    with mock.patch("sys.argv", sys_args):
        run_info = elevate_this.main()
        # with mock.patch(
        #         "health.azure.himl_configs.WorkspaceConfig.get_workspace",
    assert run_info
    captured = capsys.readouterr().out
    if local:
        assert "Successfully queued new run" not in captured
        assert message_guid not in captured
    else:
        assert "Successfully queued new run" in captured
        log_root = snapshot_root / "logs"
        log_root.mkdir(exist_ok=False)
        logs = run_info.run.get_all_logs(destination=log_root)
        print(logs)
        driver_log = log_root / "70_driver_log.txt"
        log_text = driver_log.read_text()
        assert message_guid in log_text
