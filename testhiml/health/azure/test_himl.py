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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock
from uuid import uuid4

from conftest import check_config_json
from health.azure.himl import (AzureRunInformation, RUN_RECOVERY_FILE, WORKSPACE_CONFIG_JSON,
                               submit_to_azure_if_needed)
from testhiml.health.azure.test_data.make_tests import render_environment_yaml, render_test_script
from testhiml.health.azure.util import get_most_recent_run, repository_root


INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"
EXAMPLE_SCRIPT = "elevate_this.py"
ENVIRONMENT_FILE = "environment.yml"


logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)

here = pathlib.Path(__file__).parent.resolve()


class RunTarget(Enum):
    LOCAL = 1
    AZUREML = 2


def spawn_and_monitor_subprocess(process: str, args: List[str],
                                 cwd: Path, env: Dict[str, str]) -> Tuple[int, List[str]]:
    """
    Helper function to spawn and monitor subprocesses.
    :param process: The name or path of the process to spawn.
    :param args: The args to the process.
    :param cwd: Working directory.
    :param env: The environment variables for the process (default is the environment variables of the parent).
    :return: Return code after the process has finished, and the list of lines that were written to stdout by the
    subprocess.
    """
    p = subprocess.Popen(
        [process] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env)

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


@pytest.mark.skip()
@pytest.mark.parametrize("runTarget", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_submit_to_azure_if_needed_runs_hello_world(
        runTarget: RunTarget,
        tmp_path: Path) -> None:
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
    if runTarget == RunTarget.AZUREML:
        cmd = cmd + "--azureml "

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        cwd=example_root)
    captured = result.stdout.decode('utf-8')

    if runTarget == RunTarget.LOCAL:
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


def render_test_scripts(path: Path, runTarget: RunTarget,
                        extra_options: Dict[str, str], extra_args: List[str]) -> Tuple[int, List[str]]:
    """
    Prepare test scripts, submit them, and return response.

    :param path: Where to build the test scripts.
    :param runTarget: Where to run the script.
    :param extra_options: Extra options for template rendering.
    :param extra_args: Extra command line arguments for calling script.
    :return: snapshot_root and response from spawn_and_monitor_subprocess.
    """
    repo_root = repository_root()

    environment_yaml_path = path / "environment.yml"
    latest_version_path = repo_root / "latest_version.txt"
    if latest_version_path.exists():
        latest_version = f"=={latest_version_path.read_text()}"
        logging.debug(f"pinning hi-ml to: {latest_version}")
    else:
        latest_version = ""
        logging.debug("not pinning hi-ml")
    render_environment_yaml(environment_yaml_path, latest_version)

    entry_script_path = path / "test_script.py"
    render_test_script(entry_script_path, extra_options, INEXPENSIVE_TESTING_CLUSTER_NAME, environment_yaml_path)

    score_args = [str(entry_script_path)]
    if runTarget == RunTarget.AZUREML:
        score_args.append("--azureml")
    score_args.extend(extra_args)

    env = dict(os.environ.items())

    with check_config_json(path):
        return spawn_and_monitor_subprocess(
            process=sys.executable,
            args=score_args,
            cwd=path,
            env=env)


@pytest.mark.parametrize("runTarget", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_no_config(runTarget: RunTarget, tmp_path: Path) -> None:
    """
    Test invoking hello_world.py.and
    If running in AzureML - does not elevate itself to AzureML without any config.
    Else runs locally.
    :param runTarget: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    message_guid = uuid4().hex
    extra_options = {
        'workspace_config_path': 'None',
        'environment_variables': 'None',
        'args': 'parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")',
        'body': 'print(f"The message was: {args.message}")'
    }
    extra_args = [f"--message={message_guid}"]
    code, stdout = render_test_scripts(tmp_path, runTarget, extra_options, extra_args)
    captured = "\n".join(stdout)
    if runTarget == RunTarget.LOCAL:
        assert code == 0
        expected_output = f"The message was: {message_guid}"
        assert "Successfully queued new run" not in captured
        assert expected_output in captured
    else:
        assert code == 1
        assert "Cannot glean workspace config from parameters, and so not submitting to AzureML" in captured


@pytest.mark.parametrize("runTarget", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_config(runTarget: RunTarget, tmp_path: Path) -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json.
    :param runTarget: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    message_guid = uuid4().hex
    extra_options = {
        'workspace_config_path': 'here / "config.json"',
        'environment_variables': 'None',
        'args': 'parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")',
        'body': 'print(f"The message was: {args.message}")'
    }
    extra_args = [f"--message={message_guid}"]
    code, stdout = render_test_scripts(tmp_path, runTarget, extra_options, extra_args)
    captured = "\n".join(stdout)
    assert code == 0
    expected_output = f"The message was: {message_guid}"
    if runTarget == RunTarget.LOCAL:
        assert "Successfully queued new run" not in captured
        assert expected_output in captured
    else:
        assert "Successfully queued new run test_script_" in captured

        run = get_most_recent_run(run_recovery_file=tmp_path / RUN_RECOVERY_FILE)
        assert run.status in ["Finalizing", "Completed"]
        log_root = tmp_path / "logs"
        log_root.mkdir(exist_ok=False)
        run.get_all_logs(destination=log_root)
        driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
        log_text = driver_log.read_text()
        assert expected_output in log_text


@pytest.mark.parametrize("runTarget", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_env_var(runTarget: RunTarget, tmp_path: Path) -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json,
    and that environment variables are passed through.
    :param runTarget: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    message_guid = uuid4().hex
    extra_options = {
        'workspace_config_path': 'here / "config.json"',
        'environment_variables': {'message_guid': message_guid},
        'args': '',
        'body': 'print(f"The message_guid env var was: {os.getenv(\'message_guid\')}")'
    }
    extra_args = []
    code, stdout = render_test_scripts(tmp_path, runTarget, extra_options, extra_args)
    captured = "\n".join(stdout)
    assert code == 0
    expected_output = f"The message_guid env var was: {message_guid}"
    if runTarget == RunTarget.LOCAL:
        assert "Successfully queued new run" not in captured
        assert expected_output in captured
    else:
        assert "Successfully queued new run test_script_" in captured

        run = get_most_recent_run(run_recovery_file=tmp_path / RUN_RECOVERY_FILE)
        assert run.status in ["Finalizing", "Completed"]
        log_root = tmp_path / "logs"
        log_root.mkdir(exist_ok=False)
        run.get_all_logs(destination=log_root)
        driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
        log_text = driver_log.read_text()
        assert expected_output in log_text
