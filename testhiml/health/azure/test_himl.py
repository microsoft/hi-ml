#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml.
"""

import json
import logging
import os
import pathlib
import pytest
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Dict, Generator, List, Tuple
from unittest import mock
from uuid import uuid4

from _pytest.capture import CaptureFixture

from health.azure.himl import submit_to_azure_if_needed  # type: ignore
import health.azure.examples.elevate_this as elevate_this
from health.azure.himl import AzureRunInformation
from testhiml.health.azure.utils import repository_root

INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"

logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)

here = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def check_hi_ml_import() -> Generator:
    """
    Check if hi-ml has already been imported as a package. If so, do nothing, otherwise,
    add "src" to the PYTHONPATH.
    """
    try:
        # pragma pylint: disable=import-outside-toplevel, unused-import
        from health.azure.himl import submit_to_azure_if_needed  # noqa
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
        assert message_guid in captured


# pylint: disable=redefined-outer-name
def test_submit_to_azure_if_needed(check_hi_ml_import: Generator) -> None:
    """
    Test that submit_to_azure_if_needed can be called, and returns immediately.
    """
    # pragma pylint: disable=import-outside-toplevel, import-error
    from health.azure.himl import submit_to_azure_if_needed
    # pragma pylint: enable=import-outside-toplevel, import-error

    with pytest.raises(Exception) as ex:
        submit_to_azure_if_needed(
            workspace_config=None,
            workspace_config_path=None)
    assert "We could not find config.json in:" in str(ex)


def test_invoking_hello_world() -> None:
    """
    Test that invoking hello_world.py does not elevate itself to AzureML without any config.
    """
    score_args = [
        "testhiml/health/azure/test_data/simple/hello_world.py",
        "--message=hello_world"
    ]
    env = dict(os.environ.items())
    env['PYTHONPATH'] = "src"

    code, stdout = spawn_and_monitor_subprocess(
        process=sys.executable,
        args=score_args,
        env=env)
    assert code == 1
    assert "We could not find config.json in:" in "\n".join(stdout)


def test_invoking_hello_world_config1() -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json.
    """
    config = {
        "subscription_id": os.getenv("TEST_SUBSCRIPTION_ID", ""),
        "resource_group": os.getenv("TEST_RESOURCE_GROUP", ""),
        "workspace_name": os.getenv("TEST_WORKSPACE_NAME", "")
    }
    config_path = here / "config.json"
    with open(str(config_path), 'a', encoding="utf-8") as file:
        json.dump(config, file)

    score_args = [
        "testhiml/health/azure/test_data/simple/hello_world_config1.py",
        "--message=hello_world"
    ]
    env = dict(os.environ.items())
    env['PYTHONPATH'] = "src"

    code, stdout = spawn_and_monitor_subprocess(
        process=sys.executable,
        args=score_args,
        env=env)
    assert code == 1
    assert "We could not find config.json in:" in "\n".join(stdout)

    config_path.unlink()


def test_invoking_hello_world_config2() -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with WorkspaceConfig.
    """
    score_args = [
        "testhiml/health/azure/test_data/simple/hello_world_config2.py",
        "--message=hello_world"
    ]
    env = dict(os.environ.items())
    env['PYTHONPATH'] = "src"

    code, stdout = spawn_and_monitor_subprocess(
        process=sys.executable,
        args=score_args,
        env=env)
    assert code == 1
    assert "We could not find config.json in:" in "\n".join(stdout)
