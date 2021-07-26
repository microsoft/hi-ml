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
import subprocess
import sys
from typing import Dict, Generator, List, Tuple
from uuid import uuid4
from pathlib import Path
from unittest import mock
from _pytest.capture import CaptureFixture
import pytest

try:
    from health.azure.himl import submit_to_azure_if_needed  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.health.azure.himl import submit_to_azure_if_needed  # type: ignore

from health.azure.himl import AzureRunInformation
from health.azure.himl_configs import WorkspaceConfig
from testhiml.health.azure.utils import default_aml_workspace

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


# @pytest.mark.oh_so_so_slow
@pytest.mark.parametrize("local", [True, False])
def test_submit_to_azure_if_needed_runs_hello_world(
        local: bool,
        tmp_path: Path,
        capsys: CaptureFixture) -> None:
    """
    Test that we can run a simple script which prints out a given guid
    """
    message_guid = uuid4().hex
    workspace = default_aml_workspace()
    snapshot_root = tmp_path / uuid4().hex
    snapshot_root.mkdir(exist_ok=False)
    # shutil.copy(src=repository_root() / "config.json", dst=snapshot_root / "config.json")
    conda_env_file = snapshot_root / "environment.yml"
    dependencies_txt = """
        name: simple-env
        dependencies:
        - python=3.7.3
        - pip:
            - azureml-sdk==1.23.0
            - conda-merge==0.1.5
    """
    conda_env_file.write_text(dependencies_txt)
    entry_script = snapshot_root / "print_guid.py"
    script_text = f"print('{message_guid}')\n"
    entry_script.write_text(script_text)
    sys_args = [""] if local else ["", "--azureml"]
    with mock.patch("sys.argv", sys_args):
        with mock.patch(
                "health.azure.himl_configs.WorkspaceConfig.get_workspace",
                return_value=workspace):
            run_info = submit_to_azure_if_needed(
                workspace_config=WorkspaceConfig(
                    subscription_id="a",
                    resource_group="b",
                    workspace_name="c"),
                workspace_config_path=None,
                compute_cluster_name="lite-testing-ds2",
                snapshot_root_directory=snapshot_root,
                entry_script=entry_script,
                conda_environment_file=conda_env_file,
                wait_for_completion=True,
                wait_for_completion_show_output=True)
    assert run_info
    captured = capsys.readouterr().out
    if local:
        assert "Successfully queued new run" not in captured
        assert message_guid not in captured
    else:
        assert "Successfully queued new run" in captured
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
        "tests/health/azure/test_data/simple/hello_world_config1.py",
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

    config_path.unlink()


def test_invoking_hello_world_config2() -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with WorkspaceConfig.
    """
    score_args = [
        "tests/health/azure/test_data/simple/hello_world_config2.py",
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
