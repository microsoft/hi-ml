#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml.
"""
import logging
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
