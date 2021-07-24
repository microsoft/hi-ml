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

import pytest

try:
    from health.azure.himl import submit_to_azure_if_needed  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.health.azure.himl import submit_to_azure_if_needed  # type: ignore

from health.azure.himl import AzureRunInformation
from testhiml.health.azure.utils import default_aml_workspace, repository_root

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
def test_submit_to_azure_if_needed_runs_hello_world(local: bool, tmp_path: Path) -> None:
    """
    Test that we can run a simple 'hello world' script.
    """
    workspace = default_aml_workspace()
    assert workspace  # TODO: remove when we do something real with workspace
    print(local)  # TODO: remove when we do something real with local
    shutil.copytree(
        src=repository_root() / "health",
        dst=tmp_path / "health")
    shutil.copy(
        src=repository_root() / "config.json",
        dst=tmp_path / "health" / "azure" / "examples" / "config.json")
