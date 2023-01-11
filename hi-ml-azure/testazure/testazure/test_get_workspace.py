#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for health_azure.azure_util.get_workspace and related functions.
"""
import os
from pathlib import Path
from uuid import uuid4

from azureml.core.authentication import ServicePrincipalAuthentication
from _pytest.logging import LogCaptureFixture
import pytest
from unittest.mock import MagicMock, patch

from health_azure.utils import (find_file_in_parent_folders,
                                find_file_in_parent_to_pythonpath,
                                get_authentication,
                                get_secret_from_environment,
                                get_workspace)
from health_azure.utils import (WORKSPACE_CONFIG_JSON,
                                ENV_SERVICE_PRINCIPAL_ID,
                                ENV_SERVICE_PRINCIPAL_PASSWORD,
                                ENV_TENANT_ID,
                                ENV_WORKSPACE_NAME,
                                ENV_SUBSCRIPTION_ID,
                                ENV_RESOURCE_GROUP)
from testazure.utils_testazure import (change_working_directory,
                                       himl_azure_root)


@pytest.mark.fast
def test_get_secret_from_environment() -> None:
    env_variable_name = uuid4().hex.upper()
    env_variable_value = "42"
    with pytest.raises(ValueError) as e:
        get_secret_from_environment(env_variable_name)
    assert str(e.value) == f"There is no value stored for the secret named '{env_variable_name}'"
    assert get_secret_from_environment(env_variable_name, allow_missing=True) is None
    with patch.dict(os.environ, {env_variable_name: env_variable_value}):
        assert get_secret_from_environment(env_variable_name) == env_variable_value


@pytest.mark.fast
def test_find_file_in_parent_to_pythonpath(tmp_path: Path) -> None:
    file_name = "some_file.json"
    file = tmp_path / file_name
    file.touch()
    python_root = tmp_path / "python_root"
    python_root.mkdir(exist_ok=False)
    start_path = python_root / "starting_directory"
    start_path.mkdir(exist_ok=False)
    with change_working_directory(start_path):
        found_file = find_file_in_parent_to_pythonpath(file_name)
        assert found_file
        with patch.dict(os.environ, {"PYTHONPATH": str(python_root)}):
            found_file = find_file_in_parent_to_pythonpath(file_name)
            assert not found_file


@pytest.mark.fast
def test_find_file_in_parent_folders(caplog: LogCaptureFixture) -> None:
    current_file_path = Path(__file__)
    # If no start_at arg is provided, will start looking for file at current working directory.
    # First mock this to be the hi-ml-azure root
    himl_az_root = himl_azure_root()
    himl_azure_test_root = himl_az_root / "testazure" / "testazure"
    with patch("health_azure.utils.Path.cwd", return_value=himl_azure_test_root):
        # current_working_directory = Path.cwd()
        found_file_path = find_file_in_parent_folders(
            file_name=current_file_path.name,
            stop_at_path=[himl_az_root]
        )
        last_caplog_msg = caplog.messages[-1]
        assert found_file_path == current_file_path
        assert f"Searching for file {current_file_path.name} in {himl_azure_test_root}" in last_caplog_msg

        # Now try to search for a nonexistent path in the same folder. This should return None
        nonexistent_path = himl_az_root / "idontexist.py"
        assert not nonexistent_path.is_file()
        assert find_file_in_parent_folders(
            file_name=nonexistent_path.name,
            stop_at_path=[himl_az_root]
        ) is None

        # Try to find the first path (i.e. current file name) when starting in a different folder.
        # This should not work
        assert find_file_in_parent_folders(
            file_name=current_file_path.name,
            stop_at_path=[himl_az_root],
            start_at_path=himl_az_root
        ) is None

    # Try to find the first path (i.e. current file name) when current working directory is not the testazure
    # folder. This should not work
    with patch("health_azure.utils.Path.cwd", return_value=himl_az_root):
        assert not (himl_az_root / current_file_path.name).is_file()
        assert find_file_in_parent_folders(
            file_name=current_file_path.name,
            stop_at_path=[himl_az_root.parent]
        ) is None


@pytest.mark.fast
@patch("health_azure.utils.InteractiveLoginAuthentication")
def test_get_authentication(mock_interactive_authentication: MagicMock) -> None:
    with patch.dict(os.environ, {}, clear=True):
        get_authentication()
        assert mock_interactive_authentication.called
    service_principal_id = "1"
    tenant_id = "2"
    service_principal_password = "3"
    with patch.dict(
            os.environ,
            {
                ENV_SERVICE_PRINCIPAL_ID: service_principal_id,
                ENV_TENANT_ID: tenant_id,
                ENV_SERVICE_PRINCIPAL_PASSWORD: service_principal_password
            },
            clear=True):
        spa = get_authentication()
        assert isinstance(spa, ServicePrincipalAuthentication)
        assert spa._service_principal_id == service_principal_id
        assert spa._tenant_id == tenant_id
        assert spa._service_principal_password == service_principal_password


@pytest.mark.fast
def test_get_workspace_in_azureml() -> None:
    """get_workspace should return the workspace of the current run if running in AzureML"""
    mock_workspace = "foo"
    with patch("health_azure.utils.RUN_CONTEXT") as mock_run_context:
        # A run with an experiment is considered to be running in AzureML.
        mock_run_context.experiment = MagicMock(workspace=mock_workspace)
        workspace = get_workspace(None, None)
        assert workspace == mock_workspace


@pytest.mark.fast
def test_get_workspace_with_given_workspace() -> None:
    """get_workspace should return the given workspace if one is given"""
    mock_run = MagicMock()
    assert get_workspace(aml_workspace=mock_run) == mock_run


@pytest.mark.fast
def test_get_workspace_searches_for_file() -> None:
    """get_workspace should try to load a config.json file if not provided with one"""
    found_file = Path("does_not_exist")
    with patch("health_azure.utils.find_file_in_parent_to_pythonpath", return_value=found_file) as mock_find:
        with pytest.raises(FileNotFoundError, match="Workspace config file does not exist"):
            get_workspace(None, None)
        mock_find.assert_called_once_with(WORKSPACE_CONFIG_JSON)


@pytest.mark.fast
def test_get_workspace_loads_env_variables() -> None:
    """get_workspace should access environment variables for workspace details
    if no config file is given"""
    workspace_name = "name"
    subscription = "sub"
    group = "group"

    auth = "auth"
    mock_auth = MagicMock(return_value=auth)
    workspace = MagicMock(name="workspace")
    mock_workspace_get = MagicMock(return_value=workspace)
    mock_workspace = MagicMock(get=mock_workspace_get)
    with patch.dict(os.environ,
                    {
                        ENV_WORKSPACE_NAME: workspace_name,
                        ENV_SUBSCRIPTION_ID: subscription,
                        ENV_RESOURCE_GROUP: group
                    }):
        with patch.multiple(
            "health_azure.utils",
            find_file_in_parent_to_pythonpath=MagicMock(return_value=None),
            get_authentication=mock_auth,
            Workspace=mock_workspace
        ):
            assert get_workspace(None, None) == workspace
            mock_auth.assert_called_once_with()
            mock_workspace_get.assert_called_once_with(
                name=workspace_name,
                auth=auth,
                subscription_id=subscription,
                resource_group=group)


@pytest.mark.fast
def test_get_workspace_env_variables_missing() -> None:
    """get_workspace should access environment variables for workspace details,
    and fail if one of them is missing. Missing environment variables are mocked via
    get_secret_from_environment returning None"""
    with patch.multiple(
        "health_azure.utils",
        # This ensures that no config file is found
        find_file_in_parent_to_pythonpath=MagicMock(return_value=None),
        # This ensures that no environment variables are found
        get_secret_from_environment=MagicMock(return_value=None),
        get_authentication=MagicMock(return_value="auth")
    ):
        with pytest.raises(ValueError, match="Tried all ways of identifying the workspace, but failed"):
            get_workspace()


@pytest.mark.fast
def test_get_workspace_from_existing_file(tmp_path: Path) -> None:
    """get_workspace should load a workspace from an existing file"""
    config_file = tmp_path / "config.json"
    config_file.touch()

    auth = "auth"
    mock_auth = MagicMock(return_value=auth)
    workspace = MagicMock(name="workspace")
    mock_workspace_from_config = MagicMock(return_value=workspace)
    mock_workspace = MagicMock(from_config=mock_workspace_from_config)
    with patch.multiple(
        "health_azure.utils",
        get_authentication=mock_auth,
        Workspace=mock_workspace
    ):
        assert get_workspace(workspace_config_path=config_file) == workspace
        mock_auth.assert_called_once_with()
        mock_workspace_from_config.assert_called_once_with(
            path=str(config_file),
            auth=auth)
