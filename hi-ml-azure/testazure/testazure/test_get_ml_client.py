#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for health_azure.azure_get_workspace and related functions.
"""
import os
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import ClientSecretCredential, AzureCliCredential, DeviceCodeCredential

from health_azure.auth import (
    get_credential,
    ENV_SERVICE_PRINCIPAL_ID,
    ENV_SERVICE_PRINCIPAL_PASSWORD,
    ENV_TENANT_ID,
    _get_legitimate_cli_credential,
    _get_legitimate_device_code_credential,
    _get_legitimate_service_principal_credential,
)
from health_azure.utils import (
    ENV_RESOURCE_GROUP,
    ENV_SUBSCRIPTION_ID,
    ENV_WORKSPACE_NAME,
    get_ml_client,
)


@pytest.mark.fast
def test_get_credential() -> None:
    def _mock_validation_error() -> None:
        raise ClientAuthenticationError("")

    # test the case where service principal credentials are set as environment variables
    mock_env_vars = {
        ENV_SERVICE_PRINCIPAL_ID: "foo",
        ENV_TENANT_ID: "bar",
        ENV_SERVICE_PRINCIPAL_PASSWORD: "baz",
    }

    with patch.multiple(
        "health_azure.utils",
        is_running_in_azure_ml=MagicMock(return_value=False),
        is_running_on_azure_agent=MagicMock(return_value=False),
    ):
        with patch.dict(os.environ, mock_env_vars):
            with patch.multiple(
                "health_azure.auth",
                _get_legitimate_service_principal_credential=DEFAULT,
                _get_legitimate_device_code_credential=DEFAULT,
                _get_legitimate_cli_credential=DEFAULT,
                _get_legitimate_interactive_browser_credential=DEFAULT,
            ) as mocks:
                _ = get_credential()
                mocks["_get_legitimate_service_principal_credential"].assert_called_once()
                mocks["_get_legitimate_device_code_credential"].assert_not_called()
                mocks["_get_legitimate_cli_credential"].assert_not_called()
                mocks["_get_legitimate_interactive_browser_credential"].assert_not_called()

        # if the environment variables are not set and we are running on a local machine, a
        # AzureCliCredential should be attempted first
        with patch.object(os.environ, "get", return_value={}):
            with patch.multiple(
                "health_azure.auth",
                _get_legitimate_service_principal_credential=DEFAULT,
                _get_legitimate_device_code_credential=DEFAULT,
                _get_legitimate_cli_credential=DEFAULT,
                _get_legitimate_interactive_browser_credential=DEFAULT,
            ) as mocks:
                mock_get_sp_cred = mocks["_get_legitimate_service_principal_credential"]
                mock_get_device_cred = mocks["_get_legitimate_device_code_credential"]
                mock_get_default_cred = mocks["_get_legitimate_cli_credential"]
                mock_get_browser_cred = mocks["_get_legitimate_interactive_browser_credential"]
                _ = get_credential()
                mock_get_sp_cred.assert_not_called()
                mock_get_device_cred.assert_not_called()
                mock_get_default_cred.assert_called_once()
                mock_get_browser_cred.assert_not_called()

                # if that fails, a DeviceCode credential should be attempted
                mock_get_default_cred.side_effect = _mock_validation_error
                _ = get_credential()
                mock_get_sp_cred.assert_not_called()
                mock_get_device_cred.assert_called_once()
                assert mock_get_default_cred.call_count == 2
                mock_get_browser_cred.assert_not_called()

                # if None of the previous credentials work, an InteractiveBrowser credential should be tried
                mock_get_device_cred.return_value = None
                _ = get_credential()
                mock_get_sp_cred.assert_not_called()
                assert mock_get_device_cred.call_count == 2
                assert mock_get_default_cred.call_count == 3
                mock_get_browser_cred.assert_called_once()

                # finally, if none of the methods work, an Exception should be raised
                mock_get_browser_cred.return_value = None
                with pytest.raises(Exception) as e:
                    get_credential()
                    assert (
                        "Unable to generate and validate a credential. Please see Azure ML documentation"
                        "for instructions on different options to get a credential" in str(e)
                    )


@pytest.mark.fast
def test_get_legitimate_service_principal_credential() -> None:
    # first attempt to create and valiadate a credential with non-existant service principal credentials
    # and check it fails
    mock_service_principal_id = "foo"
    mock_service_principal_password = "bar"
    mock_tenant_id = "baz"
    expected_error_msg = f"Found environment variables for {ENV_SERVICE_PRINCIPAL_ID}, "
    f"{ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID} but was not able to authenticate"
    with pytest.raises(Exception) as e:
        _get_legitimate_service_principal_credential(
            mock_tenant_id, mock_service_principal_id, mock_service_principal_password
        )
        assert expected_error_msg in str(e)

    # now mock the case where validating the credential succeeds and check the value of that
    with patch("health_azure.auth._validate_credential"):
        cred = _get_legitimate_service_principal_credential(
            mock_tenant_id, mock_service_principal_id, mock_service_principal_password
        )
        assert isinstance(cred, ClientSecretCredential)


@pytest.mark.fast
def test_get_legitimate_device_code_credential() -> None:
    def _mock_credential_fast_timeout(timeout: int) -> DeviceCodeCredential:
        return DeviceCodeCredential(timeout=1)

    with patch("health_azure.auth.DeviceCodeCredential", new=_mock_credential_fast_timeout):
        cred = _get_legitimate_device_code_credential()
        assert cred is None

    # now mock the case where validating the credential succeeds
    with patch("health_azure.auth._validate_credential"):
        cred = _get_legitimate_device_code_credential()
        assert isinstance(cred, DeviceCodeCredential)


@pytest.mark.fast
@pytest.mark.skip(reason="Default azure credential are now the default in CI, and test hence fails")
def test_get_legitimate_default_credential_fails() -> None:
    def _mock_credential_fast_timeout(timeout: int) -> AzureCliCredential:
        return AzureCliCredential(timeout=1)

    with patch("health_azure.auth.AzureCliCredential", new=_mock_credential_fast_timeout):
        exception_message = r"AzureCliCredential failed to retrieve a token from the included credentials."
        with pytest.raises(ClientAuthenticationError, match=exception_message):
            _get_legitimate_cli_credential()


@pytest.mark.fast
def test_get_legitimate_default_credential() -> None:
    with patch("health_azure.auth._validate_credential"):
        cred = _get_legitimate_cli_credential()
        assert isinstance(cred, AzureCliCredential)


@pytest.mark.fast
def test_get_ml_client_with_existing_client() -> None:
    """When passing an existing ml_client, it should be returned"""
    ml_client = "mock_ml_client"
    result = get_ml_client(ml_client=ml_client)  # type: ignore
    assert result == ml_client


@pytest.mark.fast
def test_get_ml_client_without_credentials() -> None:
    """When no credentials are available, an exception should be raised"""
    with patch("health_azure.utils.get_credential", return_value=None):
        with pytest.raises(ValueError, match="Can't connect to MLClient without a valid credential"):
            get_ml_client()


@pytest.mark.fast
def test_get_ml_client_from_config_file() -> None:
    """If a workspace config file is found, it should be used to create the MLClient"""
    mock_credentials = "mock_credentials"
    mock_config_path = Path("foo")
    mock_ml_client = MagicMock(workspace_name="workspace")
    mock_from_config = MagicMock(return_value=mock_ml_client)
    mock_resolve_config_path = MagicMock(return_value=mock_config_path)
    with patch.multiple(
        "health_azure.utils",
        get_credential=MagicMock(return_value=mock_credentials),
        resolve_workspace_config_path=mock_resolve_config_path,
        MLClient=MagicMock(from_config=mock_from_config),
    ):
        config_file = Path("foo")
        result = get_ml_client(workspace_config_path=config_file)
        assert result == mock_ml_client
        mock_resolve_config_path.assert_called_once_with(config_file)
        mock_from_config.assert_called_once_with(
            credential=mock_credentials,
            path=str(mock_config_path),
        )


@pytest.mark.fast
def test_get_ml_client_from_environment_variables() -> None:
    """When no workspace config file is found, the MLClient should be created from environment variables"""
    mock_credentials = "mock_credentials"
    the_client = "the_client"
    mock_ml_client = MagicMock(return_value=the_client)
    workspace = "workspace"
    subscription = "subscription"
    resource_group = "resource_group"
    with patch.multiple(
        "health_azure.utils",
        get_credential=MagicMock(return_value=mock_credentials),
        resolve_workspace_config_path=MagicMock(return_value=None),
        MLClient=mock_ml_client,
    ):
        with patch.dict(
            os.environ,
            {ENV_WORKSPACE_NAME: workspace, ENV_SUBSCRIPTION_ID: subscription, ENV_RESOURCE_GROUP: resource_group},
        ):
            result = get_ml_client()
            assert result == the_client
            mock_ml_client.assert_called_once_with(
                subscription_id=subscription,
                resource_group_name=resource_group,
                workspace_name=workspace,
                credential=mock_credentials,
            )


@pytest.mark.fast
def test_get_ml_client_fails() -> None:
    """If neither a workspace config file nor environment variables are found, an exception should be raised"""
    mock_credentials = "mock_credentials"
    the_client = "the_client"
    mock_ml_client = MagicMock(return_value=the_client)
    with patch.multiple(
        "health_azure.utils",
        get_credential=MagicMock(return_value=mock_credentials),
        resolve_workspace_config_path=MagicMock(return_value=None),
        MLClient=mock_ml_client,
    ):
        # In the GitHub runner, the environment variables are set. We need to unset them to test the exception
        with patch.dict(os.environ, {ENV_WORKSPACE_NAME: ""}):
            with pytest.raises(ValueError, match="Tried all ways of identifying the MLClient, but failed"):
                get_ml_client()
