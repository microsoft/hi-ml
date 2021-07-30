#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for the functions in health.azure.azure_util
"""
import os
from unittest import mock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import health.azure.azure_util as util
import pytest

RUN_ID = uuid4().hex
RUN_NUMBER = 42
EXPERIMENT_NAME = "fancy-experiment"


def oh_no() -> None:
    """
    Raise a simple exception. To be used as a side_effect for mocks.
    """
    raise ValueError("Throwing an exception")


@patch("health.azure.azure_util.Experiment")
@patch("health.azure.azure_util.Run")
def test_create_run_recovery_id(mock_run: MagicMock, mock_experiment: MagicMock) -> None:
    """
    The recovery id created for a run
    """
    mock_run.id = RUN_ID
    mock_run.experiment = mock_experiment
    mock_experiment.name = EXPERIMENT_NAME
    recovery_id = util.create_run_recovery_id(mock_run)
    assert recovery_id == EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID


@pytest.mark.parametrize("on_azure", [True, False])
def test_is_running_on_azure_agent(on_azure: bool) -> None:
    with mock.patch.dict(os.environ, {"AGENT_OS": "LINUX" if on_azure else ""}):
        assert on_azure == util.is_running_on_azure_agent()


@patch("health.azure.azure_util.Workspace")
@patch("health.azure.azure_util.Experiment")
@patch("health.azure.azure_util.Run")
def test_fetch_run(mock_run: MagicMock, mock_experiment: MagicMock, mock_workspace: MagicMock) -> None:
    mock_run.id = RUN_ID
    mock_run.experiment = mock_experiment
    mock_experiment.name = EXPERIMENT_NAME
    recovery_id = EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID
    mock_run.number = RUN_NUMBER
    with mock.patch("health.azure.azure_util.get_run", return_value=mock_run):
        run_to_recover = util.fetch_run(mock_workspace, recovery_id)
        # TODO: should we assert that the correct string is logged?
        assert run_to_recover.number == RUN_NUMBER
    mock_experiment.side_effect = oh_no
    with pytest.raises(Exception) as e:
        util.fetch_run(mock_workspace, recovery_id)
    assert str(e.value).startswith(f"Unable to retrieve run {RUN_ID}")


@patch("health.azure.azure_util.Run")
@patch("health.azure.azure_util.Experiment")
@patch("health.azure.azure_util.get_run")
def test_fetch_run_for_experiment(get_run: MagicMock, mock_experiment: MagicMock, mock_run: MagicMock) -> None:
    get_run.side_effect = oh_no
    mock_run.id = RUN_ID
    mock_experiment.get_runs = lambda: [mock_run, mock_run, mock_run]
    mock_experiment.name = EXPERIMENT_NAME
    with pytest.raises(Exception) as e:
        util.fetch_run_for_experiment(mock_experiment, RUN_ID)
    exp = f"Run {RUN_ID} not found for experiment: {EXPERIMENT_NAME}. Available runs are: {RUN_ID}, {RUN_ID}, {RUN_ID}"
    assert str(e.value) == exp


@patch("health.azure.azure_util.InteractiveLoginAuthentication")
def test_get_authentication(mock_interactive_authentication: MagicMock) -> None:
    util.get_authentication()
    assert mock_interactive_authentication.called
    service_principal_id = "1"
    tenant_id = "2"
    service_principal_password = "3"
    with mock.patch.dict(
            os.environ,
            {
                util.SERVICE_PRINCIPAL_ID: service_principal_id,
                util.TENANT_ID: tenant_id,
                util.SERVICE_PRINCIPAL_PASSWORD: service_principal_password
            }):
        spa = util.get_authentication()
        assert spa._service_principal_id == service_principal_id
        assert spa._tenant_id == tenant_id
        assert spa._service_principal_password == service_principal_password


def test_get_secret_from_environment() -> None:
    env_variable_name = uuid4().hex.upper()
    env_variable_value = "42"
    with pytest.raises(ValueError) as e:
        util.get_secret_from_environment(env_variable_name)
    assert str(e.value) == f"There is no value stored for the secret named '{env_variable_name}'"
    assert util.get_secret_from_environment(env_variable_name, allow_missing=True) is None
    with mock.patch.dict(os.environ, {env_variable_name: env_variable_value}):
        assert util.get_secret_from_environment(env_variable_name) == env_variable_value


def test_to_azure_friendly_string() -> None:
    """
    Tests the to_azure_friendly_string function which should replace everything apart from a-zA-Z0-9_ with _, and
    replace multiple _ with a single _
    """
    bad_string = "full__0f_r*bb%sh"
    good_version = util.to_azure_friendly_string(bad_string)
    assert good_version == "full_0f_r_bb_sh"
    good_string = "Not_Full_0f_Rubbish"
    good_version = util.to_azure_friendly_string(good_string)
    assert good_version == good_string
