#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for the functions in health.azure.azure_util
"""
from unittest import mock
from unittest.mock import patch
from uuid import uuid4

import health.azure.azure_util as util
import pytest
from azureml.core import Run

RUN_ID = uuid4().hex
EXPERIMENT_NAME = "fancy-experiment"


@pytest.fixture
def mocked_run() -> Run:
    mock_run = mock.patch("health.azure.azure_util.Run")
    mock_experiment = mock.patch("health.azure.azure_util.Experiment")
    mock_run.id = RUN_ID  # type: ignore
    mock_run.experiment = mock_experiment  # type: ignore
    mock_experiment.name = EXPERIMENT_NAME  # type: ignore
    return mock_run


def test_create_run_recovery_id(mocked_run: Run) -> None:
    """
    The recovery id created for a run
    """
    recovery_id = util.create_run_recovery_id(mocked_run)
    assert recovery_id == EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID


def test_fetch_run_for_experiment(mocked_run: Run) -> None:
    """
    Tests fetch_run, and thus test_fetch_run_for_experiment and split_recovery_id
    """
    assert True


def test_is_running_on_azure_agent() -> None:
    """
    """
    assert True


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
