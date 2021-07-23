"""
Global PyTest configuration -- used to define global fixtures for the entire test suite

DO NOT RENAME THIS FILE: (https://docs.pytest.org/en/latest/fixture.html#sharing-a-fixture-across-tests-in-a-module
-or-class-session)
"""
from pathlib import Path
from typing import Generator

import pytest
from azureml.core import Workspace

from health.azure.aml import WORKSPACE_CONFIG_JSON


@pytest.fixture
def aml_workspace() -> Generator:
    """
    Fixture to get the default AzureML workspace that is used for testing.
    """
    root_folder = Path(__file__).parent
    config_json = root_folder / WORKSPACE_CONFIG_JSON
    if config_json.is_file():
        workspace = Workspace.from_config()
    yield workspace
