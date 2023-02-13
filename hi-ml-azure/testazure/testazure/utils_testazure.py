#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Test utility functions for tests in the package.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Optional

from azure.ai.ml import MLClient
from azureml.core import Run
from health_azure.utils import (ENV_EXPERIMENT_NAME, WORKSPACE_CONFIG_JSON, UnitTestWorkspaceWrapper,
                                to_azure_friendly_string)
from health_azure import create_aml_run_object
from health_azure.himl import effective_experiment_name
from health_azure.utils import get_ml_client, get_workspace

DEFAULT_DATASTORE = "himldatasets"

TEST_DATASET_NAME = "test_dataset"
TEST_DATA_ASSET_NAME = "test_dataset"
TEST_INVALID_DATA_ASSET_NAME = "non_existent_dataset"
TEST_DATASTORE_NAME = "test_datastore"

USER_IDENTITY_TEST_DATASTORE = "test_identity_based_datastore"
USER_IDENTITY_TEST_ASSET = "test_identity_based_data_asset"
USER_IDENTITY_TEST_ASSET_OUTPUT = "test_identity_based_data_asset_output"
USER_IDENTITY_TEST_FILE = "test_identity_based_file.txt"
FALLBACK_SINGLE_RUN = "refs_pull_545_merge:refs_pull_545_merge_1626538212_d2b07afd"

# List of root folders to add to .amlignore
DEFAULT_IGNORE_FOLDERS = [".config", ".git", ".github", ".idea", ".mypy_cache", ".pytest_cache", ".vscode",
                          "docs", "node_modules"]

DEFAULT_WORKSPACE = UnitTestWorkspaceWrapper()


class MockRun:
    def __init__(self, run_id: str = 'run1234', tags: Optional[Dict[str, str]] = None) -> None:
        self.id = run_id
        self.tags = tags

    def download_file(self) -> None:
        # for mypy
        pass


def himl_azure_root() -> Path:
    """
    Gets the root folder of the hi-ml-azure code in the repository.
    """
    return Path(__file__).parent.parent.parent


def repository_root() -> Path:
    """
    Gets the root folder of the git repository.
    """
    return himl_azure_root().parent


def experiment_for_unittests() -> str:
    """
    Gets the name of the experiment to use for tests.
    """
    experiment_name = to_azure_friendly_string(os.getenv(ENV_EXPERIMENT_NAME, "unittests"))
    assert experiment_name is not None
    return experiment_name


@contextmanager
def change_working_directory(path_or_str: Path) -> Generator:
    """
    Context manager for changing the current working directory
    """
    new_path = Path(path_or_str).expanduser()
    old_path = Path.cwd()
    os.chdir(new_path)
    yield
    os.chdir(old_path)


def get_shared_config_json() -> Path:
    """
    Gets the path to the config.json file that should exist for running tests locally (outside github build agents).
    """
    return repository_root() / "hi-ml-azure" / "testazure" / WORKSPACE_CONFIG_JSON


def create_unittest_run_object(snapshot_directory: Optional[Path] = None) -> Run:
    return create_aml_run_object(experiment_name=effective_experiment_name("himl-tests"),
                                 workspace=DEFAULT_WORKSPACE.workspace,
                                 snapshot_directory=snapshot_directory)


def get_test_ml_client() -> MLClient:
    """Generates an MLClient object for use in tests.

    :return: MLClient object
    """

    workspace = get_workspace()
    return get_ml_client(aml_workspace=workspace)


def current_test_name() -> str:
    """Get the name of the currently executed test. This is read off an environment variable. If that
    is not found, the function returns an empty string."""
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    if current_test:
        return current_test.split(':')[-1].split(' ')[0]
    return ""
