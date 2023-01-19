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
from health_azure.utils import get_ml_client

DEFAULT_DATASTORE = "himldatasets"
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
    return get_ml_client(workspace_config_path=get_shared_config_json())
