#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Test utility functions for tests in the package.
"""
import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Optional

from azureml.core import Workspace

from health_azure.utils import (ENV_RESOURCE_GROUP, ENV_SUBSCRIPTION_ID, ENV_WORKSPACE_NAME, get_authentication,
                                get_secret_from_environment, WORKSPACE_CONFIG_JSON)

DEFAULT_DATASTORE = "himldatasets"
FALLBACK_SINGLE_RUN = "refs_pull_545_merge:refs_pull_545_merge_1626538212_d2b07afd"


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


def default_aml_workspace() -> Workspace:
    """
    Gets the default AzureML workspace that is used for testing.
    """
    config_json = repository_root() / WORKSPACE_CONFIG_JSON
    if config_json.is_file():
        return Workspace.from_config()
    else:
        workspace_name = get_secret_from_environment(ENV_WORKSPACE_NAME, allow_missing=False)
        subscription_id = get_secret_from_environment(ENV_SUBSCRIPTION_ID, allow_missing=False)
        resource_group = get_secret_from_environment(ENV_RESOURCE_GROUP, allow_missing=False)
        auth = get_authentication()
        return Workspace.get(name=workspace_name,
                             auth=auth,
                             subscription_id=subscription_id,
                             resource_group=resource_group)


class WorkspaceWrapper:
    """
    Wrapper around aml_workspace so that it is lazily loaded, once.
    """

    def __init__(self) -> None:
        """
        Init.
        """
        self._workspace: Workspace = None

    @property
    def workspace(self) -> Workspace:
        """
        Lazily load the aml_workspace.
        """
        if self._workspace is None:
            self._workspace = default_aml_workspace()
        return self._workspace


DEFAULT_WORKSPACE = WorkspaceWrapper()


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


@contextmanager
def check_config_json(script_folder: Path) -> Generator:
    """
    Create a workspace config.json file in the folder where we expect the test scripts. This is either copied
    from the repository root folder (this should be the case when executing a test on a dev machine), or create
    it from environment variables (this should trigger in builds on the github agents).
    """
    shared_config_json = repository_root() / WORKSPACE_CONFIG_JSON
    target_config_json = script_folder / WORKSPACE_CONFIG_JSON
    if shared_config_json.exists():
        logging.info(f"Copying {WORKSPACE_CONFIG_JSON} from repository root to folder {script_folder}")
        shutil.copy(shared_config_json, target_config_json)
    else:
        logging.info(f"Creating {str(target_config_json)} from environment variables.")
        with open(str(target_config_json), 'w', encoding="utf-8") as file:
            config = {
                "subscription_id": os.getenv(ENV_SUBSCRIPTION_ID, ""),
                "resource_group": os.getenv(ENV_RESOURCE_GROUP, ""),
                "workspace_name": os.getenv(ENV_WORKSPACE_NAME, "")
            }
            json.dump(config, file)
    try:
        yield
    finally:
        target_config_json.unlink()


def check_github_action_runner() -> bool:
    """
    Check if hosted in a github action runner.

    See: https://docs.github.com/en/actions/learn-github-actions/environment-variables
    :return: True if
    """
    logging.info(f"check github_ref: {os.getenv('GITHUB_REF')}")
    return os.getenv('GITHUB_REF', "false") == "true"
