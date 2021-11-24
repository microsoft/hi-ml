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

from health_azure.utils import (ENV_RESOURCE_GROUP, ENV_SUBSCRIPTION_ID, ENV_WORKSPACE_NAME, WORKSPACE_CONFIG_JSON,
                                WorkspaceWrapper)

DEFAULT_DATASTORE = "himldatasets"
FALLBACK_SINGLE_RUN = "refs_pull_545_merge:refs_pull_545_merge_1626538212_d2b07afd"

# List of root folders to add to .amlignore
DEFAULT_IGNORE_FOLDERS = [".config", ".git", ".github", ".idea", ".mypy_cache", ".pytest_cache", ".vscode",
                          "docs", "node_modules"]


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
