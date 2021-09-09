#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest

from health.azure.azure_util import ENV_RESOURCE_GROUP, ENV_SUBSCRIPTION_ID, ENV_WORKSPACE_NAME
from health.azure.himl import WORKSPACE_CONFIG_JSON, _package_setup
from testhiml.health.azure.util import repository_root


def outputs_for_tests() -> Path:
    """
    Gets the folder that will hold all temporary results for tests.
    """
    return repository_root() / "outputs"


def remove_and_create_folder(folder: Path) -> None:
    """
    Delete the folder if it exists, and remakes it. This method ignores errors that can come from
    an explorer window still being open inside of the test result folder.
    """
    folder = Path(folder)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(exist_ok=True, parents=True)


@pytest.fixture(autouse=True, scope='session')
def test_suite_setup() -> Generator:
    _package_setup()
    # create a default outputs root for all tests
    remove_and_create_folder(outputs_for_tests())
    # run the entire test suite
    yield


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


@pytest.fixture
def random_folder() -> Generator:
    """
    Fixture to automatically create a random directory before executing a test and then
    removing this directory after the test has been executed.
    """
    # create dirs before executing the test
    folder = outputs_for_tests() / str(uuid.uuid4().hex)
    remove_and_create_folder(folder)
    print(f"Created temporary folder for test: {folder}")
    yield folder
