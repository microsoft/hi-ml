#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import logging
import os
import pytest
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from health.azure.azure_util import RESOURCE_GROUP, SUBSCRIPTION_ID, WORKSPACE_NAME
from health.azure.himl import package_setup_and_hacks
from testhiml.health.azure.util import DEFAULT_WORKSPACE_CONFIG_JSON, repository_root


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
    package_setup_and_hacks()
    # create a default outputs root for all tests
    remove_and_create_folder(outputs_for_tests())
    # run the entire test suite
    yield


@contextmanager
def check_config_json(root: Path) -> Generator:
    """
    Check config.json exists. If so, do nothing, otherwise,
    create one using environment variables.
    """
    config_json = root / DEFAULT_WORKSPACE_CONFIG_JSON
    if config_json.exists():
        yield
    else:
        try:
            logging.info(f"creating {str(config_json)}")

            with open(str(config_json), 'a', encoding="utf-8") as file:
                config = {
                    "subscription_id": os.getenv(SUBSCRIPTION_ID, ""),
                    "resource_group": os.getenv(RESOURCE_GROUP, ""),
                    "workspace_name": os.getenv(WORKSPACE_NAME, "")
                }
                json.dump(config, file)

            yield
        finally:
            if config_json.exists():
                logging.info(f"deleting {str(config_json)}")
                config_json.unlink()


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
