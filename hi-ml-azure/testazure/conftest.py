#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest

from health_azure import health_azure_package_setup

# Reduce logging noise in DEBUG mode
health_azure_package_setup()


def outputs_for_tests() -> Path:
    """
    Gets the folder that will hold all temporary results for tests.
    """
    return Path(__file__).parent / "outputs"


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
    # create a default outputs root for all tests
    remove_and_create_folder(outputs_for_tests())
    # run the entire test suite
    yield


@pytest.fixture
def random_folder() -> Generator:
    """
    Fixture to automatically create a random directory before executing a test
    """
    # create dirs before executing the test
    folder = outputs_for_tests() / str(uuid.uuid4().hex)
    remove_and_create_folder(folder)
    print(f"Created temporary folder for test: {folder}")
    yield folder
