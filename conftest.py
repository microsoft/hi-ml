import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest

from health.azure.himl import package_setup_and_hacks
from testhiml.health.azure.util import TEST_OUTPUTS_PATH, repository_root


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
    remove_and_create_folder(TEST_OUTPUTS_PATH)
    # run the entire test suite
    yield


@pytest.fixture
def random_folder() -> Generator:
    """
    Fixture to automatically create a random directory before executing a test and then
    removing this directory after the test has been executed.
    """
    # create dirs before executing the test
    folder = repository_root() / TEST_OUTPUTS_PATH / str(uuid.uuid4().hex)
    remove_and_create_folder(folder)
    print(f"Created temporary folder for test: {folder}")
    yield folder
