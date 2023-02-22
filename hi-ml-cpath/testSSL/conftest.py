"""
Global PyTest configuration -- used to define global fixtures for the entire test suite

DO NOT RENAME THIS FILE: (https://docs.pytest.org/en/latest/fixture.html#sharing-a-fixture-across-tests-in-a-module
-or-class-session)
"""
import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest

from health_cpath.utils import health_cpath_package_setup  # noqa: E402
from health_ml.utils.fixed_paths import OutputFolderForTests  # noqa: E402
from testSSL.test_ssl_containers import create_cxr_test_dataset  # noqa: E402


testSSL_root_dir = Path(__file__).resolve().parent
TEST_OUTPUTS_PATH = testSSL_root_dir / "test_outputs"

# Reduce logging noise in DEBUG mode
health_cpath_package_setup()


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
    remove_and_create_folder(TEST_OUTPUTS_PATH)
    # run the entire test suite
    yield


@pytest.fixture
def test_output_dirs() -> Generator:
    """
    Fixture to automatically create a random directory before executing a test and then
    removing this directory after the test has been executed.
    """
    # create dirs before executing the test
    root_dir = make_output_dirs_for_test()
    print(f"Created temporary folder for test: {root_dir}")
    # let the test function run
    yield OutputFolderForTests(root_dir=root_dir)


def make_output_dirs_for_test() -> Path:
    """
    Create a random output directory for a test inside the global test outputs root.
    """
    test_output_dir = TEST_OUTPUTS_PATH / str(uuid.uuid4().hex)
    remove_and_create_folder(test_output_dir)

    return test_output_dir


@pytest.fixture(scope="module", autouse=True)
def tests_setup() -> Generator:
    path_to_test_dataset = TEST_OUTPUTS_PATH / "cxr_test_dataset"
    create_cxr_test_dataset(path_to_test_dataset)
    yield
    shutil.rmtree(path_to_test_dataset)
