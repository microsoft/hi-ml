"""
Global PyTest configuration -- used to define global fixtures for the entire test suite

DO NOT RENAME THIS FILE: (https://docs.pytest.org/en/latest/fixture.html#sharing-a-fixture-across-tests-in-a-module
-or-class-session)
"""
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Generator

import pytest

from testhisto.testhisto.utils.utils_testhisto import tests_root_directory
tests_root = tests_root_directory()
logging.info(f"Appending {tests_root} to path")
sys.path.insert(0, str(tests_root))
RELATIVE_TEST_OUTPUTS_PATH = "test_outputs"
TEST_OUTPUTS_PATH = tests_root / RELATIVE_TEST_OUTPUTS_PATH

# temporary workaround until these hi-ml package release
himl_root = tests_root.parent.parent
himl_package_root = himl_root / "hi-ml" / "src"
logging.info(f"Adding {str(himl_package_root)} to path")
sys.path.insert(0, str(himl_package_root))
himl_azure_package_root = himl_root / "hi-ml-azure" / "src"
logging.info(f"Adding {str(himl_azure_package_root)} to path")
sys.path.insert(0, str(himl_azure_package_root))
from health_ml.utils.fixed_paths import OutputFolderForTests  # noqa: E402


def remove_and_create_folder(folder: Path) -> None:
    """
    Delete the folder if it exists, and remakes it. This method ignores errors that can come from
    an explorer window still being open inside of the test result folder.
    """
    folder = Path(folder)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(exist_ok=True, parents=True)


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
