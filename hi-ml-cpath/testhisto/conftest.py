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

# temporary workaround until these hi-ml package release
testhisto_root_dir = Path(__file__).parent
print(f"Adding {testhisto_root_dir} to sys path")
sys.path.insert(0, str(testhisto_root_dir))

TEST_OUTPUTS_PATH = testhisto_root_dir / "test_outputs"

# temporary workaround until these hi-ml package release
himl_root = testhisto_root_dir.parent.parent
himl_package_root = himl_root / "hi-ml" / "src"
logging.info(f"Adding {str(himl_package_root)} to path")
sys.path.insert(0, str(himl_package_root))
himl_azure_package_root = himl_root / "hi-ml-azure" / "src"
logging.info(f"Adding {str(himl_azure_package_root)} to path")
sys.path.insert(0, str(himl_azure_package_root))
test_himl_azure_package_root = himl_root / "hi-ml-azure" / "testazure"
logging.info(f"Adding {str(test_himl_azure_package_root)} to path")
sys.path.insert(0, str(test_himl_azure_package_root))
from health_ml.utils.fixed_paths import OutputFolderForTests # noqa: E402


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


@pytest.fixture(scope="session")
def tmp_path_to_pathmnist_dataset(tmp_path_factory: pytest.TempPathFactory) -> Generator:
    from testhisto.mocks.utils import download_azure_dataset
    from testhisto.mocks.base_data_generator import MockHistoDataType
    tmp_dir = tmp_path_factory.mktemp(MockHistoDataType.PATHMNIST.value)
    download_azure_dataset(tmp_dir, dataset_id=MockHistoDataType.PATHMNIST.value)
    yield tmp_dir
    shutil.rmtree(tmp_dir)
