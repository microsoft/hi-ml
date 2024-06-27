"""
Global PyTest configuration -- used to define global fixtures for the entire test suite

DO NOT RENAME THIS FILE: (https://docs.pytest.org/en/latest/fixture.html#sharing-a-fixture-across-tests-in-a-module
-or-class-session)
"""

import logging
import shutil
import uuid
import pytest
from pathlib import Path
from typing import Generator

from health_azure import is_running_in_azure_ml
from health_cpath.utils import health_cpath_package_setup
from health_ml.utils.fixed_paths import OutputFolderForTests
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.tiles_generator import MockPandaTilesGenerator
from testhisto.mocks.slides_generator import MockPandaSlidesGenerator, TilesPositioningType


testhisto_root_dir = Path(__file__).parent
TEST_OUTPUTS_PATH = testhisto_root_dir / "test_outputs"

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

    if is_running_in_azure_ml():
        expected_file = Path("/pathmnist/pathmnist.npz")
    else:
        expected_file = Path("/datasetdrive/himlstorage/datasets/UI/04-19-2022_025955_UTC/pathmnist.npz")
    if not expected_file.exists():
        raise FileNotFoundError(f"Expected file not found: {expected_file}. Did you forget to run `make mount`?")
    yield expected_file.parent


@pytest.fixture(scope="session")
def mock_panda_tiles_root_dir(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_tiles")
    tiles_generator = MockPandaTilesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=4,
        n_slides=15,
        n_channels=3,
        tile_size=28,
        img_size=224,
    )
    logging.info("Generating temporary mock tiles that will be deleted at the end of the session.")
    tiles_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)


@pytest.fixture(scope="session")
def mock_panda_slides_root_dir(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_slides")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=4,
        n_slides=15,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
        tiles_pos_type=TilesPositioningType.RANDOM,
    )
    logging.info("Generating temporary mock slides that will be deleted at the end of the session.")
    wsi_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)


@pytest.fixture(scope="session")
def mock_panda_slides_root_dir_diagonal(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_wsi")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=1,
        n_slides=16,
        n_repeat_diag=4,
        n_repeat_tile=2,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
        tiles_pos_type=TilesPositioningType.DIAGONAL,
    )
    logging.info("Generating mock whole slide images that will be deleted at the end of the session.")
    wsi_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)
