import shutil
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pandas.testing import assert_frame_equal
from typing import Generator, List

from health_cpath.utils.montage import (
    MONTAGE_FILE,
    MontageCreation,
    dataset_from_folder,
    dataset_to_records,
    make_montage,
    make_montage_from_dir,
    restrict_dataset,
)
from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.scripts.create_montage import main as script_main
from health_cpath.utils.naming import SlideKey
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.slides_generator import MockPandaSlidesGenerator
from testhisto.utils.utils_testhisto import assert_binary_files_match, full_ml_test_data_path, wait_until_file_exists


# Set this to True to update all stored images in the test_data folder.
UPDATE_STORED_RESULTS = False

NUM_SLIDES = 6


def expected_results_folder() -> Path:
    """Gets the path to the folder where the expected montage results are stored.

    :return: The path to the folder where the expected results are stored.
    """
    return full_ml_test_data_path("montages")


def _create_slides_images(tmp_path: Path) -> MockPandaSlidesGenerator:
    print(f"Result folder: {tmp_path}")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_path,
        mock_type=MockHistoDataType.FAKE,
        n_tiles=4,
        n_slides=NUM_SLIDES,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
    )
    wsi_generator.generate_mock_histo_data()
    print(f"Generated images in {tmp_path}")
    return wsi_generator


@pytest.fixture(scope="module")
def temp_panda_dataset(tmp_path_factory: pytest.TempPathFactory) -> Generator:
    """A fixture that creates a PandaDataset object with randomly created slides.
    """
    tmp_path = tmp_path_factory.mktemp("mock_panda")
    _create_slides_images(tmp_path)
    usecols = [PandaDataset.SLIDE_ID_COLUMN, PandaDataset.MASK_COLUMN]
    yield PandaDataset(root=tmp_path, dataframe_kwargs={"usecols": usecols + list(PandaDataset.METADATA_COLUMNS)})


@pytest.fixture(scope="module")
def temp_slides(tmp_path_factory: pytest.TempPathFactory) -> Generator:
    """A fixture that creates a folder (Path object) with randomly created slides.
    """
    tmp_path = tmp_path_factory.mktemp("mock_wsi")
    _create_slides_images(tmp_path)
    yield tmp_path


@pytest.fixture(scope="module")
def temp_slides_dataset(tmp_path_factory: pytest.TempPathFactory) -> Generator:
    """A fixture that creates a SlidesDataset object with randomly created slides.
    """
    tmp_path = tmp_path_factory.mktemp("mock_slides")
    wsi_generator = _create_slides_images(tmp_path)
    # Create a CSV file with the 3 required columns for montage creation. Mask is optional.
    metadata = {
        SlideKey.SLIDE_ID: [f"ID {i}" for i in range(NUM_SLIDES)],
        SlideKey.IMAGE: wsi_generator.generated_files,
        SlideKey.LABEL: [f"Label {i}" for i in range(NUM_SLIDES)],
    }
    df = pd.DataFrame(data=metadata)
    csv_filename = tmp_path / SlidesDataset.DEFAULT_CSV_FILENAME
    df.to_csv(csv_filename, index=False)
    # Tests fail non-deterministically, saying that the dataset file does not exist (yet). Hence, wait.
    wait_until_file_exists(csv_filename)
    yield SlidesDataset(root=tmp_path)


def _create_folder_with_images(tmp_path: Path, num_images: int = 4, image_size: int = 20) -> None:
    """Creates a folder with images.

    :param tmp_path: The path to the folder where the images should be stored.
    :param num_images: The number of slides that should be created.
    """
    np.random.seed(42)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        image_path = tmp_path / f"image_{i}.png"
        image_np = np.random.uniform(0, 255, size=(image_size, image_size, 3)).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(image_path)


def test_montage_from_dir(tmp_path: Path) -> None:
    """Test montage creation from a directory of images."""
    print(f"Result folder: {tmp_path}")
    np.random.seed(42)
    # Create a directory of images
    image_dir = tmp_path / "images"
    thumb_size = 20
    _create_folder_with_images(image_dir, num_images=4, image_size=thumb_size)

    # Create a montage from the directory
    file_name = "montage_from_random_thumbs.png"
    montage_path = tmp_path / file_name
    montage_image = make_montage_from_dir(image_dir, num_cols=2)
    # We have 2 columns, so the montage should be 2x the size of the thumbnail, plus a 2px border and space in between
    pad = 2
    expected_size = 2 * thumb_size + 3 * pad
    assert montage_image.size == (expected_size, expected_size)

    montage_image.save(montage_path)
    assert montage_path.is_file()

    expected_file = expected_results_folder() / file_name
    if UPDATE_STORED_RESULTS:
        shutil.copyfile(montage_path, expected_file)
    assert_binary_files_match(montage_path, expected_file)


@pytest.mark.parametrize("use_masks", [True, False])
def test_montage_from_dataset(tmp_path: Path, temp_panda_dataset: PandaDataset, use_masks: bool) -> None:
    """Test if a montage can be generated from a slides dataset that uses masks."""
    dataset = dataset_to_records(temp_panda_dataset)
    montage = tmp_path / "montage.png"
    make_montage(dataset, out_path=montage, width=1000, num_parallel=1, masks=use_masks)
    assert montage.is_file()
    expected_file = expected_results_folder() / ("montage_with_masks.png" if use_masks else "montage_without_masks.png")
    if UPDATE_STORED_RESULTS:
        shutil.copyfile(montage, expected_file)
    assert_binary_files_match(montage, expected_file)


def test_restrict_dataset() -> None:
    column = "image_id"
    dataset = pd.DataFrame({column: ["a", "b", "c"]})
    included = restrict_dataset(dataset, column, ["a"], include=True)
    assert len(included) == 1
    assert included.iloc[0][column] == "a"
    excluded = restrict_dataset(dataset, column, ["a"], include=False)
    assert len(excluded) == 2
    assert excluded.iloc[0][column] == "b"
    assert excluded.iloc[1][column] == "c"

    # Check the case when the requested value is not in the dataset
    included2 = restrict_dataset(dataset, column, ["nope"], include=True)
    assert len(included2) == 0
    excluded2 = restrict_dataset(dataset, column, ["nope"], include=False)
    assert len(excluded2) == 3


def test_restrict_dataset_with_index() -> None:
    column = "image_id"
    index_column = "index"
    dataset = pd.DataFrame({column: ["a", "b", "c"], index_column: ["0", "1", "2"]})
    dataset = dataset.set_index(index_column)
    included = restrict_dataset(dataset, index_column, ["1"], include=True)
    assert len(included) == 1
    assert included.iloc[0][column] == "b"
    excluded = restrict_dataset(dataset, index_column, ["1"], include=False)
    assert len(excluded) == 2
    assert excluded.iloc[0][column] == "a"
    assert excluded.iloc[1][column] == "c"

    # Check the case when the requested value is not in the dataset
    included2 = restrict_dataset(dataset, column, ["nope"], include=True)
    assert len(included2) == 0
    excluded2 = restrict_dataset(dataset, column, ["nope"], include=False)
    assert len(excluded2) == 3


@pytest.mark.parametrize("exclude_items", [True, False])
def test_montage_included_and_excluded1(
        tmp_path: Path,
        temp_slides_dataset: SlidesDataset,
        exclude_items: bool) -> None:
    """Check that a montage with exclusion list is handled correctly."""
    config = MontageCreation()
    out_path = tmp_path / "montage"
    out_path.mkdir(exist_ok=True)
    config.output_path = out_path
    config.width = 1000
    config.parallel = 1
    config.montage_from_included_and_excluded_slides(
        temp_slides_dataset,
        items=["ID 0", "ID 1"],
        exclude_items=exclude_items,
    )
    expected_file = expected_results_folder() / ("montage_excluded.png" if exclude_items else "montage_included.png")
    montage_file = out_path / MONTAGE_FILE
    assert montage_file.is_file()
    if UPDATE_STORED_RESULTS:
        shutil.copyfile(montage_file, expected_file)
    assert_binary_files_match(montage_file, expected_file)


def test_montage_included_and_excluded2(tmp_path: Path, temp_slides_dataset: SlidesDataset) -> None:
    """Check that a montage with exclusion list supplies the correct set of images."""
    out_path = tmp_path / "montage"
    out_path.mkdir(exist_ok=True)
    config = MontageCreation()
    config.output_path = out_path
    config.parallel = 1
    config.width = 1000
    for exclude_items in [True, False]:
        with mock.patch("health_cpath.utils.montage.make_montage") as mock_montage:
            montage_file = config.montage_from_included_and_excluded_slides(
                temp_slides_dataset,
                items=["ID 0", "ID 1"],
                exclude_items=exclude_items,
            )
            assert montage_file is not None
            assert mock_montage.call_count == 1
            records = mock_montage.call_args_list[0][1]["records"]
            assert isinstance(records, List)
            slide_ids = sorted([d[SlideKey.SLIDE_ID] for d in records])
            if exclude_items:
                assert slide_ids == ["ID 2", "ID 3", "ID 4", "ID 5"]
            else:
                assert slide_ids == ["ID 0", "ID 1"]


def test_dataset_from_folder_unique(tmp_path: Path) -> None:
    """Test if a plain dataframe can be created from files in a folder.
    This tests the case where the file names alone are unique."""

    file_names = ["file1.txt", "file2.txt"]
    full_files = [tmp_path / file_name for file_name in file_names]
    for f in full_files:
        f.touch()

    df = dataset_from_folder(tmp_path)
    expected_df = pd.DataFrame({SlideKey.SLIDE_ID: file_names, SlideKey.IMAGE: map(str, full_files)})
    assert_frame_equal(df, expected_df)


def test_dataset_from_folder_duplicate_files(tmp_path: Path) -> None:
    """Test if a plain dataframe can be created from files in a folder.
    This tests the case where the file names alone are not unique."""

    # Place files of the same name in different folders. The dataset should still be created, with the full path
    # as the slide ID.
    file_names = ["folder1/file.txt", "folder2/file.txt"]
    full_files = [tmp_path / file_name for file_name in file_names]
    for f in full_files:
        f.parent.mkdir(exist_ok=True)
        f.touch()

    df = dataset_from_folder(tmp_path)
    expected_df = pd.DataFrame({SlideKey.SLIDE_ID: file_names, SlideKey.IMAGE: map(str, full_files)})
    assert_frame_equal(df, expected_df)


def test_dataset_from_folder_fails(tmp_path: Path) -> None:
    """Test if dataframe creation fails if the argument is not a folder."""
    with pytest.raises(ValueError, match="does not exist or is not a directory"):
        dataset_from_folder(tmp_path / "file.txt")


def test_montage_from_folder(tmp_path: Path, temp_slides: Path) -> None:
    """Test if a montage can be created from files in a folder."""
    dataset = dataset_from_folder(temp_slides, glob_pattern="**/*.tiff")
    assert len(dataset) == NUM_SLIDES
    config = MontageCreation()
    config.output_path = tmp_path
    config.width = 1000
    config.parallel = 2
    result_file = config.montage_from_included_and_excluded_slides(dataset)
    assert result_file is not None
    assert result_file.is_file()
    expected_file = expected_results_folder() / "montage_from_folder.png"
    if UPDATE_STORED_RESULTS:
        shutil.copyfile(result_file, expected_file)
    assert_binary_files_match(result_file, expected_file)


def test_montage_from_folder_full(tmp_path: Path, temp_slides: Path) -> None:
    """Test if a montage can be created from files in a folder, using the commandline entrypoint."""
    config = MontageCreation()
    config.image_glob_pattern = "**/*.tiff"
    config.width = 1000
    config.parallel = 1
    config.output_path = tmp_path / "outputs"
    # Cucim is the only backend that supports TIFF files as created in the test images, openslide fails.
    config.backend = "cucim"
    config.create_montage(input_folder=temp_slides)
    assert (config.output_path / "montage.png").is_file()


def test_montage_fails(tmp_path: Path) -> None:
    """Test if montage creation exits gracefully if files can't be read."""
    # Create a single invalid TIFF file. The code will fail when trying to read the images (during thumbnail
    # creation), but it should still reach the point where it creates the montage. There, it will fail because
    # there is no thumbnails present.
    image_file = tmp_path / "image.tiff"
    image_file.touch()
    config = MontageCreation()
    config.image_glob_pattern = "**/*.tiff"
    config.width = 1000
    config.parallel = 1
    config.input_folder = tmp_path
    with pytest.raises(ValueError, match="Failed to create montage"):
        config.create_montage(input_folder=tmp_path)


def test_montage_no_images(tmp_path: Path) -> None:
    """Test if montage creation fails if no files are present"""
    config = MontageCreation()
    config.input_folder = tmp_path
    config.image_glob_pattern = "**/*.tiff"
    with pytest.raises(ValueError, match="No images found"):
        config.create_montage(input_folder=tmp_path)


def test_exclusion_list(tmp_path: Path) -> None:
    """Test if exclusion lists are read correctly from a CSV file and passed to the montage creation function."""
    config = MontageCreation()
    assert config.read_exclusion_list() == []

    ids = ["id1"]
    exclusion_csv = tmp_path / "exclusion.csv"
    exclusion_df = pd.DataFrame({"col1": ids, "col2": ["something else"]})
    exclusion_df.to_csv(exclusion_csv, index=False)
    config.exclude_by_slide_id = exclusion_csv
    assert config.read_exclusion_list() == ids

    config.image_glob_pattern = "*.png"
    (tmp_path / "image.png").touch()
    with mock.patch.object(config, "montage_from_included_and_excluded_slides") as mock_mont:
        config.create_montage(input_folder=tmp_path)
        assert mock_mont.call_count == 1
        assert mock_mont.call_args[1]["items"] == ids
        assert mock_mont.call_args[1]["exclude_items"]


def test_raises_if_no_glob(tmp_path: Path) -> None:
    """Test for exception if no file pattern specified."""
    config = MontageCreation()
    with pytest.raises(ValueError, match="Unable to load dataset"):
        config.create_montage(input_folder=tmp_path)


def test_raises_if_no_images(tmp_path: Path) -> None:
    """Test for exception if no file pattern specified."""
    config = MontageCreation()
    config.image_glob_pattern = "*.png"
    with pytest.raises(ValueError, match="No images found in folder"):
        config.create_montage(input_folder=tmp_path)


def test_read_dataset_if_csv_present(temp_slides_dataset: SlidesDataset) -> None:
    """Test if a SlidesDataset can be read from a folder that contains a dataset.csv file."""
    dataset_path = temp_slides_dataset.root_dir
    config = MontageCreation()
    dataset = config.read_dataset(dataset_path)
    assert isinstance(dataset, SlidesDataset)


def test_read_dataset_fails_if_no_csv_present(tmp_path: Path) -> None:
    """Test behaviour for reading SlidesDataset if not dataset.csv file is present."""
    config = MontageCreation()
    with pytest.raises(ValueError, match="Unable to load dataset"):
        config.read_dataset(tmp_path)


def test_montage_from_slides_dataset(tmp_path: Path, temp_slides_dataset: SlidesDataset) -> None:
    """Test if a montage can be created via SlidesDataset, when the folder contains a dataset.csv file."""
    dataset_path = temp_slides_dataset.root_dir
    config = MontageCreation()
    config.width = 200
    config.parallel = 1
    outputs = tmp_path / "outputs"
    config.output_path = outputs
    config.create_montage(input_folder=dataset_path)
    montage = outputs / MONTAGE_FILE
    assert montage.is_file()


def test_montage_via_args(tmp_path: Path, temp_slides: Path) -> None:
    """Test if montage creation can be invoked correctly via commandline args."""
    outputs = tmp_path / "outputs"
    with mock.patch("sys.argv",
                    [
                        "",
                        "--dataset", str(temp_slides),
                        "--image_glob_pattern", "**/*.tiff",
                        "--output_path", str(outputs),
                        "--width", "200"
                    ]):
        script_main()
        montage = outputs / MONTAGE_FILE
        assert montage.is_file()
        expected_file = expected_results_folder() / "montage_via_args.png"
        if UPDATE_STORED_RESULTS:
            shutil.copyfile(montage, expected_file)
        assert_binary_files_match(montage, expected_file)
