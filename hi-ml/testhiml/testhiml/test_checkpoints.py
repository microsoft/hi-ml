#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Tuple
from unittest import mock
import pytest

import torch

from health_ml.configs.hello_world import HelloWorld
from health_ml.deep_learning_config import WorkflowParams
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import (
    CHECKPOINT_EPOCH_KEY,
    LAST_CHECKPOINT_FILE_NAME,
    MODEL_WEIGHTS_DIR_NAME,
    CheckpointDownloader,
    CheckpointParser,
    _get_checkpoint_files,
    download_checkpoints_from_run,
    find_recovery_checkpoint_in_downloaded_files,
    find_recovery_checkpoint_on_disk_or_cloud,
    _load_epoch_from_checkpoint)
from health_ml.utils.checkpoint_handler import CheckpointHandler
from health_ml.utils.common_utils import (
    AUTOSAVE_CHECKPOINT_CANDIDATES,
    CHECKPOINT_FOLDER,
    CHECKPOINT_SUFFIX,
    DEFAULT_AML_CHECKPOINT_DIR,
    DEFAULT_AML_UPLOAD_DIR)
from testazure.utils_testazure import create_unittest_run_object
from testhiml.utils.fixed_paths_for_tests import full_test_data_path, mock_run_id
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


def test_checkpoint_downloader_run_id() -> None:
    checkpoint_downloader = CheckpointDownloader(run_id="dummy_run_id")
    assert checkpoint_downloader.run_id == "dummy_run_id"
    assert checkpoint_downloader.checkpoint_filename == LAST_CHECKPOINT_FILE_NAME
    assert checkpoint_downloader.remote_checkpoint_dir == Path(DEFAULT_AML_CHECKPOINT_DIR)

    checkpoint_downloader = CheckpointDownloader(run_id="dummy_run_id:best.ckpt")
    assert checkpoint_downloader.run_id == "dummy_run_id"
    assert checkpoint_downloader.checkpoint_filename == "best.ckpt"
    assert checkpoint_downloader.remote_checkpoint_dir == Path(DEFAULT_AML_CHECKPOINT_DIR)

    checkpoint_downloader = CheckpointDownloader(run_id="dummy_run_id:custom/path/best.ckpt")
    assert checkpoint_downloader.run_id == "dummy_run_id"
    assert checkpoint_downloader.checkpoint_filename == "best.ckpt"
    assert checkpoint_downloader.remote_checkpoint_dir == Path("custom/path")


def _test_invalid_checkpoint(checkpoint: str) -> None:
    with pytest.raises(ValueError, match=r"Invalid checkpoint "):
        CheckpointParser(checkpoint=checkpoint)
        WorkflowParams(local_datasets=Path("foo"), src_checkpoint=checkpoint).validate()


def test_validate_checkpoint_parser() -> None:

    _test_invalid_checkpoint(checkpoint="dummy/local/path/model.ckpt")
    _test_invalid_checkpoint(checkpoint="INV@lid%RUN*id")
    _test_invalid_checkpoint(checkpoint="http/dummy_url-com")

    # The following should be okay
    checkpoint = str(full_test_data_path(suffix="hello_world_checkpoint.ckpt"))
    CheckpointParser(checkpoint=checkpoint)
    WorkflowParams(local_datasets=Path("foo"), src_checkpoint=CheckpointParser(checkpoint)).validate()
    checkpoint = mock_run_id(id=0)
    CheckpointParser(checkpoint=checkpoint)
    WorkflowParams(local_datasets=Path("foo"), src_checkpoint=CheckpointParser(checkpoint)).validate()


def get_checkpoint_handler(tmp_path: Path, src_checkpoint: str) -> Tuple[LightningContainer, CheckpointHandler]:
    container = LightningContainer()
    container.set_output_to(tmp_path)
    container.checkpoint_folder.mkdir(parents=True)
    container.src_checkpoint = CheckpointParser(src_checkpoint)
    return container, CheckpointHandler(container=container, project_root=tmp_path)


def test_load_model_checkpoints_from_url(tmp_path: Path) -> None:
    WEIGHTS_URL = (
        "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/" "simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
    )

    container, checkpoint_handler = get_checkpoint_handler(tmp_path, WEIGHTS_URL)
    download_folder = container.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
    assert container.src_checkpoint.is_url
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_path
    assert checkpoint_handler.trained_weights_path.exists()
    assert checkpoint_handler.trained_weights_path.parent == download_folder


def test_load_model_checkpoints_from_local_file(tmp_path: Path) -> None:
    local_checkpoint_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")

    container, checkpoint_handler = get_checkpoint_handler(tmp_path, str(local_checkpoint_path))
    assert container.src_checkpoint.is_local_file
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_path
    assert checkpoint_handler.trained_weights_path.exists()
    assert checkpoint_handler.trained_weights_path == local_checkpoint_path


@pytest.mark.parametrize("src_chekpoint_filename", ["", "best_val_loss.ckpt", "custom/path/model.ckpt"])
def test_load_model_checkpoints_from_aml_run_id(src_chekpoint_filename: str, tmp_path: Path) -> None:
    with mock.patch("health_ml.utils.checkpoint_utils.get_workspace") as mock_get_workspace:
        mock_get_workspace.return_value = DEFAULT_WORKSPACE.workspace
        run_id = mock_run_id(id=0)
        src_checkpoint = f"{run_id}:{src_chekpoint_filename}" if src_chekpoint_filename else run_id
        container, checkpoint_handler = get_checkpoint_handler(tmp_path, src_checkpoint)
        checkpoint_path = "custom/path" if "custom" in src_checkpoint else DEFAULT_AML_CHECKPOINT_DIR
        src_checkpoint_filename = (
            src_chekpoint_filename.split("/")[-1]
            if src_chekpoint_filename
            else LAST_CHECKPOINT_FILE_NAME
        )
        expected_weights_path = container.checkpoint_folder / run_id / checkpoint_path / src_checkpoint_filename
        assert container.src_checkpoint.is_aml_run_id
        checkpoint_handler.download_recovery_checkpoints_or_weights()
        assert checkpoint_handler.trained_weights_path
        assert checkpoint_handler.trained_weights_path.exists()
        assert checkpoint_handler.trained_weights_path == expected_weights_path


def test_custom_checkpoint_for_test(tmp_path: Path) -> None:
    """Test if the logic to choose a checkpoint for inference works.
    """
    # Default behaviour: checkpoint handler returns the default inference checkpoint specified by the container.
    container = HelloWorld()
    container.set_output_to(tmp_path)
    container.checkpoint_folder.mkdir(parents=True)
    last_checkpoint = container.checkpoint_folder / LAST_CHECKPOINT_FILE_NAME
    last_checkpoint.touch()
    checkpoint_handler = CheckpointHandler(container=container, project_root=tmp_path)
    checkpoint_handler.additional_training_done()
    assert container.get_checkpoint_to_test() == last_checkpoint
    # Now mock a container that has the get_checkpoint_to_test method overridden. If the checkpoint exists,
    # the checkpoint handler should return it.
    mock_checkpoint = tmp_path / "mock.txt"
    mock_checkpoint.touch()
    with mock.patch("health_ml.configs.hello_world.HelloWorld.get_checkpoint_to_test") as mock1:
        mock1.return_value = mock_checkpoint
        assert checkpoint_handler.get_checkpoint_to_test() == mock_checkpoint
        mock1.assert_called_once()

    # If the get_checkpoint_to_test method is overridden, and the checkpoint file does not exist, an error should
    # be raised.
    does_not_exist = Path("does_not_exist")
    with mock.patch("health_ml.configs.hello_world.HelloWorld.get_checkpoint_to_test") as mock2:
        mock2.return_value = does_not_exist
        with pytest.raises(FileNotFoundError) as ex:
            checkpoint_handler.get_checkpoint_to_test()
        assert str(does_not_exist) in str(ex)
        mock2.assert_called_once()


def write_empty_checkpoint_file(path: Path, epoch: int, file_name: str = "") -> Path:
    """Writes a dummy torch checkpoint file to the given path."""
    full_path = (path / (file_name or LAST_CHECKPOINT_FILE_NAME)).with_suffix(CHECKPOINT_SUFFIX)
    checkpoint_dict = {CHECKPOINT_EPOCH_KEY: epoch}
    torch.save(checkpoint_dict, full_path)
    return full_path


def test_find_recovery_checkpoints_local(tmp_path: Path) -> None:
    """Test if the logic to find recovery checkpoints on the local disk works.
    """
    # If no checkpoint file is found, the function should return None.
    assert find_recovery_checkpoint_on_disk_or_cloud(tmp_path) is None
    # Write checkpoint files with increasing epoch numbers, for all 3 accepted filenames.
    # Each time, this should be recognized as the now most recent checkpoint.
    for (epoch, filename) in [
        (1, AUTOSAVE_CHECKPOINT_CANDIDATES[0]),
        (2, AUTOSAVE_CHECKPOINT_CANDIDATES[1]),
        (3, LAST_CHECKPOINT_FILE_NAME),
    ]:
        write_empty_checkpoint_file(tmp_path, epoch, filename)
        assert _load_epoch_from_checkpoint(find_recovery_checkpoint_on_disk_or_cloud(tmp_path)) == epoch
    # Write a checkpoint with a name that is not recognized by the function: It should still return
    # 3 as the highest epoch
    write_empty_checkpoint_file(tmp_path, 100, f"othercheckpoint{CHECKPOINT_SUFFIX}")
    assert _load_epoch_from_checkpoint(find_recovery_checkpoint_on_disk_or_cloud(tmp_path)) == 3


def test_get_checkpoint_filenames() -> None:
    """Test matching for folders and file names for recovery checkpoints. This should cover the
    folders for different retries, and all 3 types of checkpoints (2 autosave and last)"""
    output_folder = Path(DEFAULT_AML_UPLOAD_DIR)
    file = str(output_folder / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[0])
    filtered = _get_checkpoint_files([file])
    assert filtered == [file]

    file = str(output_folder / "retry_002" / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[1])
    filtered = _get_checkpoint_files([file])
    assert filtered == [file]

    file = str(output_folder / "retry_001" / CHECKPOINT_FOLDER / LAST_CHECKPOINT_FILE_NAME)
    filtered = _get_checkpoint_files([file])
    assert filtered == [file]


def test_get_checkpoint_filenames_non_matching() -> None:
    output_folder = Path(DEFAULT_AML_UPLOAD_DIR)
    # No checkpoint folder
    file = str(output_folder / AUTOSAVE_CHECKPOINT_CANDIDATES[0])
    assert _get_checkpoint_files([file]) == []

    # Retry folder has wrong name (should be padded to 3 digits, not 4)
    file = str(output_folder / "retry_0002" / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[1])
    assert _get_checkpoint_files([file]) == []


def test_find_recovery_checkpoints_in_cloud(tmp_path: Path) -> None:
    """Test if the logic to find recovery checkpoints in AzureML works.
    """
    empty_file = (tmp_path / "empty.txt")
    empty_file.touch()
    highest_epoch = 100
    # Write 3 files, check that the highest epoch is returned.
    file1 = write_empty_checkpoint_file(tmp_path, 1, "epoch1")
    file100 = write_empty_checkpoint_file(tmp_path, highest_epoch, "epoch100")
    recovery_checkpoint1 = find_recovery_checkpoint_in_downloaded_files(tmp_path)
    assert recovery_checkpoint1 is not None
    assert _load_epoch_from_checkpoint(recovery_checkpoint1) == highest_epoch

    # Create an AzureML run, upload the files, download again, and check that the highest epoch is returned.
    run = create_unittest_run_object()
    try:
        output_folder = Path(DEFAULT_AML_UPLOAD_DIR)
        other_file = "some_other_file.txt"
        # Create 3 files in the run: one in the default folder, no checkpoint in retry folder 001, and a valid
        # checkpoint file in retry folder 002
        highest_epoch_file = output_folder / "retry_002" / CHECKPOINT_FOLDER / LAST_CHECKPOINT_FILE_NAME
        files_to_upload = [
            (file1, output_folder / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[0]),
            (empty_file, output_folder / "retry_001" / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[1]),
            (file100, highest_epoch_file),
            (empty_file, output_folder / other_file)
        ]
        for (file, name) in files_to_upload:
            run.upload_file(name=str(name), path_or_stream=str(file))
        run.flush()

        # Check if we can download all those files to a local folder.
        new_folder = tmp_path / "new_folder"
        download_checkpoints_from_run(run, new_folder)
        for file in [
            output_folder / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[0],
            output_folder / "retry_001" / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[1],
            output_folder / "retry_002" / CHECKPOINT_FOLDER / LAST_CHECKPOINT_FILE_NAME,
        ]:
            assert (new_folder / file).is_file()
        # A file that is not a checkpoint should not be downloaded.
        assert list(new_folder.glob(f"**/{other_file}")) == []

        # When choosing the best checkpoint in that folder, it should be the one with the highest epoch.
        found_highest_epoch_file = find_recovery_checkpoint_in_downloaded_files(new_folder)
        assert found_highest_epoch_file is not None
        assert str(found_highest_epoch_file).endswith(str(highest_epoch_file)), \
            f"Highest epoch file should be {highest_epoch_file}, but was {found_highest_epoch_file}"
        assert _load_epoch_from_checkpoint(found_highest_epoch_file) == highest_epoch

        empty_temp_folder = tmp_path / "no_such_folder"
        empty_temp_folder.mkdir()
        with mock.patch("health_ml.utils.checkpoint_utils.is_running_in_azure_ml", return_value=True):
            with mock.patch("health_ml.utils.checkpoint_utils.RUN_CONTEXT", run):
                recovery_checkpoint = find_recovery_checkpoint_on_disk_or_cloud(empty_temp_folder)
                assert recovery_checkpoint is not None
                assert _load_epoch_from_checkpoint(recovery_checkpoint) == highest_epoch
    finally:
        run.complete()
