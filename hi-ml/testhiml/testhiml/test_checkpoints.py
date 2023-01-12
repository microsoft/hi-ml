#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock
import pytest

import torch

from health_azure import RUN_CONTEXT
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
    download_highest_epoch_checkpoint,
    find_checkpoint_with_highest_epoch,
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


def checkpoint_handler_for_hello_world(tmp_path: Path) -> CheckpointHandler:
    """Create a CheckpointHandler for the HelloWorld model. The output is set to `tmp_path`, the checkpoint folder
    of the container is created.

    :param tmp_path: A temporary path to use as output folder.
    :return: CheckpointHandler for the HelloWorld model.
    """
    container = HelloWorld()
    container.set_output_to(tmp_path)
    container.checkpoint_folder.mkdir(parents=True)
    return CheckpointHandler(container=container, project_root=tmp_path)


def test_custom_checkpoint_for_test_1(tmp_path: Path) -> None:
    """Test if the logic to choose a checkpoint for inference works if training has been carried out
    and the inference checkpoint exists: checkpoint handler returns the default inference checkpoint
    specified by the container.
    """
    checkpoint_handler = checkpoint_handler_for_hello_world(tmp_path)
    last_checkpoint = checkpoint_handler.container.checkpoint_folder / LAST_CHECKPOINT_FILE_NAME
    last_checkpoint.touch()
    assert checkpoint_handler.container.get_checkpoint_to_test() == last_checkpoint


def test_custom_checkpoint_for_test_2(tmp_path: Path) -> None:
    """Test if the logic to choose a checkpoint for inference works:
    Mock a container that has the get_checkpoint_to_test method overridden. If the checkpoint exists,
    the checkpoint handler should return it.
    """
    handler = checkpoint_handler_for_hello_world(tmp_path)
    handler.additional_training_done()
    mock_checkpoint = tmp_path / "mock.txt"
    mock_checkpoint.touch()
    with mock.patch("health_ml.configs.hello_world.HelloWorld.get_checkpoint_to_test") as mock_1:
        mock_1.return_value = mock_checkpoint
        assert handler.get_checkpoint_to_test() == mock_checkpoint
        mock_1.assert_called_once()


def test_custom_checkpoint_for_test_3(tmp_path: Path) -> None:
    """Test if the logic to choose a checkpoint for inference works:
    If the get_checkpoint_to_test method is overridden, and the checkpoint file does not exist, an error should
    be raised.
    """
    handler = checkpoint_handler_for_hello_world(tmp_path)
    handler.additional_training_done()
    does_not_exist = Path("does_not_exist")
    with mock.patch("health_ml.configs.hello_world.HelloWorld.get_checkpoint_to_test") as mock_1:
        mock_1.return_value = does_not_exist
        with pytest.raises(FileNotFoundError, match="No inference checkpoint file found"):
            handler.get_checkpoint_to_test()
        mock_1.assert_called_once()


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


def test_find_recovery_checkpoint_in_downloaded_files(tmp_path: Path) -> None:
    """Test the logic to find the checkpoints with highest epoch.
    """
    highest_epoch = 100
    # Write 3 files, check that the highest epoch is returned.
    file_1 = write_empty_checkpoint_file(tmp_path, 1, "epoch1")
    file_2 = write_empty_checkpoint_file(tmp_path, 2, "epoch2")
    file_100 = write_empty_checkpoint_file(tmp_path, highest_epoch, "epoch100")

    checkpoint_highest = find_recovery_checkpoint_in_downloaded_files(tmp_path)
    assert checkpoint_highest is not None
    assert checkpoint_highest == file_100
    assert _load_epoch_from_checkpoint(checkpoint_highest) == highest_epoch

    # When supplying the `delete_files` argument, all files apart from the one with the highest epoch should be deleted.
    checkpoint_highest2 = find_checkpoint_with_highest_epoch([file_1, file_2, file_100], delete_files=True)
    assert checkpoint_highest2 is not None
    assert checkpoint_highest == file_100
    assert checkpoint_highest.is_file()
    assert not file_1.is_file()
    assert not file_2.is_file()


def test_find_recovery_checkpoints_in_cloud(tmp_path: Path) -> None:
    """Test if the logic to find recovery checkpoints in AzureML works.
    """
    empty_file = (tmp_path / "empty.txt")
    empty_file.touch()
    highest_epoch = 100
    # Write 3 files, check that the highest epoch is returned.
    file_1 = write_empty_checkpoint_file(tmp_path, 1, "epoch1")
    file_100 = write_empty_checkpoint_file(tmp_path, highest_epoch, "epoch100")
    recovery_checkpoint_1 = find_recovery_checkpoint_in_downloaded_files(tmp_path)
    assert recovery_checkpoint_1 is not None
    assert _load_epoch_from_checkpoint(recovery_checkpoint_1) == highest_epoch

    # Create an AzureML run, upload the files, download again, and check that the highest epoch is returned.
    run = create_unittest_run_object()
    try:
        output_folder = Path(DEFAULT_AML_UPLOAD_DIR)
        other_file = "some_other_file.txt"
        # Create 3 files in the run: one in the default folder, no checkpoint in retry folder 001, and a valid
        # checkpoint file in retry folder 002
        highest_epoch_file = output_folder / "retry_002" / CHECKPOINT_FOLDER / LAST_CHECKPOINT_FILE_NAME
        files_to_upload = [
            (file_1, output_folder / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[0]),
            (empty_file, output_folder / "retry_001" / CHECKPOINT_FOLDER / AUTOSAVE_CHECKPOINT_CANDIDATES[1]),
            (file_100, highest_epoch_file),
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


def test_download_highest_epoch_checkpoint_no_checkpoint(tmp_path: Path) -> None:
    """Test logic for downloading the highest epoch checkpoint from a run, when there is no checkpoint."""
    with mock.patch("health_ml.utils.checkpoint_utils.download_files_by_suffix", return_value=[]):
        assert download_highest_epoch_checkpoint(run=None, checkpoint_suffix="", output_folder=tmp_path) is None


def test_download_highest_epoch_checkpoint_invalid(tmp_path: Path) -> None:
    """Test logic for downloading the highest epoch checkpoint from a run, when there is an invalid checkpoint."""
    # There is a file on the run, but it is not a valid checkpoint, and no epoch information can be extracted
    invalid_checkpoint = tmp_path / "invalid_checkpoint.ckpt"
    invalid_checkpoint.touch()
    with mock.patch("health_ml.utils.checkpoint_utils.download_files_by_suffix", return_value=[invalid_checkpoint]):
        assert download_highest_epoch_checkpoint(run=None, checkpoint_suffix="", output_folder=tmp_path) is None
        # Files that are not checkpoints should not be deleted.
        assert invalid_checkpoint.is_file()


def test_download_highest_epoch_checkpoint(tmp_path: Path) -> None:
    """Test logic for downloading the highest epoch checkpoint from a run.
    This is done by mocking the result of downloading checkpoint files one-by-one."""

    # There is a file on the run, and it is a valid checkpoint: Return that.
    file_1 = write_empty_checkpoint_file(tmp_path, 1, "epoch_1")
    with mock.patch("health_ml.utils.checkpoint_utils.download_files_by_suffix", return_value=[file_1]):
        assert download_highest_epoch_checkpoint(run=None, checkpoint_suffix="", output_folder=tmp_path) == file_1
        assert file_1.is_file()

    # Create a case where there are multiple files on the run, and the highest epoch is returned.
    file_200 = write_empty_checkpoint_file(tmp_path, 200, "epoch_200")
    with mock.patch("health_ml.utils.checkpoint_utils.download_files_by_suffix", return_value=[file_1, file_200]):
        assert download_highest_epoch_checkpoint(run=None, checkpoint_suffix="", output_folder=tmp_path) == file_200
        # The file with the highest epoch should be returned, and the other file should be deleted.
        assert not file_1.is_file()
        assert file_200.is_file()


def test_get_relative_checkpoint_path(tmp_path: Path) -> None:
    """Test if the relative checkpoint path is correct."""
    handler = checkpoint_handler_for_hello_world(tmp_path)
    assert str(handler.get_relative_inference_checkpoint_path()) == f"{CHECKPOINT_FOLDER}/{LAST_CHECKPOINT_FILE_NAME}"


def test_get_relative_checkpoint_path_fails(tmp_path: Path) -> None:
    """Test creating a relative checkpoint path when the checkpoint is not in the output folder.
    For that, we need to mock both the checkpoint method and the outputs folder separately: get_checkpoint_to_test
    would take the modified output folder into account.
    """
    handler = checkpoint_handler_for_hello_world(tmp_path)
    original_checkpoint_path = handler.container.get_checkpoint_to_test()
    handler.container.file_system_config.outputs_folder = Path("no_such_folder")
    with mock.patch.object(handler.container, "get_checkpoint_to_test", return_value=original_checkpoint_path):
        with pytest.raises(ValueError, match="Inference checkpoint path should be relative to the container's output"):
            handler.get_relative_inference_checkpoint_path()


def test_download_inference_checkpoint_outside_azureml(tmp_path: Path) -> None:
    """Test if we can download the inference checkpoint via the CheckpointHandler class.
    This test is not running in AzureML: downloading inference checkpoints from the current run should be a no-op.
    """
    handler = checkpoint_handler_for_hello_world(tmp_path)
    assert handler.download_inference_checkpoint() is None


def test_download_inference_checkpoint_in_azureml(tmp_path: Path) -> None:
    """Test if we can download the inference checkpoint via the CheckpointHandler class."""
    handler = checkpoint_handler_for_hello_world(tmp_path)
    relative_checkpoint_path = f"{CHECKPOINT_FOLDER}/{LAST_CHECKPOINT_FILE_NAME}"
    assert str(handler.get_relative_inference_checkpoint_path()) == relative_checkpoint_path
    with mock.patch.multiple("health_ml.utils.checkpoint_handler",
                             is_running_in_azure_ml=MagicMock(return_value=True),
                             is_global_rank_zero=MagicMock(return_value=True)):
        # Mock the case where there is no checkpoint available in the AzureML run.
        mock_download_highest_epoch_checkpoint = mock.MagicMock(return_value=None)
        with mock.patch.multiple("health_ml.utils.checkpoint_handler",
                                 download_highest_epoch_checkpoint=mock_download_highest_epoch_checkpoint):
            assert handler.download_inference_checkpoint(download_folder=tmp_path) is None
            mock_download_highest_epoch_checkpoint.assert_called_once_with(
                run=RUN_CONTEXT,
                checkpoint_suffix=f"{DEFAULT_AML_UPLOAD_DIR}/{relative_checkpoint_path}",
                output_folder=tmp_path)

        # Mock the case where there is at least one checkpoint in the AzureML run.
        epoch = 123
        checkpoint_file = write_empty_checkpoint_file(tmp_path, epoch)
        mock_download_highest_epoch_checkpoint = mock.MagicMock(return_value=checkpoint_file)
        with mock.patch.multiple("health_ml.utils.checkpoint_handler",
                                 download_highest_epoch_checkpoint=mock_download_highest_epoch_checkpoint):
            downloaded = handler.download_inference_checkpoint(download_folder=tmp_path)
            mock_download_highest_epoch_checkpoint.assert_called_once()
            assert downloaded is not None
            assert downloaded.is_file()
            assert _load_epoch_from_checkpoint(downloaded) == epoch


def test_checkpoint_download_triggered_failed(tmp_path: Path) -> None:
    """Test if the download of inference checkpoints from AzureML is triggering, when the checkpoint handler
    is unable to use the pre-trained weights, nor was training done.
    """
    handler = checkpoint_handler_for_hello_world(tmp_path)
    assert not handler.container.get_checkpoint_to_test().is_file()
    with pytest.raises(ValueError, match="Unable to determine which checkpoint should be used for testing"):
        handler.get_checkpoint_to_test()


def test_checkpoint_download_triggered(tmp_path: Path) -> None:
    """Test if the download of inference checkpoints from AzureML is triggering, if the local checkpoint folder
    does not contain a suitable checkpoint.
    """
    handler = checkpoint_handler_for_hello_world(tmp_path)
    # Setting the "training done" flag only triggers the check for checkpoints or downloading
    handler.additional_training_done()

    # Mock that no checkpoint was available in AzureML
    mock_download = MagicMock(return_value=None)
    with mock.patch.object(handler, "download_inference_checkpoint", mock_download):
        with pytest.raises(FileNotFoundError, match="No inference checkpoint file found locally nor in AzureML"):
            handler.get_checkpoint_to_test()
        mock_download.assert_called_once()

    # Mock that there was actually a checkpoint available in AzureML:
    file = tmp_path / "file"
    file.touch()
    mock_download = MagicMock(return_value=file)
    with mock.patch.object(handler, "download_inference_checkpoint", mock_download):
        assert handler.get_checkpoint_to_test() == file
        mock_download.assert_called_once()
