#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Tuple
from unittest import mock
import pytest

from health_ml.configs.hello_world import HelloWorld
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_handler import CheckpointHandler
from health_ml.utils.checkpoint_utils import (
    LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
    MODEL_WEIGHTS_DIR_NAME,
    CheckpointDownloader,
)
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from testhiml.utils.fixed_paths_for_tests import full_test_data_path
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


def test_checkpoint_downloader_run_id() -> None:
    with mock.patch("health_ml.utils.checkpoint_utils.CheckpointDownloader.download_checkpoint_if_necessary"):
        checkpoint_downloader = CheckpointDownloader(run_id="dummy_run_id")
        assert checkpoint_downloader.run_id == "dummy_run_id"
        assert checkpoint_downloader.checkpoint_filename == LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
        assert checkpoint_downloader.remote_checkpoint_dir == Path(DEFAULT_AML_CHECKPOINT_DIR)

        checkpoint_downloader = CheckpointDownloader(run_id="dummy_run_id:best.ckpt")
        assert checkpoint_downloader.run_id == "dummy_run_id"
        assert checkpoint_downloader.checkpoint_filename == "best.ckpt"
        assert checkpoint_downloader.remote_checkpoint_dir == Path(DEFAULT_AML_CHECKPOINT_DIR)

        checkpoint_downloader = CheckpointDownloader(run_id="dummy_run_id:custom/path/best.ckpt")
        assert checkpoint_downloader.run_id == "dummy_run_id"
        assert checkpoint_downloader.checkpoint_filename == "best.ckpt"
        assert checkpoint_downloader.remote_checkpoint_dir == Path("custom/path")


def get_checkpoint_handler(tmp_path: Path, src_checkpoint: str) -> Tuple[LightningContainer, CheckpointHandler]:
    container = LightningContainer()
    container.set_output_to(tmp_path)
    container.checkpoint_folder.mkdir(parents=True)
    container.src_checkpoint = src_checkpoint
    return container, CheckpointHandler(container=container, project_root=tmp_path)


def test_load_model_chcekpoints_from_url(tmp_path: Path) -> None:
    WEIGHTS_URL = (
        "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/" "simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
    )

    container, checkpoint_handler = get_checkpoint_handler(tmp_path, WEIGHTS_URL)
    download_folder = container.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
    assert container.checkpoint_is_url
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_path
    assert checkpoint_handler.trained_weights_path.exists()
    assert checkpoint_handler.trained_weights_path.parent == download_folder


def test_load_model_checkpoints_from_local_file(tmp_path: Path) -> None:
    local_checkpoint_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")

    container, checkpoint_handler = get_checkpoint_handler(tmp_path, str(local_checkpoint_path))
    assert container.checkpoint_is_local_file
    checkpoint_handler.download_recovery_checkpoints_or_weights()
    assert checkpoint_handler.trained_weights_path
    assert checkpoint_handler.trained_weights_path.exists()
    assert checkpoint_handler.trained_weights_path == local_checkpoint_path


@pytest.mark.parametrize("src_chekpoint_filename", ["", "best_val_loss.ckpt", "custom/path/model.ckpt"])
def test_load_model_checkpoints_from_aml_run_id(src_chekpoint_filename: str, tmp_path: Path, mock_run_id: str) -> None:
    with mock.patch("health_azure.utils.get_workspace") as mock_get_workspace:
        mock_get_workspace.return_value = DEFAULT_WORKSPACE.workspace
        src_checkpoint = f"{mock_run_id}:{src_chekpoint_filename}" if src_chekpoint_filename else mock_run_id
        container, checkpoint_handler = get_checkpoint_handler(tmp_path, src_checkpoint)
        checkpoint_path = "custom/path" if "custom" in src_checkpoint else DEFAULT_AML_CHECKPOINT_DIR
        src_checkpoint_filename = (
            src_chekpoint_filename.split("/")[-1] if src_chekpoint_filename else LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
        )
        expected_weights_path = container.outputs_folder / mock_run_id / checkpoint_path / src_checkpoint_filename
        assert container.checkpoint_is_aml_run_id
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
    last_checkpoint = container.checkpoint_folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
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
