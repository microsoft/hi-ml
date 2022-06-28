import shutil
from pathlib import Path

import pytest
from typing import Generator, Tuple
from unittest.mock import patch

from health_ml.configs.hello_world import HelloWorld  # type: ignore
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner
from health_ml.utils.checkpoint_handler import CheckpointHandler


@pytest.fixture(scope="module")
def ml_runner_no_setup() -> MLRunner:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = LightningContainer(num_epochs=1)
    runner = MLRunner(experiment_config=experiment_config, container=container)
    return runner


@pytest.fixture(scope="module")
def ml_runner() -> Generator:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = LightningContainer(num_epochs=1)
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture(scope="module")
def ml_runner_with_container() -> Generator:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _mock_model_train(
    chekpoint_handler: CheckpointHandler, container: LightningContainer, num_nodes: int
) -> Tuple[str, str]:
    return "trainer", "storing_logger"


def test_ml_runner_setup(ml_runner_no_setup: MLRunner) -> None:
    """Check that all the necessary methods get called during setup"""
    assert not ml_runner_no_setup._has_setup_run
    with patch.object(ml_runner_no_setup, "container", spec=LightningContainer) as mock_container:
        with patch("health_ml.run_ml.seed_everything") as mock_seed:
            ml_runner_no_setup.setup()
            mock_container.get_effective_random_seed.assert_called()
            mock_container.setup.assert_called_once()
            mock_container.create_lightning_module_and_store.assert_called_once()
            assert ml_runner_no_setup._has_setup_run
            mock_seed.assert_called_once()


def test_set_run_tags_from_parent(ml_runner: MLRunner) -> None:
    """Test that set_run_tags_from_parents causes set_tags to get called"""
    with pytest.raises(AssertionError) as ae:
        ml_runner.set_run_tags_from_parent()
        assert "should only be called in a Hyperdrive run" in str(ae)

    with patch("health_ml.run_ml.PARENT_RUN_CONTEXT") as mock_parent_run_context:
        with patch("health_ml.run_ml.RUN_CONTEXT") as mock_run_context:
            mock_parent_run_context.get_tags.return_value = {"tag": "dummy_tag"}
            ml_runner.set_run_tags_from_parent()
            mock_run_context.set_tags.assert_called()


def test_run(ml_runner: MLRunner) -> None:
    """Test that model runner gets called """
    ml_runner.setup()
    assert not ml_runner.checkpoint_handler.has_continued_training
    with patch.object(ml_runner, "run_inference"):
        with patch.object(ml_runner, "checkpoint_handler"):
            with patch("health_ml.run_ml.model_train", new=_mock_model_train):
                ml_runner.run()
                assert ml_runner._has_setup_run
                # expect _mock_model_train to be called and the result of ml_runner.storing_logger
                # updated accordingly
                assert ml_runner.storing_logger == "storing_logger"
                assert ml_runner.checkpoint_handler.has_continued_training


def test_run_inference(ml_runner_with_container: MLRunner, tmp_path: Path) -> None:
    """
    Test that run_inference gets called as expected.
    """
    def _expected_files_exist() -> bool:
        output_dir = ml_runner_with_container.container.outputs_folder
        if not output_dir.is_dir():
            return False
        expected_files = ["test_mse.txt", "test_mae.txt"]
        return all([(output_dir / p).exists() for p in expected_files])

    # create the test data
    import numpy as np
    import torch

    N = 100
    x = torch.rand((N, 1)) * 10
    y = 0.2 * x + 0.1 * torch.randn(x.size())
    xy = torch.cat((x, y), dim=1)
    data_path = tmp_path / "hellocontainer.csv"
    np.savetxt(data_path, xy.numpy(), delimiter=",")

    expected_ckpt_path = ml_runner_with_container.container.outputs_folder / "checkpoints" / "last.ckpt"
    assert not expected_ckpt_path.exists()
    # update the container to look for test data at this location
    ml_runner_with_container.container.local_dataset_dir = tmp_path
    assert not _expected_files_exist()

    actual_train_ckpt_path = ml_runner_with_container.checkpoint_handler.get_recovery_or_checkpoint_path_train()
    assert actual_train_ckpt_path is None
    ml_runner_with_container.run()
    actual_train_ckpt_path = ml_runner_with_container.checkpoint_handler.get_recovery_or_checkpoint_path_train()
    assert actual_train_ckpt_path == expected_ckpt_path

    actual_test_ckpt_path = ml_runner_with_container.checkpoint_handler.get_checkpoint_to_test()
    assert actual_test_ckpt_path == expected_ckpt_path
    assert actual_test_ckpt_path.is_file()
    # After training, the outputs directory should now exist and contain the 2 error files
    assert _expected_files_exist()
