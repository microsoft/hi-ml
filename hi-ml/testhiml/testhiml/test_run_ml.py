import shutil
from pathlib import Path

import pytest
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

from health_ml.configs.hello_world import HelloWorld  # type: ignore
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner
from health_ml.utils.common_utils import is_gpu_available
from health_azure.utils import is_global_rank_zero


no_gpu = not is_gpu_available()


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


@pytest.fixture()
def ml_runner_with_container() -> Generator:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


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


def test_get_multiple_trainloader_mode(ml_runner: MLRunner) -> None:
    multiple_trainloader_mode = ml_runner.get_multiple_trainloader_mode()
    assert multiple_trainloader_mode == "max_size_cycle", "train_loader_cycle_mode is available now, "
    "`get_multiple_trainloader_mode` workaround can be safely removed."


def _test_init_training(ml_runner: MLRunner) -> None:
    """Test that training is initialized correctly"""
    ml_runner.setup()
    assert not ml_runner.checkpoint_handler.has_continued_training
    assert ml_runner.trainer is None
    assert ml_runner.storing_logger is None
    with patch("health_ml.run_ml.write_experiment_summary_file") as mock_write_experiment_summary_file:
        ml_runner.init_training()
        if is_global_rank_zero():
            mock_write_experiment_summary_file.assert_called()
        assert ml_runner.storing_logger
        assert ml_runner.trainer


def test_init_training_cpu(ml_runner: MLRunner) -> None:
    """Test that training is initialized correctly"""
    ml_runner.container.max_num_gpus = 0
    _test_init_training(ml_runner)


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_init_training_gpu(ml_runner: MLRunner) -> None:
    """Test that training is initialized correctly in DDP mode"""
    _test_init_training(ml_runner)


def test_run_training() -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    runner = MLRunner(experiment_config=experiment_config, container=container)

    with patch.object(container, "get_data_module"):
        with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
            runner.setup()
            mock_trainer = MagicMock()
            mock_storing_logger = MagicMock()
            mock_create_trainer.return_value = mock_trainer, mock_storing_logger
            runner.init_training()

            assert runner.trainer == mock_trainer
            assert runner.storing_logger == mock_storing_logger

            mock_trainer.fit = Mock()
            mock_close_logger = Mock()
            mock_trainer.logger = MagicMock(close=mock_close_logger)

            runner.run_training()

            mock_trainer.fit.assert_called_once()
            mock_trainer.logger.finalize.assert_called_once()


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_run_validation(run_extra_val_epoch: bool) -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.create_lightning_module_and_store()
    container.run_extra_val_epoch = run_extra_val_epoch
    container.model.run_extra_val_epoch = run_extra_val_epoch  # type: ignore
    runner = MLRunner(experiment_config=experiment_config, container=container)

    with patch.object(container, "get_data_module"):
        with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
            runner.setup()
            mock_trainer = MagicMock()
            mock_storing_logger = MagicMock()
            mock_create_trainer.return_value = mock_trainer, mock_storing_logger
            runner.init_training()

            assert runner.trainer == mock_trainer
            assert runner.storing_logger == mock_storing_logger

            mock_trainer.validate = Mock()

            if run_extra_val_epoch:
                runner.run_validation()

            assert mock_trainer.validate.called == run_extra_val_epoch


def test_run_inference(ml_runner_with_container: MLRunner, tmp_path: Path) -> None:
    """
    Test that run_inference gets called as expected.
    """
    ml_runner_with_container.container.max_num_gpus = 0

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


def test_run(ml_runner_with_container: MLRunner) -> None:
    """Test that model runner gets called """
    ml_runner_with_container.setup()
    assert not ml_runner_with_container.checkpoint_handler.has_continued_training
    with patch.object(ml_runner_with_container, "checkpoint_handler"):
        with patch.object(ml_runner_with_container, "load_model_checkpoint_after_training") as mock_load:
            with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
                mock_trainer = MagicMock()
                mock_storing_logger = MagicMock()
                mock_create_trainer.return_value = mock_trainer, mock_storing_logger

                ml_runner_with_container.run()

                mock_load.assert_called_once()
                assert ml_runner_with_container._has_setup_run
                assert ml_runner_with_container.checkpoint_handler.has_continued_training
