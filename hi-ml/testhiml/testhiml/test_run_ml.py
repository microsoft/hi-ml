import shutil
from pathlib import Path

import pytest
from typing import Generator, Tuple
from unittest.mock import patch, MagicMock

from health_ml.configs.hello_container import HelloContainer
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner


@pytest.fixture(scope="module")
def ml_runner_no_setup() -> MLRunner:
    experiment_config = ExperimentConfig(model="HelloContainer")
    container = LightningContainer(num_epochs=1)
    runner = MLRunner(experiment_config=experiment_config, container=container)
    return runner


@pytest.fixture(scope="module")
def ml_runner() -> Generator:
    experiment_config = ExperimentConfig(model="HelloContainer")
    container = LightningContainer(num_epochs=1)
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture(scope="module")
def ml_runner_with_container() -> Generator:
    experiment_config = ExperimentConfig(model="HelloContainer")
    container = HelloContainer()
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _mock_model_train(chekpoint_path: Path, container: LightningContainer) -> Tuple[str, str]:
    return "trainer", "storing_logger"


def test_ml_runner_setup(ml_runner_no_setup: MLRunner) -> None:
    """Check that all the necessary methods get called during setup"""
    assert not ml_runner_no_setup._has_setup_run
    with patch.object(ml_runner_no_setup, "container", spec=LightningContainer) as mock_container:
        with patch("health_ml.run_ml.seed_everything") as mock_seed:
            ml_runner_no_setup.setup()
            mock_container.get_effective_random_seed.assert_called_once()
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
    with patch.object(ml_runner, "run_inference_for_lightning_models"):
        with patch.object(ml_runner, "checkpoint_handler"):
            with patch("health_ml.run_ml.model_train", new=_mock_model_train):
                ml_runner.run()
                assert ml_runner._has_setup_run
                # expect _mock_model_train to be called and the result of ml_runner.storing_logger
                # updated accordingly
                assert ml_runner.storing_logger == "storing_logger"
                assert ml_runner.checkpoint_handler.has_continued_training


def test_run_inference_for_lightning_models(ml_runner_with_container: MLRunner) -> None:
    """
    Test that run_inference_for_lightning_models gets called as expected. If no checkpoint paths are
    provided, should raise an error. Otherwise, expect the trainer object's test method to be called
    """
    with patch.object(ml_runner_with_container, "checkpoint_handler") as mock_checkpoint_handler:
        with patch("health_ml.run_ml.model_train", new=_mock_model_train):
            with pytest.raises(ValueError) as e:
                ml_runner_with_container.run()
                assert "expects exactly 1 checkpoint for inference, but got 0" in str(e)

        with patch.object(ml_runner_with_container.container, "load_model_checkpoint"):
            with patch("health_ml.model_trainer.create_lightning_trainer") as mock_create_train_trainer:
                with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_inference_trainer:
                    mock_trainer = MagicMock()
                    mock_create_train_trainer.return_value = mock_trainer, ""
                    mock_create_inference_trainer.return_value = mock_trainer, ""
                    mock_checkpoint_handler.get_checkpoints_to_test.return_value = ['dummypath']
                    ml_runner_with_container.run()
                    mock_trainer.test.assert_called_once()
