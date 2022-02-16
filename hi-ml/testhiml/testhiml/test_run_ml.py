from pathlib import Path

import pytest
from typing import Tuple
from unittest.mock import patch, MagicMock, Mock

from pytorch_lightning import Callback

import health_ml
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner


@pytest.fixture
def ml_runner() -> MLRunner:
    experiment_config = ExperimentConfig()
    container = LightningContainer(num_epochs=1)
    return MLRunner(experiment_config=experiment_config, container=container)


def test_ml_runner_setup(ml_runner: MLRunner) -> None:
    """
    Check that all the necessary methods get called during setup
    """
    assert not ml_runner._has_setup_run
    with patch.object(ml_runner, "container", spec=LightningContainer) as mock_container:
        with patch("health_ml.run_ml.seed_everything") as mock_seed:
            # mock_container.get_effectie_random_seed = Mock()
            ml_runner.setup()
            mock_container.get_effective_random_seed.assert_called_once()
            mock_container.setup.assert_called_once()
            mock_container.create_lightning_module_and_store.assert_called_once()
            assert ml_runner._has_setup_run
            mock_seed.assert_called_once()


def test_set_run_tags_from_parent(ml_runner: MLRunner) -> None:
    with pytest.raises(AssertionError) as ae:
        ml_runner.set_run_tags_from_parent()
        assert "should only be called in a Hyperdrive run" in str(ae)

    with patch("health_ml.run_ml.PARENT_RUN_CONTEXT") as mock_parent_run_context:
        with patch("health_ml.run_ml.RUN_CONTEXT") as mock_run_context:
            mock_parent_run_context.get_tags.return_value = {"tag": "dummy_tag"}
            ml_runner.set_run_tags_from_parent()
            mock_run_context.set_tags.assert_called()


def _mock_model_train(chekpoint_path: Path, container: LightningContainer) -> Tuple[str, str]:
        return "trainer", "storing_logger"


def test_run(ml_runner: MLRunner) -> None:
    """Test that model runner gets called """
    ml_runner.setup()
    assert not ml_runner.has_continued_training
    with patch.object(ml_runner, "run_inference_for_lightning_models"):
        with patch.object(ml_runner, "checkpoint_handler"):
            with patch("health_ml.run_ml.model_train", new=_mock_model_train):
                ml_runner.run()
                assert ml_runner._has_setup_run
                # expect _mock_model_train to be called and the result of ml_runner.storing_logger
                # updated accordingly
                assert ml_runner.storing_logger == "storing_logger"
                assert ml_runner.has_continued_training


@patch("health_ml.run_ml.create_lightning_trainer")
def test_run_inference_for_lightning_models(mock_create_trainer: MagicMock, ml_runner: MLRunner,
                                            tmp_path: Path) -> None:
    """
    Check that all expected methods are called during inference3
    """
    mock_trainer = MagicMock()
    mock_test_result = [{"result": 1.0}]
    mock_trainer.test.return_value = mock_test_result
    mock_create_trainer.return_value = mock_trainer, ""

    with patch.object(ml_runner, "container") as mock_container:
        mock_container.num_gpus_per_node.return_value = 0
        mock_container.get_trainer_arguments.return_value = {"callbacks": Callback()}
        mock_container.load_model_checkpoint.return_value = Mock()
        mock_container.get_data_module.return_value = Mock()
        mock_container.pl_progress_bar_refresh_rate = None
        mock_container.detect_anomaly = False
        mock_container.pl_limit_train_batches = 1.0
        mock_container.pl_limit_val_batches = 1.0
        mock_container.outputs_folder = tmp_path

        checkpoint_paths = [Path("dummy")]
        result = ml_runner.run_inference_for_lightning_models(checkpoint_paths)
        assert result == mock_test_result

        mock_create_trainer.assert_called_once()
        mock_container.load_model_checkpoint.assert_called_once()
        mock_container.get_data_module.assert_called_once()
        mock_trainer.test.assert_called_once()


def test_on_test_epoch_end(ml_runner: MLRunner):
    ml_runner.setup()
    with patch.object(ml_runner, "checkpoint_handler") as mock_checkpoint_handler:
        with patch("health_ml.run_ml.model_train", new=_mock_model_train):
            with pytest.raises(ValueError) as e:
                ml_runner.run()
                assert "expects exactly 1 checkpoint for inference, but got 0" in str(e)

        with patch.object(ml_runner.container, "load_model_checkpoint"):
            with patch("health_ml.model_trainer.create_lightning_trainer") as mock_create_trainer:
                mock_trainer = MagicMock()
                mock_create_trainer.return_value = mock_trainer, ""
                mock_checkpoint_handler.get_checkpoints_to_test.return_value = ['dummypath']
                ml_runner.run()
                mock_trainer.test.assert_called_once()