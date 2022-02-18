from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, Mock

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint, ModelSummary, TQDMProgressBar

from health_ml.configs.hello_container import HelloContainer  # type: ignore
from health_ml.lightning_container import LightningContainer
from health_ml.model_trainer import (create_lightning_trainer, write_experiment_summary_file, model_train)
from health_ml.utils.common_utils import EXPERIMENT_SUMMARY_FILE
from health_ml.utils.config_loader import ModelConfigLoader
from health_ml.utils.lightning_loggers import StoringLogger


def test_write_experiment_summary_file(tmp_path: Path) -> None:
    config = {
        "Container": {
            "_min_l_rate": 0.0,
            "_model_name": "HelloContainer",
            "adam_betas": "(0.9, 0.999)",
            "azure_datasets": "[]"}
    }
    expected_args_path = tmp_path / EXPERIMENT_SUMMARY_FILE
    write_experiment_summary_file(config, tmp_path)

    actual_args = expected_args_path.read_text()
    assert actual_args == str(config)


def test_create_lightning_trainer() -> None:
    container = LightningContainer()
    trainer, storing_logger = create_lightning_trainer(container)

    assert trainer.num_gpus == container.num_gpus_per_node()
    # by default, trainer's num_nodes is 1
    assert trainer.num_nodes == 1
    assert trainer.default_root_dir == str(container.outputs_folder)
    assert trainer.limit_train_batches == 1.0
    assert trainer._detect_anomaly == container.detect_anomaly

    assert isinstance(trainer.callbacks[0], TQDMProgressBar)
    assert isinstance(trainer.callbacks[1], ModelSummary)
    assert isinstance(trainer.callbacks[2], GradientAccumulationScheduler)
    assert isinstance(trainer.callbacks[3], ModelCheckpoint)

    assert isinstance(storing_logger, StoringLogger)
    assert storing_logger.hyperparams is None
    assert len(storing_logger.results_per_epoch) == 0
    assert len(storing_logger.train_diagnostics) == 0
    assert len(storing_logger.val_diagnostics) == 0
    assert len(storing_logger.results_without_epoch) == 0


class MyCallback(Callback):
    def on_init_start(self, trainer: Trainer) -> None:
        print("Starting to init trainer")


def test_create_lightning_trainer_with_callbacks() -> None:
    """
    Test that create_lightning_trainer picks up on additional Container callbacks
    """
    def _get_trainer_arguments() -> Dict[str, Any]:
        callbacks = [MyCallback()]
        return {"callbacks": callbacks}

    model_name = "HelloContainer"
    model_config_loader = ModelConfigLoader(model=model_name)
    container = model_config_loader.create_model_config_from_name(model_name)
    container.monitor_gpu = False
    container.monitor_loading = False
    # mock get_trainer_arguments method, since default HelloContainer class doesn't specify any additional callbacks
    container.get_trainer_arguments = _get_trainer_arguments  # type: ignore

    kwargs = container.get_trainer_arguments()
    assert "callbacks" in kwargs
    # create_lightning_trainer(container, )
    trainer, storing_logger = create_lightning_trainer(container)
    # expect trainer to have 5 default callbacks: TQProgressBar, ModelSummary, GradintAccumlationScheduler
    # and 2 ModelCheckpoints, plus any additional callbacks specified in get_trainer_arguments method
    kwarg_callbacks = kwargs.get("callbacks") or []
    expected_num_callbacks = len(kwarg_callbacks) + 5
    assert len(trainer.callbacks) == expected_num_callbacks, f"Found callbacks: {trainer.callbacks}"
    assert any([isinstance(c, MyCallback) for c in trainer.callbacks])

    assert isinstance(storing_logger, StoringLogger)


def test_model_train() -> None:
    container = HelloContainer()
    container.create_lightning_module_and_store()

    with patch.object(container, "get_data_module"):
        with patch("health_ml.model_trainer.create_lightning_trainer") as mock_create_trainer:
            mock_trainer = MagicMock()
            mock_storing_logger = MagicMock()
            mock_create_trainer.return_value = mock_trainer, mock_storing_logger

            mock_trainer.fit = Mock()
            mock_close_logger = Mock()
            mock_trainer.logger = MagicMock(close=mock_close_logger)
            checkpoint_path = None
            trainer, storing_logger = model_train(checkpoint_path, container)

            mock_trainer.fit.assert_called_once()
            mock_trainer.logger.finalize.assert_called_once()

            assert trainer == mock_trainer
            assert storing_logger == mock_storing_logger
