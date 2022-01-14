from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

from health_ml.configs.hello_container import HelloContainer
from health_ml.lightning_container import LightningContainer
from health_ml.model_trainer import (create_lightning_trainer, write_args_file, model_train)
from health_ml.utils.common_utils import ARGS_TXT
from health_ml.utils.lightning_loggers import StoringLogger


def test_write_args_file(tmp_path: Path) -> None:
    config = {
        "Container": {
            "_min_l_rate": 0.0,
            "_model_name": "HelloContainer",
            "adam_betas": "(0.9, 0.999)",
            "azure_datasets": "[]"}
    }
    expected_args_path = tmp_path / ARGS_TXT
    write_args_file(config, tmp_path)

    actual_args = expected_args_path.read_text()
    assert actual_args == str(config)


def test_create_lightning_trainer() -> None:
    container = LightningContainer()
    trainer, storing_logger = create_lightning_trainer(container)

    assert trainer.gpus == container.num_gpus_per_node()
    # by default, trainer's num_nodes is 1
    assert trainer.num_nodes == 1
    assert trainer.default_root_dir == str(container.outputs_folder)
    assert trainer.limit_train_batches == 1.0
    assert trainer.terminate_on_nan == container.detect_anomaly

    assert isinstance(storing_logger, StoringLogger)
    assert storing_logger.hyperparams is None
    assert len(storing_logger.results_per_epoch) == 0
    assert len(storing_logger.train_diagnostics) == 0
    assert len(storing_logger.val_diagnostics) == 0
    assert len(storing_logger.results_without_epoch) == 0


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

            trainer, storing_logger = model_train(container)

            mock_trainer.fit.assert_called_once()
            mock_trainer.logger.close.assert_called_once()

            assert trainer == mock_trainer
            assert storing_logger == mock_storing_logger
