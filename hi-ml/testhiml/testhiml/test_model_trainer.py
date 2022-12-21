from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from numpy import random
import pytest
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint, ModelSummary, TQDMProgressBar
from pytorch_lightning.profiler import PyTorchProfiler, PassThroughProfiler, AdvancedProfiler, SimpleProfiler

from health_ml.lightning_container import LightningContainer
from health_ml.model_trainer import create_lightning_trainer, write_experiment_summary_file
from health_ml.utils.common_utils import EXPERIMENT_SUMMARY_FILE
from health_ml.utils.config_loader import ModelConfigLoader
from health_ml.utils.diagnostics import TrainingDiagnoticsCallback
from health_ml.utils.lightning_loggers import StoringLogger


def test_write_experiment_summary_file(tmp_path: Path) -> None:
    config = {
        "Container": {
            "_min_l_rate": 0.0,
            "_model_name": "HelloWorld",
            "adam_betas": "(0.9, 0.999)",
            "azure_datasets": "[]",
        }
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
    assert trainer.accumulate_grad_batches == 1

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


@pytest.mark.parametrize("monitor_training", [True, False])
def test_create_lightning_trainer_with_callbacks(monitor_training: bool) -> None:
    """
    Test that create_lightning_trainer picks up on additional Container callbacks along with the default ones.
    """

    def _get_trainer_arguments() -> Dict[str, Any]:
        callbacks = [MyCallback()]
        return {"callbacks": callbacks}

    model_name = "HelloWorld"
    model_config_loader = ModelConfigLoader()
    container = model_config_loader.create_model_config_from_name(model_name)
    container.monitor_gpu = False
    container.monitor_loading = False
    container.monitor_training = monitor_training
    # mock get_trainer_arguments method, since default HelloWorld class doesn't specify any additional callbacks
    container.get_trainer_arguments = _get_trainer_arguments  # type: ignore

    kwargs = container.get_trainer_arguments()
    assert "callbacks" in kwargs
    # create_lightning_trainer(container, )
    trainer, storing_logger = create_lightning_trainer(container)
    # expect trainer to have 5 default callbacks: TQProgressBar, ModelSummary, GradintAccumlationScheduler
    # and 2 ModelCheckpoints, plus any additional callbacks specified in get_trainer_arguments method
    kwarg_callbacks = kwargs.get("callbacks") or []
    expected_num_callbacks = len(kwarg_callbacks) + 5 + int(monitor_training)
    assert len(trainer.callbacks) == expected_num_callbacks, f"Found callbacks: {trainer.callbacks}"
    assert any([isinstance(c, MyCallback) for c in trainer.callbacks])
    assert any([isinstance(c, TrainingDiagnoticsCallback) for c in trainer.callbacks]) if monitor_training else True

    assert isinstance(storing_logger, StoringLogger)


def _get_trainer_arguments_custom_profiler() -> Dict[str, Any]:
    return {"profiler": PyTorchProfiler(profile_memory=True, with_stack=True)}


def test_custom_profiler() -> None:
    """Test that we can specify a custom profiler.
    """

    container = LightningContainer()
    container.get_trainer_arguments = _get_trainer_arguments_custom_profiler  # type: ignore
    trainer, _ = create_lightning_trainer(container)
    assert isinstance(trainer.profiler, PyTorchProfiler)
    assert trainer.profiler._profiler_kwargs["profile_memory"]
    assert trainer.profiler._profiler_kwargs["with_stack"]


def test_pl_profiler_argument_overrides_custom_profiler() -> None:
    """Test that pl_profiler argument overrides any custom profiler ser in get_trainer_arguments of the container.
    """

    container = LightningContainer()
    container.pl_profiler = "advanced"
    container.get_trainer_arguments = _get_trainer_arguments_custom_profiler  # type: ignore
    trainer, _ = create_lightning_trainer(container)
    assert isinstance(trainer.profiler, AdvancedProfiler)


@pytest.mark.parametrize("pl_profiler", ["", "simple", "advanced", "pytorch"])
def test_pl_profiler_properly_instantiated(pl_profiler: str) -> None:
    """Test that profiler is properly instantiated for all supported options.
    """

    pl_profilers = {
        "": PassThroughProfiler,
        "simple": SimpleProfiler,
        "advanced": AdvancedProfiler,
        "pytorch": PyTorchProfiler,
    }
    container = LightningContainer()
    container.pl_profiler = pl_profiler
    trainer, _ = create_lightning_trainer(container)
    assert isinstance(trainer.profiler, pl_profilers[pl_profiler])


def test_create_lightning_trainer_limit_batches() -> None:
    model_name = "HelloWorld"
    model_config_loader = ModelConfigLoader()
    container = model_config_loader.create_model_config_from_name(model_name)
    container.monitor_gpu = False
    container.monitor_loading = False
    container.max_epochs = 1

    container.create_lightning_module_and_store()
    lightning_model = container.model
    data_module = container.get_data_module()

    _mock_logger = MagicMock()
    _mock_logger.log.return_value = None

    # First create a trainer and check what the default number of train, val and test batches is
    trainer, _ = create_lightning_trainer(container)
    # We have to call the 'fit' method on the trainer before it updates the number of batches
    trainer.fit(lightning_model, data_module)
    original_num_train_batches = int(trainer.num_training_batches)
    original_num_val_batches = int(trainer.num_val_batches[0])
    original_num_test_batches = len(data_module.test_dataloader())

    # Now try to limit the number of batches to an integer number
    limit_train_batches_int = random.randint(1, original_num_train_batches)
    limit_val_batches_int = random.randint(1, original_num_val_batches)
    limit_test_batches_int = random.randint(1, original_num_test_batches)
    container.pl_limit_train_batches = limit_train_batches_int
    container.pl_limit_val_batches = limit_val_batches_int
    container.pl_limit_test_batches = limit_test_batches_int

    trainer2, _ = create_lightning_trainer(container)
    assert trainer2.limit_train_batches == limit_train_batches_int
    assert trainer2.limit_val_batches == limit_val_batches_int
    assert trainer2.limit_test_batches == limit_test_batches_int
    trainer2.fit(lightning_model, data_module)
    trainer2.test(model=lightning_model, datamodule=data_module)
    assert trainer2.num_training_batches == limit_train_batches_int
    assert trainer2.num_val_batches[0] == limit_val_batches_int
    assert trainer2.num_test_batches[0] == limit_test_batches_int

    # Try to limit the number of batches with float number (i.e. proportion of full data)
    limit_train_batches_float = random.uniform(0.1, 1.0)
    limit_val_batches_float = random.uniform(0.1, 1.0)
    limit_test_batches_float = random.uniform(0.1, 1.0)
    container.pl_limit_train_batches = limit_train_batches_float
    container.pl_limit_val_batches = limit_val_batches_float
    container.pl_limit_test_batches = limit_test_batches_float
    trainer3, _ = create_lightning_trainer(container)
    assert trainer3.limit_train_batches == limit_train_batches_float
    assert trainer3.limit_val_batches == limit_val_batches_float
    trainer3.fit(lightning_model, data_module)
    trainer3.test(model=lightning_model, datamodule=data_module)
    # The number of batches should be a proportion of the full available set
    assert trainer3.num_training_batches == int(limit_train_batches_float * original_num_train_batches)
    assert trainer3.num_val_batches[0] == int(limit_val_batches_float * original_num_val_batches)
    assert trainer3.num_test_batches[0] == int(limit_test_batches_float * original_num_test_batches)


def test_flag_grad_accum() -> None:
    num_batches = 4
    container = LightningContainer()
    container.pl_accumulate_grad_batches = num_batches
    trainer, _ = create_lightning_trainer(container)
    assert trainer.accumulate_grad_batches == num_batches
