#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
import pytest
import numpy as np
import torch
from math import isclose
from pathlib import Path
from typing import Generator
from unittest import mock
from unittest.mock import MagicMock, Mock, patch
from _pytest.logging import LogCaptureFixture
from pytorch_lightning import LightningModule

import mlflow
from pytorch_lightning import Trainer

from health_ml.configs.hello_world import HelloWorld  # type: ignore
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner
from health_ml.utils.checkpoint_handler import CheckpointHandler
from health_ml.utils.checkpoint_utils import CheckpointParser
from health_ml.utils.common_utils import is_gpu_available
from health_ml.utils.lightning_loggers import HimlMLFlowLogger, StoringLogger, get_mlflow_run_id_from_trainer
from health_azure.utils import ENV_EXPERIMENT_NAME, is_global_rank_zero
from testazure.utils_testazure import DEFAULT_WORKSPACE, experiment_for_unittests
from testhiml.utils.fixed_paths_for_tests import mock_run_id

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


@pytest.fixture()
def ml_runner_with_run_id() -> Generator:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.save_checkpoint = True
    container.src_checkpoint = CheckpointParser(mock_run_id(id=0))
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture()
def regression_datadir(tmp_path: Path) -> Generator:
    """Create a temporary directory with a dummy dataset for regression testing."""
    N = 100
    x = torch.rand((N, 1)) * 10
    y = 0.2 * x + 0.1 * torch.randn(x.size())
    xy = torch.cat((x, y), dim=1)
    data_path = tmp_path / "hellocontainer.csv"
    np.savetxt(data_path, xy.numpy(), delimiter=",")
    yield tmp_path
    shutil.rmtree(tmp_path)


def test_ml_runner_setup(ml_runner_no_setup: MLRunner) -> None:
    """Check that all the necessary methods get called during setup"""
    assert not ml_runner_no_setup._has_setup_run
    with patch.object(ml_runner_no_setup, "container", spec=LightningContainer) as mock_container:
        with patch.object(ml_runner_no_setup, "checkpoint_handler", spec=CheckpointHandler) as mock_checkpoint_handler:
            with patch("health_ml.run_ml.seed_everything") as mock_seed:
                with patch("health_ml.run_ml.seed_monai_if_available") as mock_seed_monai:
                    ml_runner_no_setup.setup()
                    mock_container.get_effective_random_seed.assert_called()
                    mock_seed.assert_called_once()
                    mock_seed_monai.assert_called_once()
                    mock_container.create_filesystem.assert_called_once()
                    mock_checkpoint_handler.download_recovery_checkpoints_or_weights.assert_called_once()
                    mock_container.setup.assert_called_once()
                    mock_container.create_lightning_module_and_store.assert_called_once()
                    assert ml_runner_no_setup._has_setup_run


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


def _test_init_training(run_inference_only: bool, ml_runner: MLRunner, caplog: LogCaptureFixture) -> None:
    """Test that training is initialized correctly"""
    ml_runner.container.run_inference_only = run_inference_only
    ml_runner.setup()
    assert not ml_runner.checkpoint_handler.has_continued_training
    assert ml_runner.trainer is None
    assert ml_runner.storing_logger is None

    with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
        with patch.object(ml_runner.container, "get_data_module") as mock_get_data_module:
            with patch("health_ml.run_ml.write_experiment_summary_file") as mock_write_experiment_summary_file:
                with patch.object(
                    ml_runner.checkpoint_handler, "get_recovery_or_checkpoint_path_train"
                ) as mock_get_recovery_or_checkpoint_path_train:
                    with patch("health_ml.run_ml.seed_everything") as mock_seed:
                        mock_create_trainer.return_value = MagicMock(), MagicMock()
                        mock_get_recovery_or_checkpoint_path_train.return_value = "dummy_path"

                        ml_runner.init_training()

                        # Make sure write_experiment_summary_file is only called on rank 0
                        if is_global_rank_zero():
                            mock_write_experiment_summary_file.assert_called()
                        else:
                            mock_write_experiment_summary_file.assert_not_called()

                        # Make sure seed is set correctly with workers=True
                        mock_seed.assert_called_once()
                        assert mock_seed.call_args[0][0] == ml_runner.container.get_effective_random_seed()
                        assert mock_seed.call_args[1]["workers"]

                        mock_get_data_module.assert_called_once()
                        assert ml_runner.data_module is not None

                        if not run_inference_only:
                            mock_get_recovery_or_checkpoint_path_train.assert_called_once()
                            # Validate that the trainer is created correctly
                            assert mock_create_trainer.call_args[1]["resume_from_checkpoint"] == "dummy_path"
                            assert ml_runner.storing_logger is not None
                            assert ml_runner.trainer is not None
                            assert "Environment variables:" in caplog.messages[-1]
                        else:
                            assert ml_runner.trainer is None
                            assert ml_runner.storing_logger is None
                            mock_get_recovery_or_checkpoint_path_train.assert_not_called()


@pytest.mark.parametrize("run_inference_only", [True, False])
def test_init_training_cpu(run_inference_only: bool, ml_runner: MLRunner, caplog: LogCaptureFixture) -> None:
    """Test that training is initialized correctly"""
    ml_runner.container.max_num_gpus = 0
    _test_init_training(run_inference_only, ml_runner, caplog)


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("run_inference_only", [True, False])
def test_init_training_gpu(run_inference_only: bool, ml_runner: MLRunner, caplog: LogCaptureFixture) -> None:
    """Test that training is initialized correctly in DDP mode"""
    _test_init_training(run_inference_only, ml_runner, caplog)


def test_run_training() -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    runner = MLRunner(experiment_config=experiment_config, container=container)

    with patch.object(container, "get_data_module") as mock_get_data_module:
        with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
            runner.setup()
            mock_trainer = MagicMock()
            mock_storing_logger = MagicMock()
            mock_create_trainer.return_value = mock_trainer, mock_storing_logger
            mock_get_data_module.return_value = "dummy_data_module"
            runner.init_training()

            assert runner.trainer == mock_trainer
            assert runner.storing_logger == mock_storing_logger

            mock_trainer.fit = Mock()
            mock_close_logger = Mock()
            mock_trainer.loggers = [MagicMock(close=mock_close_logger)]

            runner.run_training()

            mock_trainer.fit.assert_called_once()
            assert mock_trainer.fit.call_args[0][0] == runner.container.model
            assert mock_trainer.fit.call_args[1]["datamodule"] == "dummy_data_module"
            mock_trainer.loggers[0].finalize.assert_called_once()


@pytest.mark.parametrize("max_num_gpus_inf", [2, 1])
def test_end_training(max_num_gpus_inf: int) -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.max_num_gpus_inference = max_num_gpus_inf
    runner = MLRunner(experiment_config=experiment_config, container=container)

    with patch.object(container, "get_data_module"):
        with patch("health_ml.run_ml.create_lightning_trainer", return_value=(MagicMock(), MagicMock())):
            runner.setup()
            runner.init_training()
            runner.run_training()

        with patch.object(runner.checkpoint_handler, "additional_training_done") as mock_additional_training_done:
            with patch.object(runner, "after_ddp_cleanup") as mock_after_ddp_cleanup:
                with patch("health_ml.run_ml.cleanup_checkpoints") as mock_cleanup_ckpt:
                    environ_before_training = {"old": "environ"}
                    runner.end_training(environ_before_training=environ_before_training)
                    mock_additional_training_done.assert_called_once()
                    mock_cleanup_ckpt.assert_called_once()
                    if max_num_gpus_inf == 1:
                        mock_after_ddp_cleanup.assert_called_once()
                        mock_after_ddp_cleanup.assert_called_with(environ_before_training)
                    else:
                        mock_after_ddp_cleanup.assert_not_called()


@pytest.mark.parametrize("max_num_gpus_inf", [2, 1])
@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
@pytest.mark.parametrize("run_inference_only", [True, False])
def test_init_inference(
    run_inference_only: bool, run_extra_val_epoch: bool, max_num_gpus_inf: int, ml_runner_with_run_id: MLRunner
) -> None:
    ml_runner_with_run_id.container.run_inference_only = run_inference_only
    ml_runner_with_run_id.container.run_extra_val_epoch = run_extra_val_epoch
    ml_runner_with_run_id.container.max_num_gpus_inference = max_num_gpus_inf
    assert ml_runner_with_run_id.container.max_num_gpus == -1  # This is the default value of max_num_gpus
    ml_runner_with_run_id.init_training()
    if run_inference_only:
        expected_mlflow_run_id = None
    else:
        assert ml_runner_with_run_id.trainer is not None
        expected_mlflow_run_id = ml_runner_with_run_id.trainer.loggers[1].run_id  # type: ignore
    if not run_inference_only:
        ml_runner_with_run_id.checkpoint_handler.additional_training_done()
    with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
        with patch.object(ml_runner_with_run_id.container, "get_checkpoint_to_test") as mock_get_checkpoint_to_test:
            with patch.object(ml_runner_with_run_id.container, "get_data_module") as mock_get_data_module:
                mock_checkpoint = MagicMock(is_file=MagicMock(return_value=True))
                mock_get_checkpoint_to_test.return_value = mock_checkpoint
                mock_trainer = MagicMock()
                mock_create_trainer.return_value = mock_trainer, MagicMock()
                mock_get_data_module.return_value = "dummy_data_module"

                assert ml_runner_with_run_id.inference_checkpoint is None
                assert not ml_runner_with_run_id.container.model._on_extra_val_epoch

                ml_runner_with_run_id.init_inference()

                expected_ckpt = str(ml_runner_with_run_id.checkpoint_handler.trained_weights_path)
                expected_ckpt = expected_ckpt if run_inference_only else str(mock_checkpoint)
                assert ml_runner_with_run_id.inference_checkpoint == expected_ckpt

                assert hasattr(ml_runner_with_run_id.container.model, "on_run_extra_validation_epoch")
                assert ml_runner_with_run_id.container.model._on_extra_val_epoch == run_extra_val_epoch

                mock_create_trainer.assert_called_once()
                assert ml_runner_with_run_id.trainer == mock_trainer
                assert ml_runner_with_run_id.container.max_num_gpus == max_num_gpus_inf
                assert mock_create_trainer.call_args[1]["container"] == ml_runner_with_run_id.container
                assert mock_create_trainer.call_args[1]["num_nodes"] == 1
                assert mock_create_trainer.call_args[1]["mlflow_run_for_logging"] == expected_mlflow_run_id
                mock_get_data_module.assert_called_once()
                assert ml_runner_with_run_id.data_module == "dummy_data_module"


@pytest.mark.parametrize("run_inference_only", [True, False])
@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_run_validation(
    run_extra_val_epoch: bool, run_inference_only: bool, ml_runner_with_run_id: MLRunner, caplog: LogCaptureFixture
) -> None:
    ml_runner_with_run_id.container.run_extra_val_epoch = run_extra_val_epoch
    ml_runner_with_run_id.container.run_inference_only = run_inference_only
    ml_runner_with_run_id.init_training()
    mock_datamodule = MagicMock()
    with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
        with patch.object(ml_runner_with_run_id.container, "get_data_module", return_value=mock_datamodule):
            mock_trainer = MagicMock()
            mock_create_trainer.return_value = mock_trainer, MagicMock()
            ml_runner_with_run_id.init_inference()
            assert ml_runner_with_run_id.trainer == mock_trainer
            mock_trainer.validate = Mock()
            ml_runner_with_run_id.run_validation()
            if run_extra_val_epoch or run_inference_only:
                mock_trainer.validate.assert_called_once()
                assert mock_trainer.validate.call_args[1]["ckpt_path"] == ml_runner_with_run_id.inference_checkpoint
                assert mock_trainer.validate.call_args[1]["datamodule"] == mock_datamodule
            else:
                assert "Skipping extra validation" in caplog.messages[-1]
                mock_trainer.validate.assert_not_called()


def test_model_extra_val_epoch_missing_hook(caplog: LogCaptureFixture) -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")

    def _create_model(self) -> LightningModule:  # type: ignore
        return LightningModule()

    with patch("health_ml.configs.hello_world.HelloWorld.create_model", _create_model):
        container = HelloWorld()
        container.create_lightning_module_and_store()
        container.run_extra_val_epoch = True
        runner = MLRunner(experiment_config=experiment_config, container=container)
        runner.setup()
        runner.checkpoint_handler.additional_training_done()
        runner.container.outputs_folder.mkdir(parents=True, exist_ok=True)
        with patch.object(container, "get_data_module"):
            with patch("health_ml.run_ml.create_lightning_trainer", return_value=(MagicMock(), MagicMock())):
                with patch.object(runner.container, "get_checkpoint_to_test") as mock_get_checkpoint_to_test:
                    mock_get_checkpoint_to_test.return_value = MagicMock(is_file=MagicMock(return_value=True))
                    runner.init_inference()
                    runner.run_validation()
                    latest_message = caplog.records[-1].getMessage()
                    assert "Hook `on_run_extra_validation_epoch` is not implemented" in latest_message


def test_run_inference(ml_runner_with_container: MLRunner, regression_datadir: Path) -> None:
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

    expected_ckpt_path = ml_runner_with_container.container.outputs_folder / "checkpoints" / "last.ckpt"
    assert not expected_ckpt_path.exists()
    # update the container to look for test data at this location
    ml_runner_with_container.container.local_dataset_dir = regression_datadir
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


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
@pytest.mark.parametrize("run_inference_only", [True, False])
def test_run(run_inference_only: bool, run_extra_val_epoch: bool, ml_runner_with_container: MLRunner) -> None:
    """Test that model runner gets called """
    ml_runner_with_container.container.run_inference_only = run_inference_only
    ml_runner_with_container.container.run_extra_val_epoch = run_extra_val_epoch
    ml_runner_with_container.setup()
    assert not ml_runner_with_container.checkpoint_handler.has_continued_training

    with patch("health_ml.run_ml.create_lightning_trainer", return_value=(MagicMock(), MagicMock())):
        with patch.multiple(
            ml_runner_with_container,
            checkpoint_handler=mock.DEFAULT,
            run_training=mock.DEFAULT,
            run_validation=mock.DEFAULT,
            run_inference=mock.DEFAULT,
            end_training=mock.DEFAULT,
        ) as mocks:
            ml_runner_with_container.run()
            assert ml_runner_with_container.container.has_custom_test_step()
            assert ml_runner_with_container._has_setup_run
            assert mocks["end_training"] != run_inference_only
            assert mocks["run_training"].called != run_inference_only
            mocks["run_validation"].assert_called_once()
            mocks["run_inference"].assert_called_once()


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_run_inference_only(run_extra_val_epoch: bool, ml_runner_with_run_id: MLRunner) -> None:
    """Test inference only mode. Validation should be run regardless of run_extra_val_epoch status."""
    ml_runner_with_run_id.container.run_inference_only = True
    ml_runner_with_run_id.container.run_extra_val_epoch = run_extra_val_epoch
    assert ml_runner_with_run_id.checkpoint_handler.trained_weights_path
    mock_datamodule = MagicMock()
    with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
        with patch.object(ml_runner_with_run_id.container, "get_data_module", return_value=mock_datamodule):
            with patch.multiple(
                ml_runner_with_run_id,
                run_training=mock.DEFAULT,
            ) as mocks:
                mock_trainer = MagicMock()
                mock_create_trainer.return_value = mock_trainer, MagicMock()
                ml_runner_with_run_id.run()
                mock_create_trainer.assert_called_once()
                mocks["run_training"].assert_not_called()

                mock_trainer.validate.assert_called_once()
                assert mock_trainer.validate.call_args[1]["ckpt_path"] == ml_runner_with_run_id.inference_checkpoint
                assert mock_trainer.validate.call_args[1]["datamodule"] == mock_datamodule
                mock_trainer.test.assert_called_once()
                assert mock_trainer.test.call_args[1]["ckpt_path"] == ml_runner_with_run_id.inference_checkpoint
                assert mock_trainer.test.call_args[1]["datamodule"] == mock_datamodule


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_resume_training_from_run_id(run_extra_val_epoch: bool, ml_runner_with_run_id: MLRunner) -> None:
    ml_runner_with_run_id.container.run_extra_val_epoch = run_extra_val_epoch
    ml_runner_with_run_id.container.max_num_gpus = 0
    ml_runner_with_run_id.container.max_epochs += 10
    assert ml_runner_with_run_id.checkpoint_handler.trained_weights_path
    mock_trainer = MagicMock()
    with patch("health_ml.run_ml.create_lightning_trainer", return_value=(mock_trainer, MagicMock())):
        with patch.object(ml_runner_with_run_id.container, "get_checkpoint_to_test") as mock_get_checkpoint_to_test:
            with patch.object(ml_runner_with_run_id, "run_inference") as mock_run_inference:
                with patch("health_ml.run_ml.cleanup_checkpoints") as mock_cleanup_ckpt:
                    mock_get_checkpoint_to_test.return_value = MagicMock(is_file=MagicMock(return_value=True))
                    ml_runner_with_run_id.run()
                    mock_get_checkpoint_to_test.assert_called_once()
                    mock_cleanup_ckpt.assert_called_once()
                    assert mock_trainer.validate.called == run_extra_val_epoch
                    mock_run_inference.assert_called_once()


def test_model_weights_when_resume_training() -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.max_num_gpus = 0
    container.src_checkpoint = CheckpointParser(mock_run_id(id=0))
    container.resume_training = True
    runner = MLRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    assert runner.checkpoint_handler.trained_weights_path.is_file()  # type: ignore
    with patch("health_ml.run_ml.create_lightning_trainer") as mock_create_trainer:
        mock_create_trainer.return_value = MagicMock(), MagicMock()
        runner.init_training()
        mock_create_trainer.assert_called_once()
        recovery_checkpoint = mock_create_trainer.call_args[1]["resume_from_checkpoint"]
        assert recovery_checkpoint == runner.checkpoint_handler.trained_weights_path


@pytest.mark.parametrize("log_from_vm", [True, False])
def test_log_on_vm(log_from_vm: bool) -> None:
    """Test if the AzureML logger is called when the experiment is run outside AzureML."""
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.max_epochs = 1
    # Mimic an experiment name given on the command line.
    container.experiment = experiment_for_unittests()
    # The tag is used to identify the run, similar to the behaviour when submitting a run to AzureML.
    tag = f"test_log_on_vm [{log_from_vm}]"
    container.tag = tag
    container.log_from_vm = log_from_vm
    runner = MLRunner(experiment_config=experiment_config, container=container)
    # When logging to AzureML, need to provide the unit test AML workspace.
    # When not logging to AzureML, no workspace (and no authentication) should be needed.
    if log_from_vm:
        with patch("health_azure.utils.get_workspace", return_value=DEFAULT_WORKSPACE.workspace):
            runner.run()
    else:
        runner.run()
    # The PL trainer object is created in the init_training method.
    # Check that the AzureML logger is set up correctly.
    assert runner.trainer is not None
    assert runner.trainer.loggers is not None
    assert len(runner.trainer.loggers) > 1
    logger = runner.trainer.loggers[1]
    assert isinstance(logger, HimlMLFlowLogger)


@pytest.mark.fast
def test_experiment_name() -> None:
    """Test that the experiment name is set correctly, choosing either the experiment name given on the commandline
    or the model name"""
    # When the test suite runs on the Github, the environment variable "HIML_EXPERIMENT_NAME" will be set.
    # Remove it to test the default behaviour.
    with patch.dict(os.environ):
        os.environ.pop(ENV_EXPERIMENT_NAME, None)
        container = HelloWorld()
        # No experiment name given on the commandline: use the model name
        model_name = "some_model"
        container._model_name = model_name
        assert container.effective_experiment_name == model_name
        # Experiment name given on the commandline: use the experiment name
        experiment_name = experiment_for_unittests()
        container.experiment = experiment_name
        assert container.effective_experiment_name == experiment_name


def test_get_mlflow_run_id_from_trainer() -> None:
    trainer_without_loggers = Trainer()
    run_id = get_mlflow_run_id_from_trainer(trainer_without_loggers)
    assert run_id is None

    loggers_not_inc_mlflow = [StoringLogger()]
    trainer_with_single_logger = Trainer(logger=loggers_not_inc_mlflow)
    run_id = get_mlflow_run_id_from_trainer(trainer_with_single_logger)
    assert run_id is None

    mock_run_id = "run_id_123"
    loggers_inc_mlflow = [StoringLogger(), HimlMLFlowLogger(run_id=mock_run_id)]
    trainer_with_loggers = Trainer(logger=loggers_inc_mlflow)
    with patch.object(mlflow.tracking.client.TrackingServiceClient, "get_run"):
        run_id = get_mlflow_run_id_from_trainer(trainer_with_loggers)
        assert run_id == mock_run_id


def test_inference_only_metrics_correctness(ml_runner_with_run_id: MLRunner, regression_datadir: Path) -> None:
    ml_runner_with_run_id.container.run_inference_only = True
    ml_runner_with_run_id.container.local_dataset_dir = regression_datadir
    ml_runner_with_run_id.run()
    with open(ml_runner_with_run_id.container.outputs_folder / "test_mse.txt") as f:
        mse = float(f.readlines()[0])
    assert isclose(mse, 0.010806690901517868, abs_tol=1e-3)
    with open(ml_runner_with_run_id.container.outputs_folder / "test_mae.txt") as f:
        mae = float(f.readlines()[0])
    assert isclose(mae, 0.08260975033044815, abs_tol=1e-3)
