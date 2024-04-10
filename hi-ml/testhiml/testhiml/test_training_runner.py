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

from health_ml.configs.hello_world import TEST_MAE_FILE, TEST_MSE_FILE, HelloWorld  # type: ignore
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.runner_base import RunnerBase
from health_ml.training_runner import TrainingRunner
from health_ml.utils.checkpoint_handler import CheckpointHandler
from health_ml.utils.checkpoint_utils import CheckpointParser
from health_ml.utils.common_utils import EFFECTIVE_RANDOM_SEED_KEY_NAME, is_gpu_available
from health_ml.utils.lightning_loggers import HimlMLFlowLogger, StoringLogger, get_mlflow_run_id_from_trainer
from health_azure.utils import ENV_EXPERIMENT_NAME, is_global_rank_zero
from testazure.utils_testazure import DEFAULT_WORKSPACE, experiment_for_unittests
from testhiml.utils.fixed_paths_for_tests import full_test_data_path

no_gpu = not is_gpu_available()
hello_world_checkpoint = full_test_data_path(suffix="hello_world_checkpoint.ckpt")


@pytest.fixture()
def training_runner_no_setup(tmp_path: Path) -> TrainingRunner:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = LightningContainer(num_epochs=1)
    container.set_output_to(tmp_path)
    runner = TrainingRunner(experiment_config=experiment_config, container=container)
    return runner


@pytest.fixture()
def training_runner(tmp_path: Path) -> Generator:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = LightningContainer(num_epochs=1)
    container.set_output_to(tmp_path)
    runner = TrainingRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture()
def training_runner_hello_world(tmp_path: Path) -> Generator:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.set_output_to(tmp_path)
    runner = TrainingRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    yield runner
    output_dir = runner.container.file_system_config.outputs_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture()
def training_runner_hello_world_with_checkpoint() -> Generator:
    """
    A fixture with a training runner for the HelloWorld model that has a src_checkpoint set.
    """
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.save_checkpoint = True
    container.src_checkpoint = CheckpointParser(str(hello_world_checkpoint))
    runner = TrainingRunner(experiment_config=experiment_config, container=container)
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


def create_mlflow_trash_folder(runner: RunnerBase) -> None:
    """Create a trash folder where MLFlow expects its deleted runs.
    This is a workaround for sporadic test failures: When reading out the run_id, MLFlow checks its own
    deleted runs folder, but that (or one of its parents) does not exist
    """
    trash_folder = runner.container.outputs_folder / "mlruns" / ".trash"
    trash_folder.mkdir(exist_ok=True, parents=True)


def test_ml_runner_setup(training_runner_no_setup: TrainingRunner) -> None:
    """Check that all the necessary methods get called during setup"""
    assert not training_runner_no_setup._has_setup_run
    with patch.object(training_runner_no_setup, "container", spec=LightningContainer) as mock_container:
        # Without that, it would try to create a local run object for logging and fail there.
        mock_container.log_from_vm = False
        with patch.object(
            training_runner_no_setup, "checkpoint_handler", spec=CheckpointHandler
        ) as mock_checkpoint_handler:
            with patch("health_ml.runner_base.seed_everything") as mock_seed:
                with patch("health_ml.runner_base.seed_monai_if_available") as mock_seed_monai:
                    training_runner_no_setup.setup()
                    mock_container.get_effective_random_seed.assert_called()
                    mock_seed.assert_called_once()
                    mock_seed_monai.assert_called_once()
                    mock_container.create_filesystem.assert_called_once()
                    mock_checkpoint_handler.download_recovery_checkpoints_or_weights.assert_called_once()
                    mock_container.setup.assert_called_once()
                    mock_container.create_lightning_module_and_store.assert_called_once()
                    assert training_runner_no_setup._has_setup_run


def test_setup_azureml(training_runner: TrainingRunner) -> None:
    """Test that setup_azureml causes set_tags to get called when running in Hyperdrive"""
    with patch("health_ml.runner_base.RUN_CONTEXT") as mock_run_context:
        training_runner.setup_azureml()
        # Tests always run outside of a Hyperdrive run. In those cases, no tags should be set on
        # the current run.
        mock_run_context.set_tags.assert_not_called()

        with patch("health_ml.runner_base.PARENT_RUN_CONTEXT") as mock_parent_run_context:
            # Mock the presence of a parent run, and tags that are present there
            tag_name = "tag"
            tag_value = "dummy_tag"
            mock_parent_run_context.get_tags.return_value = {tag_name: tag_value}
            training_runner.setup_azureml()
            # The function should read out tags from the parent run, and set them on the current run
            mock_parent_run_context.get_tags.assert_called_once_with()
            mock_run_context.set_tags.assert_called_once()
            call_args = mock_run_context.set_tags.call_args[0][0]
            assert tag_name in call_args
            assert call_args[tag_name] == tag_value
            assert EFFECTIVE_RANDOM_SEED_KEY_NAME in call_args


def test_get_multiple_trainloader_mode(training_runner: TrainingRunner) -> None:
    training_runner.init_training()
    multiple_trainloader_mode = training_runner.get_multiple_trainloader_mode()
    assert multiple_trainloader_mode == "max_size_cycle", "train_loader_cycle_mode is available now, "
    "`get_multiple_trainloader_mode` workaround can be safely removed."


def _test_init_training(run_inference_only: bool, training_runner: TrainingRunner, caplog: LogCaptureFixture) -> None:
    """Test that training is initialized correctly"""
    training_runner.container.run_inference_only = run_inference_only
    training_runner.setup()
    assert not training_runner.checkpoint_handler.has_continued_training
    assert training_runner.trainer is None
    assert training_runner.storing_logger is None

    with patch("health_ml.training_runner.create_lightning_trainer") as mock_create_trainer:
        with patch.object(training_runner.container, "get_data_module") as mock_get_data_module:
            with patch("health_ml.training_runner.write_experiment_summary_file") as mock_write_experiment_summary_file:
                with patch.object(
                    training_runner.checkpoint_handler, "get_recovery_or_checkpoint_path_train"
                ) as mock_get_recovery_or_checkpoint_path_train:
                    with patch("health_ml.training_runner.seed_everything") as mock_seed:
                        mock_create_trainer.return_value = MagicMock(), MagicMock()
                        mock_get_recovery_or_checkpoint_path_train.return_value = "dummy_path"

                        training_runner.init_training()

                        # Make sure write_experiment_summary_file is only called on rank 0
                        if is_global_rank_zero():
                            mock_write_experiment_summary_file.assert_called()
                        else:
                            mock_write_experiment_summary_file.assert_not_called()

                        # Make sure seed is set correctly with workers=True
                        mock_seed.assert_called_once()
                        assert mock_seed.call_args[0][0] == training_runner.container.get_effective_random_seed()
                        assert mock_seed.call_args[1]["workers"]

                        mock_get_data_module.assert_called_once()
                        assert training_runner.data_module is not None

                        if not run_inference_only:
                            mock_get_recovery_or_checkpoint_path_train.assert_called_once()
                            # Validate that the trainer is created correctly
                            assert mock_create_trainer.call_args[1]["resume_from_checkpoint"] == "dummy_path"
                            assert training_runner.storing_logger is not None
                            assert training_runner.trainer is not None
                            assert "Environment variables:" in caplog.messages[-1]
                        else:
                            assert training_runner.trainer is None
                            assert training_runner.storing_logger is None
                            mock_get_recovery_or_checkpoint_path_train.assert_not_called()


@pytest.mark.parametrize("run_inference_only", [True, False])
def test_init_training_cpu(
    run_inference_only: bool, training_runner: TrainingRunner, caplog: LogCaptureFixture
) -> None:
    """Test that training is initialized correctly"""
    training_runner.container.max_num_gpus = 0
    _test_init_training(run_inference_only, training_runner, caplog)


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("run_inference_only", [True, False])
def test_init_training_gpu(
    run_inference_only: bool, training_runner: TrainingRunner, caplog: LogCaptureFixture
) -> None:
    """Test that training is initialized correctly in DDP mode"""
    _test_init_training(run_inference_only, training_runner, caplog)


def test_run_training() -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    runner = TrainingRunner(experiment_config=experiment_config, container=container)

    with patch.object(container, "get_data_module") as mock_get_data_module:
        with patch("health_ml.training_runner.create_lightning_trainer") as mock_create_trainer:
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
    runner = TrainingRunner(experiment_config=experiment_config, container=container)

    with patch.object(container, "get_data_module"):
        with patch("health_ml.training_runner.create_lightning_trainer", return_value=(MagicMock(), MagicMock())):
            runner.setup()
            runner.init_training()
            runner.run_training()

        with patch.object(runner.checkpoint_handler, "additional_training_done") as mock_additional_training_done:
            with patch.object(runner, "after_ddp_cleanup") as mock_after_ddp_cleanup:
                with patch("health_ml.training_runner.cleanup_checkpoints") as mock_cleanup_ckpt:
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
    run_inference_only: bool,
    run_extra_val_epoch: bool,
    max_num_gpus_inf: int,
    training_runner_hello_world_with_checkpoint: TrainingRunner,
    caplog: LogCaptureFixture,
) -> None:
    training_runner_hello_world_with_checkpoint.container.run_inference_only = run_inference_only
    training_runner_hello_world_with_checkpoint.container.run_extra_val_epoch = run_extra_val_epoch
    training_runner_hello_world_with_checkpoint.container.max_num_gpus_inference = max_num_gpus_inf
    assert (
        training_runner_hello_world_with_checkpoint.container.max_num_gpus == -1
    )  # This is the default value of max_num_gpus
    training_runner_hello_world_with_checkpoint.init_training()
    if run_inference_only:
        expected_mlflow_run_id = None
    else:
        assert training_runner_hello_world_with_checkpoint.trainer is not None
        create_mlflow_trash_folder(training_runner_hello_world_with_checkpoint)
        expected_mlflow_run_id = training_runner_hello_world_with_checkpoint.trainer.loggers[1].run_id  # type: ignore
    if not run_inference_only:
        training_runner_hello_world_with_checkpoint.checkpoint_handler.additional_training_done()
    with patch("health_ml.runner_base.create_lightning_trainer") as mock_create_trainer:
        with patch.object(
            training_runner_hello_world_with_checkpoint.container, "get_checkpoint_to_test"
        ) as mock_get_checkpoint_to_test:
            with patch.object(
                training_runner_hello_world_with_checkpoint.container, "get_data_module"
            ) as mock_get_data_module:
                mock_checkpoint = MagicMock(is_file=MagicMock(return_value=True))
                mock_get_checkpoint_to_test.return_value = mock_checkpoint
                mock_trainer = MagicMock()
                mock_create_trainer.return_value = mock_trainer, MagicMock()
                mock_get_data_module.return_value = "dummy_data_module"

                assert training_runner_hello_world_with_checkpoint.inference_checkpoint is None
                assert not training_runner_hello_world_with_checkpoint.container.model._on_extra_val_epoch

                training_runner_hello_world_with_checkpoint.init_inference()
                if run_extra_val_epoch:
                    assert (
                        caplog.messages[-3]
                        == "Preparing to run an extra validation epoch to evaluate the model on the validation set."
                    )
                assert caplog.messages[-2] == "Preparing runner for inference."

                expected_ckpt = str(training_runner_hello_world_with_checkpoint.checkpoint_handler.trained_weights_path)
                expected_ckpt = expected_ckpt if run_inference_only else str(mock_checkpoint)
                assert training_runner_hello_world_with_checkpoint.inference_checkpoint == expected_ckpt

                assert hasattr(
                    training_runner_hello_world_with_checkpoint.container.model, "on_run_extra_validation_epoch"
                )
                assert (
                    training_runner_hello_world_with_checkpoint.container.model._on_extra_val_epoch
                    == run_extra_val_epoch
                )

                mock_create_trainer.assert_called_once()
                assert training_runner_hello_world_with_checkpoint.trainer == mock_trainer
                assert training_runner_hello_world_with_checkpoint.container.max_num_gpus == max_num_gpus_inf
                assert (
                    mock_create_trainer.call_args[1]["container"]
                    == training_runner_hello_world_with_checkpoint.container
                )
                assert mock_create_trainer.call_args[1]["num_nodes"] == 1
                assert mock_create_trainer.call_args[1]["mlflow_run_for_logging"] == expected_mlflow_run_id
                mock_get_data_module.assert_called_once()
                assert training_runner_hello_world_with_checkpoint.data_module == "dummy_data_module"


@pytest.mark.parametrize("run_inference_only", [True, False])
@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_run_validation(
    run_extra_val_epoch: bool,
    run_inference_only: bool,
    training_runner_hello_world_with_checkpoint: TrainingRunner,
    caplog: LogCaptureFixture,
) -> None:
    training_runner_hello_world_with_checkpoint.container.run_extra_val_epoch = run_extra_val_epoch
    training_runner_hello_world_with_checkpoint.container.run_inference_only = run_inference_only
    training_runner_hello_world_with_checkpoint.init_training()
    mock_datamodule = MagicMock()
    create_mlflow_trash_folder(training_runner_hello_world_with_checkpoint)
    with patch("health_ml.runner_base.create_lightning_trainer") as mock_create_trainer:
        with patch.object(
            training_runner_hello_world_with_checkpoint.container, "get_data_module", return_value=mock_datamodule
        ):
            mock_trainer = MagicMock()
            mock_create_trainer.return_value = mock_trainer, MagicMock()
            training_runner_hello_world_with_checkpoint.init_inference()
            assert training_runner_hello_world_with_checkpoint.trainer == mock_trainer
            mock_trainer.validate = Mock()
            training_runner_hello_world_with_checkpoint.run_validation()
            if run_extra_val_epoch or run_inference_only:
                mock_trainer.validate.assert_called_once()
                assert (
                    mock_trainer.validate.call_args[1]["ckpt_path"]
                    == training_runner_hello_world_with_checkpoint.inference_checkpoint
                )
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
        runner = TrainingRunner(experiment_config=experiment_config, container=container)
        runner.setup()
        runner.checkpoint_handler.additional_training_done()
        runner.container.outputs_folder.mkdir(parents=True, exist_ok=True)
        with patch.object(container, "get_data_module"):
            with patch("health_ml.runner_base.create_lightning_trainer", return_value=(MagicMock(), MagicMock())):
                with patch.object(runner.container, "get_checkpoint_to_test") as mock_get_checkpoint_to_test:
                    mock_get_checkpoint_to_test.return_value = MagicMock(is_file=MagicMock(return_value=True))
                    runner.init_inference()
                    runner.run_validation()
                    assert "Hook `on_run_extra_validation_epoch` is not implemented" in caplog.messages[-3]


def test_run_inference(training_runner_hello_world: TrainingRunner, regression_datadir: Path) -> None:
    """
    Test that run_inference gets called as expected.
    """
    training_runner_hello_world.container.max_num_gpus = 0

    def _expected_files_exist() -> bool:
        output_dir = training_runner_hello_world.container.outputs_folder
        if not output_dir.is_dir():
            return False
        expected_files = [TEST_MSE_FILE, TEST_MAE_FILE]
        return all([(output_dir / p).exists() for p in expected_files])

    expected_ckpt_path = training_runner_hello_world.container.outputs_folder / "checkpoints" / "last.ckpt"
    assert not expected_ckpt_path.exists()
    # update the container to look for test data at this location
    training_runner_hello_world.container.local_dataset_dir = regression_datadir
    assert not _expected_files_exist()

    actual_train_ckpt_path = training_runner_hello_world.checkpoint_handler.get_recovery_or_checkpoint_path_train()
    assert actual_train_ckpt_path is None

    training_runner_hello_world.run()

    actual_train_ckpt_path = training_runner_hello_world.checkpoint_handler.get_recovery_or_checkpoint_path_train()
    assert actual_train_ckpt_path == expected_ckpt_path

    actual_test_ckpt_path = training_runner_hello_world.checkpoint_handler.get_checkpoint_to_test()
    assert actual_test_ckpt_path == expected_ckpt_path
    assert actual_test_ckpt_path.is_file()
    # After training, the outputs directory should now exist and contain the 2 error files
    assert _expected_files_exist()


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
@pytest.mark.parametrize("run_inference_only", [True, False])
def test_run(run_inference_only: bool, run_extra_val_epoch: bool, training_runner_hello_world: TrainingRunner) -> None:
    """Test that model runner gets called"""
    training_runner_hello_world.container.run_inference_only = run_inference_only
    training_runner_hello_world.container.run_extra_val_epoch = run_extra_val_epoch
    training_runner_hello_world.setup()
    assert not training_runner_hello_world.checkpoint_handler.has_continued_training

    with patch("health_ml.runner_base.create_lightning_trainer", return_value=(MagicMock(), MagicMock())):
        with patch.multiple(
            training_runner_hello_world,
            checkpoint_handler=mock.DEFAULT,
            run_training=mock.DEFAULT,
            run_validation=mock.DEFAULT,
            run_inference=mock.DEFAULT,
            end_training=mock.DEFAULT,
        ) as mocks:
            training_runner_hello_world.run()
            assert training_runner_hello_world.container.has_custom_test_step()
            assert training_runner_hello_world._has_setup_run
            assert mocks["end_training"] != run_inference_only
            assert mocks["run_training"].called != run_inference_only
            mocks["run_validation"].assert_called_once()
            mocks["run_inference"].assert_called_once()


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_run_inference_only(
    run_extra_val_epoch: bool, training_runner_hello_world_with_checkpoint: TrainingRunner
) -> None:
    """Test inference only mode. Validation should be run regardless of run_extra_val_epoch status."""
    training_runner_hello_world_with_checkpoint.container.run_inference_only = True
    training_runner_hello_world_with_checkpoint.container.run_extra_val_epoch = run_extra_val_epoch
    assert training_runner_hello_world_with_checkpoint.checkpoint_handler.trained_weights_path
    mock_datamodule = MagicMock()
    with patch("health_ml.runner_base.create_lightning_trainer") as mock_create_trainer:
        with patch.object(
            training_runner_hello_world_with_checkpoint.container, "get_data_module", return_value=mock_datamodule
        ):
            with patch.multiple(
                training_runner_hello_world_with_checkpoint,
                run_training=mock.DEFAULT,
            ) as mocks:
                mock_trainer = MagicMock()
                mock_create_trainer.return_value = mock_trainer, MagicMock()
                training_runner_hello_world_with_checkpoint.run()
                mock_create_trainer.assert_called_once()
                mocks["run_training"].assert_not_called()

                mock_trainer.validate.assert_called_once()
                assert (
                    mock_trainer.validate.call_args[1]["ckpt_path"]
                    == training_runner_hello_world_with_checkpoint.inference_checkpoint
                )
                assert mock_trainer.validate.call_args[1]["datamodule"] == mock_datamodule
                mock_trainer.test.assert_called_once()
                assert (
                    mock_trainer.test.call_args[1]["ckpt_path"]
                    == training_runner_hello_world_with_checkpoint.inference_checkpoint
                )
                assert mock_trainer.test.call_args[1]["datamodule"] == mock_datamodule


@pytest.mark.parametrize("run_extra_val_epoch", [True, False])
def test_resume_training_from_run_id(
    run_extra_val_epoch: bool, training_runner_hello_world_with_checkpoint: TrainingRunner
) -> None:
    training_runner_hello_world_with_checkpoint.container.run_extra_val_epoch = run_extra_val_epoch
    training_runner_hello_world_with_checkpoint.container.max_num_gpus = 0
    training_runner_hello_world_with_checkpoint.container.max_epochs += 10
    assert training_runner_hello_world_with_checkpoint.checkpoint_handler.trained_weights_path
    mock_trainer = MagicMock()
    with patch("health_ml.runner_base.create_lightning_trainer", return_value=(mock_trainer, MagicMock())):
        with patch.object(
            training_runner_hello_world_with_checkpoint.container, "get_checkpoint_to_test"
        ) as mock_get_checkpoint_to_test:
            with patch.object(training_runner_hello_world_with_checkpoint, "run_inference") as mock_run_inference:
                with patch("health_ml.training_runner.cleanup_checkpoints") as mock_cleanup_ckpt:
                    mock_get_checkpoint_to_test.return_value = MagicMock(is_file=MagicMock(return_value=True))
                    training_runner_hello_world_with_checkpoint.run()
                    mock_get_checkpoint_to_test.assert_called_once()
                    mock_cleanup_ckpt.assert_called_once()
                    assert mock_trainer.validate.called == run_extra_val_epoch
                    mock_run_inference.assert_called_once()


def test_model_weights_when_resume_training() -> None:
    experiment_config = ExperimentConfig(model="HelloWorld")
    container = HelloWorld()
    container.max_num_gpus = 0
    container.src_checkpoint = CheckpointParser(str(hello_world_checkpoint))
    container.resume_training = True
    runner = TrainingRunner(experiment_config=experiment_config, container=container)
    runner.setup()
    assert runner.checkpoint_handler.trained_weights_path.is_file()  # type: ignore
    with patch("health_ml.training_runner.create_lightning_trainer") as mock_create_trainer:
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
    runner = TrainingRunner(experiment_config=experiment_config, container=container)
    # When logging to AzureML, need to provide the unit test AML workspace.
    # When not logging to AzureML, no workspace (and no authentication) should be needed.
    if log_from_vm:
        with patch("health_azure.utils.get_workspace", return_value=DEFAULT_WORKSPACE.workspace):
            runner.run_and_cleanup()
    else:
        runner.run_and_cleanup()
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


def test_inference_only_metrics_correctness(
    training_runner_hello_world_with_checkpoint: TrainingRunner, regression_datadir: Path
) -> None:
    training_runner_hello_world_with_checkpoint.container.run_inference_only = True
    training_runner_hello_world_with_checkpoint.container.local_dataset_dir = regression_datadir
    training_runner_hello_world_with_checkpoint.run()
    with open(training_runner_hello_world_with_checkpoint.container.outputs_folder / TEST_MSE_FILE) as f:
        mse = float(f.readlines()[0])
    assert isclose(mse, 0.010806690901517868, abs_tol=1e-3)
    with open(training_runner_hello_world_with_checkpoint.container.outputs_folder / TEST_MAE_FILE) as f:
        mae = float(f.readlines()[0])
    assert isclose(mae, 0.08260975033044815, abs_tol=1e-3)


def test_training_and_inference_metrics_match(tmp_path: Path) -> None:
    """
    Test if the metrics on the test set when training a model match with those when running inference on the
    test set.
    """
    experiment_config = ExperimentConfig(model="HelloWorld")
    container1 = HelloWorld()
    container1.set_output_to(tmp_path / "training")
    training_runner = TrainingRunner(experiment_config=experiment_config, container=container1)
    training_runner.run_and_cleanup()
    mse_training = float((training_runner.container.outputs_folder / TEST_MSE_FILE).read_text()[0])
    checkpoint = training_runner.container.checkpoint_folder / "last.ckpt"
    assert checkpoint.is_file()
    container2 = HelloWorld()
    container2.set_output_to(tmp_path / "eval")
    container2.src_checkpoint = CheckpointParser(str(checkpoint))
    container2.run_inference_only = True
    eval_runner = TrainingRunner(experiment_config=experiment_config, container=container2)
    eval_runner.run_and_cleanup()
    mse_eval = float((eval_runner.container.outputs_folder / TEST_MSE_FILE).read_text()[0])
    assert isclose(mse_training, mse_eval, rel_tol=1e-10)
