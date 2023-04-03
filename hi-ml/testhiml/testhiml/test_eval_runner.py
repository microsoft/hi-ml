from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from health_ml import TrainingRunner
from health_ml.configs.hello_world import (
    TEST_MAE_FILE,
    TEST_MSE_FILE,
    HelloWorld,
    HelloWorldDataModule,
)
from health_ml.eval_runner import EvalRunner
from health_ml.experiment_config import ExperimentConfig, RunnerMode
from health_ml.runner import Runner
from health_ml.utils.checkpoint_utils import CheckpointParser
from testhiml.test_training_runner import training_runner_hello_world
from testhiml.utils.fixed_paths_for_tests import full_test_data_path


hello_world_checkpoint = full_test_data_path(suffix="hello_world_checkpoint.ckpt")


def test_eval_runner_no_checkpoint(mock_runner: Runner) -> None:
    """Test of the evaluation mode fails if no checkpoint source is provided"""
    arguments = ["", f"--model=HelloWorld", f"--mode={RunnerMode.EVAL_FULL.value}"]
    with pytest.raises(ValueError, match="To use model evaluation, you need to provide a checkpoint to use"):
        with patch.object(sys, "argv", arguments):
            mock_runner.run()


def test_eval_runner_end_to_end(mock_runner: Runner) -> None:
    """Test the end-to-end integration of the EvalRunner class into the overall Runner"""
    arguments = [
        "",
        f"--model=HelloWorld",
        f"--mode={RunnerMode.EVAL_FULL.value}",
        f"--src_checkpoint={hello_world_checkpoint}",
    ]
    with patch("health_ml.training_runner.TrainingRunner.run_and_cleanup") as mock_training_run:
        with patch.object(sys, "argv", arguments):
            mock_runner.run()
        # The training runner should not be invoked
        mock_training_run.assert_not_called()
        # The eval runner should have been invoked. The test step writes two files with metrics, check that
        # they exist
        output_folder = mock_runner.lightning_container.outputs_folder
        for file_name in [TEST_MSE_FILE, TEST_MAE_FILE]:
            assert (output_folder / file_name).exists()


def test_eval_runner_methods_called(tmp_path: Path) -> None:
    """Test if the eval runner uses the right data module from the HelloWorld model"""
    container = HelloWorld()
    container.src_checkpoint = CheckpointParser(str(hello_world_checkpoint))
    eval_runner = EvalRunner(
        container=container, experiment_config=ExperimentConfig(mode=RunnerMode.EVAL_FULL), project_root=tmp_path
    )
    with patch("health_ml.configs.hello_world.HelloWorld.get_eval_data_module") as mock_get_data_module:
        mock_get_data_module.return_value = HelloWorldDataModule(crossval_count=1, seed=1)
        eval_runner.run_and_cleanup()
        mock_get_data_module.assert_called_once_with()


def test_eval_runner_no_extra_validation_epoch_called(tmp_path: Path) -> None:
    """
    Ensure that the eval runner does not invoke the hook the extra validation epoch that is used by the training runner.
    """
    container = HelloWorld()
    container.run_extra_val_epoch = True
    container.src_checkpoint = CheckpointParser(str(hello_world_checkpoint))
    eval_runner = EvalRunner(
        container=container, experiment_config=ExperimentConfig(mode=RunnerMode.EVAL_FULL), project_root=tmp_path
    )
    with patch("health_ml.configs.hello_world.HelloRegression.on_run_extra_validation_epoch") as mock_hook:
        eval_runner.run_and_cleanup()
        mock_hook.assert_not_called()
