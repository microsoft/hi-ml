from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from health_ml import MLRunner
from health_ml.configs.hello_world import (
    TEST_MAE_FILE,
    TEST_MSE_FILE,
    HelloWorld,
    HelloWorldDataModule,
)
from health_ml.eval_runner import EvalRunner
from health_ml.experiment_config import ExperimentConfig
from health_ml.runner import Runner
from health_ml.utils.checkpoint_utils import CheckpointParser
from testhiml.test_run_ml import ml_runner_with_container


@pytest.fixture(scope="function")
def hello_world_checkpoint(ml_runner_with_container: MLRunner, tmp_path: Path) -> None:
    container = ml_runner_with_container.container
    container.set_output_to(tmp_path)
    container.max_epochs = 5
    ml_runner_with_container.run()
    # Read out the trained checkpoint and use the runner in eval mode
    return container.get_checkpoint_to_test()


def test_eval_runner_no_checkpoint(mock_runner: Runner) -> None:
    """Test of the evaluation mode fails if no checkpoint source is provided"""
    arguments = ["", f"--model=HelloWorld", "--mode=eval"]
    with pytest.raises(ValueError, match="To use model evaluation, you need to provide a checkpoint to use"):
        with patch.object(sys, "argv", arguments):
            mock_runner.run()


def test_eval_runner_end_to_end(mock_runner: Runner, hello_world_checkpoint: Path) -> None:
    """Test the end-to-end integration of the EvalRunner class into the overall Runner"""
    arguments = ["", f"--model=HelloWorld", "--mode=eval", f"--src_checkpoint={hello_world_checkpoint}"]
    with patch("health_ml.ml_runner.MLRunner.run_and_cleanup") as mock_training_run:
        with patch.object(sys, "argv", arguments):
            mock_runner.run()
        # The training runner should not be invoked
        mock_training_run.assert_not_called()
        # The eval runner should have been invoked. The test step writes two files with metrics, check that
        # they exist
        output_folder = mock_runner.lightning_container.outputs_folder
        for file_name in [TEST_MSE_FILE, TEST_MAE_FILE]:
            assert (output_folder / file_name).exists()


def test_eval_runner_methods_called(hello_world_checkpoint: Path, tmp_path: Path) -> None:
    """Test if the eval runner uses the right data module from the HelloWorld model"""
    container = HelloWorld()
    container.src_checkpoint = CheckpointParser(hello_world_checkpoint)
    eval_runner = EvalRunner(
        container=container, experiment_config=ExperimentConfig(mode="eval"), project_root=tmp_path
    )
    with patch("health_ml.configs.hello_world.HelloWorld.get_eval_data_module") as mock_get_data_module:
        mock_get_data_module.return_value = HelloWorldDataModule(crossval_count=1, seed=1)
        eval_runner.run_and_cleanup()
        mock_get_data_module.assert_called_once_with()
