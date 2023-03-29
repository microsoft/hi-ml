from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from health_ml import MLRunner
from health_ml.configs.hello_world import TEST_MAE_FILE, TEST_MSE_FILE, HelloRegression, HelloWorld


def test_eval_runner_no_checkpoint(mock_runner: MLRunner) -> None:
    """Test of the evaluation mode fails if no checkpoint source is provided"""
    arguments = ["", f"--model=HelloWorld", "--mode=eval"]
    with pytest.raises(ValueError, match="To use model evaluation, you need to provide a checkpoint to use"):
        with patch.object(sys, "argv", arguments):
            mock_runner.run()


def test_eval_runner_end_to_end(mock_runner: MLRunner, tmp_path: Path) -> None:
    """Test the end-to-end integration of the EvalRunner class into the overall Runner"""
    container = HelloWorld()
    empty_model = container.create_model()
    # Write all model parameters to a checkpoint file
    torch.save
    checkpoint = empty_model
    model_name = "HelloWorld"
    arguments = ["", f"--model={model_name}", "--mode=eval"]
    with patch("health_ml.ml_runner.MLRunner.run_and_cleanup") as mock_training_run:
        with patch.object(sys, "argv", arguments):
            _ = mock_runner.run()
        # The training runner should not be invoked
        mock_training_run.assert_not_called()
        # The eval runner should have been invoked. The test step writes two files with metrics, check that
        # they exist
        output_folder = mock_runner.container.outputs_folder
        for file_name in [TEST_MSE_FILE, TEST_MAE_FILE]:
            assert (output_folder / file_name).exists()
