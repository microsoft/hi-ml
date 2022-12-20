#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from _pytest.capture import SysCapture
from _pytest.logging import LogCaptureFixture
from azureml._restclient.constants import RunStatus
from azureml.core import Run

from health_azure import RUN_CONTEXT
from health_ml.utils import AzureMLLogger, AzureMLProgressBar, log_learning_rate, log_on_epoch
from health_ml.utils.logging import _preprocess_hyperparams
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


def test_log_on_epoch() -> None:
    """
    Tests if the helper function to log metrics per epoch works.
    """
    module = mock.MagicMock()
    module.trainer = None
    with pytest.raises(AssertionError) as ex1:
        log_on_epoch(module, metrics={"foo": 1})
    assert "No trainer is set" in str(ex1)
    module.trainer = mock.MagicMock()
    module.trainer.world_size = 1
    with pytest.raises(ValueError) as ex2:
        log_on_epoch(module, name="foo")
    assert "'name' and 'value' must be provided" in str(ex2)
    with pytest.raises(ValueError) as ex3:
        log_on_epoch(module, value=1.0)
    assert "'name' and 'value' must be provided" in str(ex3)
    foo_value = 1
    metrics = {"bar": torch.tensor(2.0)}
    module.device = 'cpu'
    module.log_dict = mock.MagicMock()
    log_on_epoch(module, name="foo", value=foo_value, metrics=metrics)
    # Test if all metrics that are not tensors are converted to floating point tensors
    actual_args = module.log_dict.call_args
    actual_metrics = actual_args[0][0]
    for metric_name in ["foo", "bar"]:
        assert metric_name in actual_metrics, f"Metric missing: {metric_name}"
        assert torch.is_tensor(actual_metrics[metric_name]), f"Metric {metric_name}: not a tensor"
        assert actual_metrics[metric_name].dtype == torch.float, f"Metric {metric_name}: should be float tensor"
    assert actual_metrics["foo"].item() == float(foo_value)
    # Default arguments for the call to module.log
    assert actual_args[1] == {'on_epoch': True,
                              'on_step': False,
                              'reduce_fx': torch.mean,
                              'sync_dist': False}, "Failed for world_size==1"
    # Test if sync_dist is computed correctly from world size: world size is now 2, so sync_dist should be True
    module.trainer.world_size = 2
    log_on_epoch(module, metrics=metrics)
    assert module.log_dict.call_args[1] == {'on_epoch': True,
                                            'on_step': False,
                                            'reduce_fx': torch.mean,
                                            'sync_dist': True}, "Failed for world_size==2"
    # Test if overrides for sync_dist and the other aggregation args are passed correctly
    module.trainer.world_size = 2
    log_on_epoch(module, metrics=metrics, reduce_fx="reduce", sync_dist=False)  # type: ignore
    assert module.log_dict.call_args[1] == {'on_epoch': True,
                                            'on_step': False,
                                            'sync_dist': False,
                                            'reduce_fx': "reduce"}, "Failed for sync_dist==True"
    module.trainer.world_size = 1
    log_on_epoch(module, metrics=metrics, reduce_fx="reduce", sync_dist=True)  # type: ignore
    assert module.log_dict.call_args[1] == {'on_epoch': True,
                                            'on_step': False,
                                            'sync_dist': True,
                                            'reduce_fx': "reduce"}, "Failed for sync_dist==True"


def test_log_learning_rate_singleton() -> None:
    """
    Test the method that logs learning rates, when there is a single LR scheduler.
    """
    module = mock.MagicMock()
    module.lr_schedulers = mock.MagicMock(return_value=None)
    with pytest.raises(ValueError) as ex:
        log_learning_rate(module)
        assert "can only be used during training" in str(ex)
    scheduler = mock.MagicMock()
    lr = 1.234
    scheduler.get_last_lr = mock.MagicMock(return_value=[lr])
    module.lr_schedulers = mock.MagicMock(return_value=scheduler)
    module.trainer = mock.MagicMock(world_size=1)
    with mock.patch("health_ml.utils.logging.log_on_epoch") as mock_log_on_epoch:
        log_learning_rate(module)
        assert mock_log_on_epoch.call_args[0] == (module,)
        assert mock_log_on_epoch.call_args[1] == {'metrics': {'learning_rate': lr}}


def test_log_learning_rate_multiple() -> None:
    """
    Test the method that logs learning rates, when there are multiple schedulers with non-scalar return values.
    """
    scheduler1 = mock.MagicMock()
    lr1 = [1]
    scheduler1.get_last_lr = mock.MagicMock(return_value=lr1)
    scheduler2 = mock.MagicMock()
    lr2 = [2, 3]
    scheduler2.get_last_lr = mock.MagicMock(return_value=lr2)
    module = mock.MagicMock()
    module.lr_schedulers = mock.MagicMock(return_value=[scheduler1, scheduler2])
    with mock.patch("health_ml.utils.logging.log_on_epoch") as mock_log_on_epoch:
        log_learning_rate(module, name="foo")
        assert mock_log_on_epoch.call_args[0] == (module,)
        assert mock_log_on_epoch.call_args[1] == {'metrics': {'foo/0/0': lr1[0],
                                                              'foo/1/0': lr2[0],
                                                              'foo/1/1': lr2[1]}}


def create_mock_logger() -> AzureMLLogger:
    """
    Create an AzureMLLogger that has a run field set to a MagicMock.
    """
    run_mock = MagicMock()
    with mock.patch("health_ml.utils.logging.create_aml_run_object", return_value=run_mock):
        return AzureMLLogger(enable_logging_outside_azure_ml=True)


def test_azureml_logger() -> None:
    """
    Tests logging to an AzureML run via PytorchLightning
    """
    logger = create_mock_logger()
    # On all build agents, this should not be detected as an AzureML run.
    assert not logger.is_running_in_azure_ml
    assert not logger.has_user_provided_run
    logger.log_metrics({"foo": 1.0})
    assert logger.run is not None
    logger.run.log.assert_called_once_with("foo", 1.0, step=None)

    # All the following methods of LightningLoggerBase are not implemented
    assert logger.name() == ""
    assert logger.version() == 0
    assert logger.experiment() is None

    # Finalizing should call the "Complete" method of the run
    logger.finalize(status="foo")
    logger.run.complete.assert_called_once()


def test_azureml_log_hyperparameters1() -> None:
    """
    Test logging of hyperparameters
    """
    logger = create_mock_logger()
    assert logger.run is not None
    # No logging should happen with empty params
    logger.log_hyperparams(None)  # type: ignore
    assert logger.run.log.call_count == 0
    logger.log_hyperparams({})
    assert logger.run.log.call_count == 0
    logger.log_hyperparams(Namespace())
    assert logger.run.log.call_count == 0
    # Logging of hyperparameters that are plain dictionaries
    fake_params = {"foo": 1.0}
    logger.log_hyperparams(fake_params)
    # Dictionary should be logged as name/value pairs, one value per row
    logger.run.log_table.assert_called_once_with("hyperparams", {'name': ['foo'], 'value': ["1.0"]})


def test_azureml_log_hyperparameters2() -> None:
    """
    Logging of hyperparameters that are Namespace objects from the arg parser
    """
    logger = create_mock_logger()
    assert logger.run is not None

    class Dummy:
        def __str__(self) -> str:
            return "dummy"

    fake_namespace = Namespace(foo="bar", complex_object=Dummy())
    logger.log_hyperparams(fake_namespace)
    # Complex objects are converted to str
    expected_dict: Dict[str, Any] = {'name': ['foo', 'complex_object'], 'value': ['bar', 'dummy']}
    logger.run.log_table.assert_called_once_with("hyperparams", expected_dict)


def test_azureml_log_hyperparameters3() -> None:
    """
    Logging of hyperparameters that are nested dictionaries. They should first be flattened, than each complex
    object to str
    """
    logger = create_mock_logger()
    assert logger.run is not None
    fake_namespace = Namespace(foo={"bar": 1, "baz": {"level3": Namespace(a="17")}})
    logger.log_hyperparams(fake_namespace)
    expected_dict = {"name": ["foo/bar", "foo/baz/level3/a"], "value": ["1", "17"]}
    logger.run.log_table.assert_called_once_with("hyperparams", expected_dict)


def test_azureml_logger_many_hyperparameters(tmpdir: Path) -> None:
    """
    Test if large number of hyperparameters are logged correctly.
    Earlier versions of the code had a bug that only allowed a maximum of 15 hyperparams to be logged.
    """
    many_hyperparams: Dict[str, Any] = {f"param{i}": i for i in range(0, 20)}
    many_hyperparams["A long list"] = ["foo", 1.0, "abc"]
    expected_metrics = {key: str(value) for key, value in many_hyperparams.items()}
    logger: Optional[AzureMLLogger] = None
    try:
        logger = AzureMLLogger(enable_logging_outside_azure_ml=True, workspace=DEFAULT_WORKSPACE.workspace)
        assert logger.run is not None
        logger.log_hyperparams(many_hyperparams)
        logger.run.flush()
        time.sleep(1)
        metrics = logger.run.get_metrics(name=AzureMLLogger.HYPERPARAMS_NAME)
        print(f"metrics = {metrics}")
        actual = metrics[AzureMLLogger.HYPERPARAMS_NAME]
        assert actual["name"] == list(expected_metrics.keys())
        assert actual["value"] == list(expected_metrics.values())
    finally:
        if logger:
            logger.finalize("done")


def test_azureml_logger_hyperparams_processing() -> None:
    """
    Test flattening of hyperparameters: Lists were not handled correctly in previous versions.
    """
    hyperparams = {"A long list": ["foo", 1.0, "abc"],
                   "foo": 1.0}
    actual = _preprocess_hyperparams(hyperparams)
    assert actual == {"A long list": "['foo', 1.0, 'abc']", "foo": "1.0"}


def test_azureml_logger_step() -> None:
    """
    Test if the AzureML logger correctly handles epoch-level and step metrics
    """
    logger = create_mock_logger()
    assert logger.run is not None
    logger.log_metrics(metrics={"foo": 1.0, "epoch": 123}, step=78)
    assert logger.run.log.call_count == 2
    assert logger.run.log.call_args_list[0][0] == ("foo", 1.0)
    assert logger.run.log.call_args_list[0][1] == {"step": None}, "For epoch-level metrics, no step should be provided"
    assert logger.run.log.call_args_list[1][0] == ("epoch", 123)
    assert logger.run.log.call_args_list[1][1] == {"step": None}, "For epoch-level metrics, no step should be provided"
    logger.run.reset_mock()  # type: ignore
    logger.log_metrics(metrics={"foo": 1.0}, step=78)
    logger.run.log.assert_called_once_with("foo", 1.0, step=78)


def test_azureml_logger_init1() -> None:
    """
    Test the logic to choose the run, inside of the constructor of AzureMLLogger.
    """
    # When running in AzureML, the RUN_CONTEXT should be used
    with mock.patch("health_ml.utils.logging.is_running_in_azure_ml", return_value=True):
        with mock.patch("health_ml.utils.logging.RUN_CONTEXT", "foo"):
            logger = AzureMLLogger(enable_logging_outside_azure_ml=True)
            assert logger.is_running_in_azure_ml
            assert logger.enable_logging_outside_azure_ml
            assert not logger.has_user_provided_run
            assert logger.run == "foo"
            # We should be able to call finalize without any effect (logger.run == "foo", which has no
            # "Complete" method). When running in AzureML, the logger should not
            # modify the run in any way, and in particular not complete it.
            logger.finalize("nothing")


def test_azureml_logger_init2() -> None:
    """
    Test the logic to choose the run, inside of the constructor of AzureMLLogger.
    """
    # When disabling offline logging, the logger should be a no-op, and not log anything
    logger = AzureMLLogger(enable_logging_outside_azure_ml=False)
    assert logger.run is None
    logger.log_metrics({"foo": 1.0})
    logger.finalize(status="nothing")


def test_azureml_logger_actual_run() -> None:
    """
    When running outside of AzureML, a new run should be created.
    """
    logger = AzureMLLogger(enable_logging_outside_azure_ml=True,
                           workspace=DEFAULT_WORKSPACE.workspace,
                           run_name="test_azureml_logger_actual_run")
    assert not logger.is_running_in_azure_ml
    assert logger.run is not None
    assert logger.run != RUN_CONTEXT
    assert isinstance(logger.run, Run)
    assert logger.run.experiment.name == "azureml_logger"
    assert not logger.has_user_provided_run
    expected_metrics = {"foo": 1.0, "bar": 2.0}
    logger.log_metrics(expected_metrics)
    logger.run.flush()
    actual_metrics = logger.run.get_metrics()
    assert actual_metrics == expected_metrics
    assert logger.run.status != RunStatus.COMPLETED
    logger.finalize("nothing")

    # The AzureML run has been complete now, insert a mock to check if
    logger.run = MagicMock()
    logger.finalize("nothing")
    logger.run.complete.assert_called_once_with()


def test_azureml_logger_init4() -> None:
    """
    Test the logic to choose the run, inside of the constructor of AzureMLLogger.
    """
    # Check that all arguments are respected
    run_mock = MagicMock()
    with mock.patch("health_ml.utils.logging.create_aml_run_object", return_value=run_mock) as mock_create:
        logger = AzureMLLogger(enable_logging_outside_azure_ml=True,
                               experiment_name="exp",
                               run_name="run",
                               snapshot_directory="snapshot",
                               workspace="workspace",  # type: ignore
                               workspace_config_path=Path("config_path"))
        assert not logger.has_user_provided_run
        assert logger.run == run_mock
        mock_create.assert_called_once_with(experiment_name="exp",
                                            run_name="run",
                                            snapshot_directory="snapshot",
                                            workspace="workspace",
                                            workspace_config_path=Path("config_path"))
    # The run created in the constructor is under the control of the AzureML logger, and should be completed.
    # Check that the finalize method calls the run's complete method, but not the run's flush method.
    run_mock.flush = MagicMock()
    run_mock.complete = MagicMock()
    logger.finalize(status="nothing")
    run_mock.flush.assert_not_called()
    run_mock.complete.assert_called_once()


def test_azureml_logger_finalize() -> None:
    """Test if the finalize method correctly updates the run status. It should only operate on runs that are
    outside of AzureML."""
    run_mock = MagicMock()
    logger = AzureMLLogger(enable_logging_outside_azure_ml=True, run=run_mock)
    assert logger.run is not None
    assert logger.has_user_provided_run
    run_mock.flush = MagicMock()
    run_mock.complete = MagicMock()
    # When providing a run explicitly, the finalize method should not call the run's complete method. Completing
    # the run is the responsibility of the user.
    logger.finalize(status="nothing")
    run_mock.flush.assert_called_once()
    run_mock.complete.assert_not_called()


def test_progress_bar_enable() -> None:
    """
    Test the logic for disabling the progress bar.
    """
    bar = AzureMLProgressBar(refresh_rate=0)
    assert not bar.is_enabled
    assert bar.is_disabled
    bar = AzureMLProgressBar(refresh_rate=1)
    assert bar.is_enabled
    bar.disable()
    assert not bar.is_enabled
    bar.enable()
    assert bar.is_enabled


def test_progress_bar(capsys: SysCapture) -> None:
    bar = AzureMLProgressBar(refresh_rate=1)
    mock_module = mock.MagicMock(global_step=34)
    mock_trainer = mock.MagicMock(current_epoch=12,
                                  lightning_module=mock_module,
                                  num_training_batches=10,
                                  num_val_batches=5,
                                  emable_validation=False,
                                  num_test_batches=[20],
                                  num_predict_batches=[30])
    bar.setup(mock_trainer, mock_module)
    assert bar.trainer == mock_trainer

    def latest_message() -> str:
        return capsys.readouterr().out.splitlines()[-1]  # type: ignore

    # Messages in training
    bar.on_train_epoch_start(mock_trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_TRAIN
    with patch("health_ml.utils.AzureMLProgressBar.train_batch_idx", PropertyMock(return_value=1)):
        bar.on_train_batch_end(None, None, None, None, None)  # type: ignore
        latest = latest_message()
        assert "Training epoch 12 (step 34)" in latest
        assert "1/10 ( 10%) completed" in latest

    # Messages in validation
    with patch("health_ml.utils.AzureMLProgressBar.total_val_batches", PropertyMock(return_value=5)):
        bar.on_validation_start(mock_trainer, None)  # type: ignore
        assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_VAL
        with patch("health_ml.utils.AzureMLProgressBar.val_batch_idx", PropertyMock(return_value=1)):
            bar.on_validation_batch_end(None, None, None, None, None, None)  # type: ignore
            latest = latest_message()
            assert "Validation epoch 12: " in latest
            assert "1/5 ( 20%) completed" in latest

    # Messages in testing
    bar.on_test_epoch_start(mock_trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_TEST
    test_count = 2
    with patch("health_ml.utils.AzureMLProgressBar.test_batch_idx", PropertyMock(return_value=test_count)):
        bar.on_test_batch_end(None, None, None, None, None, None)  # type: ignore
        latest = latest_message()
        assert "Testing:" in latest
        assert f"{test_count}/20 ( 10%)" in latest

    # Messages in prediction
    bar.on_predict_epoch_start(mock_trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_PREDICT
    predict_count = 3
    with patch("health_ml.utils.AzureMLProgressBar.predict_batch_idx", PropertyMock(return_value=predict_count)):
        bar.on_predict_batch_end(None, None, None, None, None, None)  # type: ignore
        latest = latest_message()
        assert "Prediction:" in latest
        assert f"{predict_count}/30 ( 10%)" in latest
        assert "since epoch start" in latest

    # Test behaviour when a batch count is infinity
    with patch("health_ml.utils.AzureMLProgressBar.predict_batch_idx", PropertyMock(return_value=predict_count + 1)):
        bar.total_num_batches = math.inf  # type: ignore
        bar.on_predict_batch_end(None, None, None, None, None, None)  # type: ignore
        latest = latest_message()
        assert "4 batches completed" in latest
        assert "since epoch start" in latest


def test_progress_bar_to_logging(caplog: LogCaptureFixture) -> None:
    """
    Check that the progress bar correctly writes to logging
    """
    to_logging = AzureMLProgressBar(write_to_logging_info=True)
    message = "A random message"
    with caplog.at_level(logging.INFO):
        to_logging._print(message)
        assert message in caplog.text


@pytest.mark.parametrize("print_timestamp", [True, False])
def test_progress_bar_to_stdout(capsys: SysCapture, print_timestamp: bool) -> None:
    """
    Check that the progress bar correctly writes to stdout, and that timestamps are generated if requested.
    """
    message = "A random message"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    to_stdout = AzureMLProgressBar(write_to_logging_info=False, print_timestamp=print_timestamp)
    to_stdout._print(message)
    stdout: str = capsys.readouterr().out  # type: ignore
    print(f"Output: {stdout}")
    assert message in stdout
    assert stdout.startswith(today) == print_timestamp
