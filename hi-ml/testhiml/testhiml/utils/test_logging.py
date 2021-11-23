#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

import pytest
import torch
from _pytest.capture import SysCapture
from _pytest.logging import LogCaptureFixture
from azureml.core import Run
from pytorch_lightning import Trainer

from health_azure import create_aml_run_object
from health_ml.utils import AzureMLLogger, AzureMLProgressBar, log_learning_rate, log_on_epoch
from testazure.util import DEFAULT_WORKSPACE


def create_unittest_run_object(snapshot_directory: Optional[Path] = None) -> Run:
    return create_aml_run_object(experiment_name="himl-tests",
                                 workspace=DEFAULT_WORKSPACE.workspace,
                                 snapshot_directory=snapshot_directory or ".")


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
                              'sync_dist': False,
                              'sync_dist_op': 'mean'}, "Failed for world_size==1"
    # Test if sync_dist is computed correctly from world size: world size is now 2, so sync_dist should be True
    module.trainer.world_size = 2
    log_on_epoch(module, metrics=metrics)
    assert module.log_dict.call_args[1] == {'on_epoch': True,
                                            'on_step': False,
                                            'reduce_fx': torch.mean,
                                            'sync_dist': True,
                                            'sync_dist_op': 'mean'}, "Failed for world_size==2"
    # Test if overrides for sync_dist and the other aggregation args are passed correctly
    module.trainer.world_size = 2
    log_on_epoch(module, metrics=metrics, reduce_fx="reduce", sync_dist=False, sync_dist_op="nothing")  # type: ignore
    assert module.log_dict.call_args[1] == {'on_epoch': True,
                                            'on_step': False,
                                            'sync_dist': False,
                                            'reduce_fx': "reduce",
                                            'sync_dist_op': "nothing"}, "Failed for sync_dist==True"
    module.trainer.world_size = 1
    log_on_epoch(module, metrics=metrics, reduce_fx="reduce", sync_dist=True, sync_dist_op="nothing")  # type: ignore
    assert module.log_dict.call_args[1] == {'on_epoch': True,
                                            'on_step': False,
                                            'sync_dist': True,
                                            'reduce_fx': "reduce",
                                            'sync_dist_op': "nothing"}, "Failed for sync_dist==True"


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


def test_azureml_logger() -> None:
    """
    Tests logging to an AzureML run via PytorchLightning
    """
    logger = AzureMLLogger()
    # On all build agents, this should not be detected as an AzureML run.
    assert not logger.is_running_in_azure_ml
    # No logging should happen when outside AzureML
    with mock.patch("health_azure.utils.RUN_CONTEXT.log") as log_mock:
        logger.log_metrics({"foo": 1.0})
        assert log_mock.call_count == 0
    # Pretend to be running in AzureML
    logger.is_running_in_azure_ml = True
    with mock.patch("health_azure.utils.RUN_CONTEXT.log") as log_mock:
        logger.log_metrics({"foo": 1.0})
        assert log_mock.call_count == 1
        assert log_mock.call_args[0] == ("foo", 1.0), "Should be called with the unrolled dictionary of metrics"
    # All the following methods of LightningLoggerBase are not implemented
    assert logger.name() == ""
    assert logger.version() == 0
    assert logger.experiment() is None


def test_azureml_logger_hyperparams() -> None:
    """
    Tests logging of hyperparameters to an AzureML
    """
    logger = AzureMLLogger()
    # On all build agents, this should not be detected as an AzureML run.
    assert not logger.is_running_in_azure_ml
    # No logging should happen when outside AzureML
    with mock.patch("health_azure.utils.RUN_CONTEXT.log_table") as log_mock:
        logger.log_hyperparams({"foo": 1.0})
        assert log_mock.call_count == 0
    # Pretend to be running in AzureML
    logger.is_running_in_azure_ml = True
    # No logging should happen with empty params
    with mock.patch("health_azure.utils.RUN_CONTEXT.log_table") as log_mock:
        logger.log_hyperparams(None)  # type: ignore
        assert log_mock.call_count == 0
        logger.log_hyperparams({})
        assert log_mock.call_count == 0
        logger.log_hyperparams(Namespace())
        assert log_mock.call_count == 0
    # Logging of hyperparameters that are plain dictionaries
    with mock.patch("health_azure.utils.RUN_CONTEXT.log_table") as log_mock:
        fake_params = {"foo": 1.0}
        logger.log_hyperparams(fake_params)
        assert log_mock.call_count == 1
        # Dictionary should be logged as name/value pairs, one value per row
        assert log_mock.call_args[0] == ("hyperparams", {'name': ['foo'], 'value': [1.0]})


def test_azureml_logger_hyperparams2() -> None:
    """
    Tests logging of complex hyperparameters to AzureML
    """

    class Dummy:
        def __str__(self) -> str:
            return "dummy"

    logger = AzureMLLogger()
    # Pretend to be running in AzureML
    logger.is_running_in_azure_ml = True

    # Logging of hyperparameters that are Namespace objects from the arg parser
    with mock.patch("health_azure.utils.RUN_CONTEXT.log_table") as log_mock:
        fake_namespace = Namespace(foo="bar", complex_object=Dummy())
        logger.log_hyperparams(fake_namespace)
        assert log_mock.call_count == 1
        # Complex objects are converted to str
        expected_dict: Dict[str, Any] = {'name': ['foo', 'complex_object'], 'value': ['bar', 'dummy']}
        assert log_mock.call_args[0] == ("hyperparams", expected_dict)

    # Logging of hyperparameters that are nested dictionaries. They should first be flattened, than each complex
    # object to str
    with mock.patch("health_azure.utils.RUN_CONTEXT.log_table") as log_mock:
        fake_namespace = Namespace(foo={"bar": 1, "baz": {"level3": Namespace(a="17")}})
        logger.log_hyperparams(fake_namespace)
        assert log_mock.call_count == 1
        expected_dict = {"name": ["foo/bar", "foo/baz/level3/a"], "value": [1, "17"]}
        assert log_mock.call_args[0] == ("hyperparams", expected_dict)


def test_azureml_logger_many_hyperparameters(tmpdir: Path) -> None:
    """
    Test if large number of hyperparameters are logged correctly. Earlier versions of the code had a bug that only
    allowed a maximum of 15 hyperparams to be logged.
    """
    # Change to a newly created random folder to minimize the time taken for snapshot creation
    many_hyperparams = {f"param{i}": i for i in range(0, 100)}
    try:
        run = create_unittest_run_object(tmpdir)
        with mock.patch("health_ml.utils.logging.RUN_CONTEXT", run):
            logger = AzureMLLogger()
            logger.is_running_in_azure_ml = True
            logger.log_hyperparams(many_hyperparams)
            run.flush()
            metrics = run.get_metrics(name=AzureMLLogger.HYPERPARAMS_NAME)
            actual = metrics[AzureMLLogger.HYPERPARAMS_NAME]
            assert actual["name"] == list(many_hyperparams.keys())
            assert actual["value"] == list(many_hyperparams.values())
    finally:
        run.complete()


def test_azureml_logger_step() -> None:
    """
    Test if the AzureML logger correctly handles epoch-level and step metrics
    """
    logger = AzureMLLogger()
    # Pretend to be running in AzureML
    logger.is_running_in_azure_ml = True
    with mock.patch("health_azure.utils.RUN_CONTEXT.log") as log_mock:
        logger.log_metrics(metrics={"foo": 1.0, "epoch": 123}, step=78)
        assert log_mock.call_count == 2
        assert log_mock.call_args_list[0][0] == ("foo", 1.0)
        assert log_mock.call_args_list[0][1] == {"step": None}, "For epoch-level metrics, no step should be provided"
        assert log_mock.call_args_list[1][0] == ("epoch", 123)
        assert log_mock.call_args_list[1][1] == {"step": None}, "For epoch-level metrics, no step should be provided"
    with mock.patch("health_azure.utils.RUN_CONTEXT.log") as log_mock:
        logger.log_metrics(metrics={"foo": 1.0}, step=78)
        assert log_mock.call_count == 1
        assert log_mock.call_args[0] == ("foo", 1.0), "Should be called with the unrolled dictionary of metrics"
        assert log_mock.call_args[1] == {"step": 78}, "For step-level metrics, the step argument should be provided"


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
    mock_trainer = mock.MagicMock(current_epoch=12,
                                  lightning_module=mock.MagicMock(global_step=34),
                                  num_training_batches=10,
                                  emable_validation=False,
                                  num_test_batches=[20],
                                  num_predict_batches=[30])
    bar.on_init_end(mock_trainer)  # type: ignore
    assert bar.trainer == mock_trainer

    def latest_message() -> str:
        return capsys.readouterr().out.splitlines()[-1]  # type: ignore

    # Messages in training
    trainer = Trainer()
    bar.on_train_epoch_start(trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_TRAIN
    assert bar.train_batch_idx == 0
    assert bar.val_batch_idx == 0
    assert bar.test_batch_idx == 0
    assert bar.predict_batch_idx == 0
    bar.on_train_batch_end(None, None, None, None, None)  # type: ignore
    assert bar.train_batch_idx == 1
    latest = latest_message()
    assert "Training epoch 12 (step 34)" in latest
    assert "1/10 ( 10%) completed" in latest
    # When starting the next training epoch, the counters should be reset
    bar.on_train_epoch_start(trainer, None)  # type: ignore
    assert bar.train_batch_idx == 0
    # Messages in validation
    bar.on_validation_start(trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_VAL
    assert bar.total_num_batches == 0
    assert bar.val_batch_idx == 0
    # Number of validation batches is difficult to fake, tweak the field where it is stored in the progress bar
    bar.total_num_batches = 5
    bar.on_validation_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.val_batch_idx == 1
    latest = latest_message()
    assert "Validation epoch 12: " in latest
    assert "1/5 ( 20%) completed" in latest
    # Messages in testing
    bar.on_test_epoch_start(trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_TEST
    test_count = 2
    for _ in range(test_count):
        bar.on_test_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.test_batch_idx == test_count
    latest = latest_message()
    assert "Testing:" in latest
    assert f"{test_count}/20 ( 10%)" in latest
    # Messages in prediction
    bar.on_predict_epoch_start(trainer, None)  # type: ignore
    assert bar.stage == AzureMLProgressBar.PROGRESS_STAGE_PREDICT
    predict_count = 3
    for _ in range(predict_count):
        bar.on_predict_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.predict_batch_idx == predict_count
    latest = latest_message()
    assert "Prediction:" in latest
    assert f"{predict_count}/30 ( 10%)" in latest
    assert "since epoch start" in latest
    # Test behaviour when a batch count is infinity
    bar.total_num_batches = math.inf  # type: ignore
    bar.on_predict_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.predict_batch_idx == 4
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
