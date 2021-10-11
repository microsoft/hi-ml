#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from unittest import mock

import pytest
import torch

from health_ml.utils import log_on_epoch, log_learning_rate, AzureMLLogger

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
    module.trainer = mock.MagicMock(world_size = 1)
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
    assert logger.is_running_in_azure_ml == False
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
    assert logger.experiment() is None
    logger.log_hyperparams(params=None)
