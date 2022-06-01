#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Callable, Dict
from unittest import mock

import torch
from _pytest.logging import LogCaptureFixture

from health_ml.utils.diagnostics import BatchTimeCallback, EpochTimers


def test_epoch_timers(caplog: LogCaptureFixture) -> None:
    """
    Test the class that measures batch and epoch times.
    """
    caplog.set_level(logging.INFO)
    batch_index = 123
    epoch = 24
    timer = EpochTimers(max_batch_load_time_seconds=100)
    assert timer.total_load_time == 0.0

    # First batch should always generate a message
    timer.batch_start(batch_index=0, epoch=epoch, message_prefix="prefix")
    assert timer.total_load_time > 0.0
    message = caplog.messages[-1]
    assert "prefix: Loaded the first minibatch of data in" in message
    old_num_batches = timer.num_batches
    old_batch_start_time = timer.batch_start_time
    timer.batch_end()
    assert timer.num_batches == old_num_batches + 1
    assert timer.batch_start_time > old_batch_start_time

    # Second minibatch should only generate a message if above load time threshold. Set threshold very high
    old_num_messages = len(caplog.messages)
    old_total_load_time = timer.total_load_time
    timer.max_batch_load_time_seconds = 10.0
    assert timer.num_load_time_exceeded == 0
    timer.batch_start(batch_index=batch_index, epoch=epoch, message_prefix="prefix")
    # This should be updated in any case
    assert timer.total_load_time > old_total_load_time
    # But this batch should not be recognized as having gone over the threshold
    assert timer.num_load_time_exceeded == 0
    assert len(timer.load_time_warning_epochs) == 0
    assert len(caplog.messages) == old_num_messages
    assert timer.num_load_time_warnings == 0

    # Third minibatch considered as above threshold: set threshold to 0 for that
    old_total_load_time = timer.total_load_time
    timer.max_batch_load_time_seconds = 0.0
    timer.batch_start(batch_index=batch_index, epoch=epoch, message_prefix="prefix")
    # This should be updated in any case
    assert timer.total_load_time > old_total_load_time
    # Batch should not be recognized as having gone over the threshold
    assert timer.num_load_time_exceeded == 1
    assert epoch in timer.load_time_warning_epochs
    message = caplog.messages[-1]
    assert f"prefix: Loading minibatch {batch_index} took" in message
    assert f"This message will be printed at most {timer.max_load_time_warnings} times"
    assert timer.num_load_time_warnings > 0

    # Epoch end time should be stored
    assert timer.total_epoch_time == 0.0
    old_epoch_end_time = timer.epoch_end_time
    timer.epoch_end()
    assert timer.epoch_end_time > old_epoch_end_time
    assert timer.total_epoch_time > 0.0

    # Test the resetting logic
    timer.epoch_start()
    assert timer.total_load_time == 0.0
    assert timer.num_load_time_warnings == 0
    # The object should keep track of all epochs in which warnings were printed
    assert len(timer.load_time_warning_epochs) > 0

    # Test if the warnings disappear after the max number of warnings
    assert timer.should_warn_in_this_epoch
    timer.max_load_time_epochs = 0
    timer.load_time_warning_epochs = {1}
    assert not timer.should_warn_in_this_epoch


def test_batch_time_callback(caplog: LogCaptureFixture) -> None:
    """
    Test the callback that measures data loading times.
    """
    caplog.set_level(logging.INFO)
    callback = BatchTimeCallback()
    epoch = 1234
    # This dictionary stores all metrics that are written via module.log
    logged_metrics = {}

    def mock_log(name: str, value: float, reduce_fx: Callable, **kwargs: Dict) -> None:
        logged_metrics[name] = (value, reduce_fx)

    mock_module = mock.MagicMock(current_epoch=epoch, log=mock_log)
    callback.on_fit_start(trainer=None, pl_module=mock_module)  # type: ignore
    assert callback.module == mock_module

    # Upon epoch start, the timers should be reset. We can check that by looking at epoch_start_time
    assert callback.train_timers.epoch_start_time == 0.0
    callback.on_train_epoch_start(None, None)  # type: ignore
    assert callback.train_timers.epoch_start_time > 0.0
    assert callback.val_timers.epoch_start_time == 0.0
    old_train_epoch_end_time = callback.train_timers.epoch_end_time
    callback.on_validation_epoch_start(None, None)  # type: ignore
    assert callback.val_timers.epoch_start_time > 0.0
    # When calling epoch_start for validation, training epoch should be ended
    assert callback.train_timers.epoch_end_time > old_train_epoch_end_time

    # Run 1 training batch
    callback.on_train_batch_start(None, None, None, batch_idx=0, dataloader_idx=0)  # type: ignore
    callback.on_train_batch_end(None, None, None, None, batch_idx=0, dataloader_idx=0)  # type: ignore
    assert len(logged_metrics) == 2

    # Upon batch end, we should see metrics being logged. Batch level timings should be logged both as averages and max
    def check_batch_metrics(train_or_val: str) -> None:
        for suffix in [" avg", " max"]:
            name = f"timing/{train_or_val}/batch_time [sec]" + suffix
            assert name in logged_metrics
            assert logged_metrics[name][1] == max if suffix == " max" else torch.mean

    check_batch_metrics("train")
    assert caplog.messages[-1].startswith(f"Epoch {epoch} training: Loaded the first")
    # Run 2 validation batches
    for batch_idx in range(2):
        callback.on_validation_batch_start(None, None, None, batch_idx=batch_idx, dataloader_idx=0)  # type: ignore
        callback.on_validation_batch_end(None, None, None, None, batch_idx=batch_idx, dataloader_idx=0)  # type: ignore
    assert caplog.messages[-1].startswith(f"Epoch {epoch} validation: Loaded the first")
    assert callback.train_timers.num_batches == 1
    assert callback.val_timers.num_batches == 2
    check_batch_metrics("val")

    # Check that the metrics are written at the end of the validation epoch.
    # Hack the timers to trigger the warning message for validation only
    callback.val_timers.num_load_time_exceeded = 1
    callback.val_timers.total_extra_load_time = 100.00
    callback.val_timers.max_batch_load_time_seconds = 2.0
    assert callback.val_timers.should_warn_in_this_epoch
    old_val_epoch_end_time = callback.train_timers.epoch_end_time
    callback.on_validation_epoch_end(None, None)  # type: ignore
    assert callback.val_timers.epoch_end_time > old_val_epoch_end_time
    assert len(logged_metrics) > 0

    assert f"Epoch {epoch} training took " in caplog.messages[-4]
    assert f"Epoch {epoch} validation took " in caplog.messages[-3]
    assert "The dataloaders were not fast enough" in caplog.messages[-2]
    assert "in less than 2.00sec" in caplog.messages[-2]
    assert "1 out of 2 batches exceeded the load time threshold" in caplog.messages[-1]
    assert "Total loading time for the slow batches was 100.00sec" in caplog.messages[-1]

    for prefix in [BatchTimeCallback.TRAIN_PREFIX, BatchTimeCallback.VAL_PREFIX]:
        for metric in [BatchTimeCallback.EPOCH_TIME, BatchTimeCallback.EXCESS_LOADING_TIME]:
            assert f"timing/{prefix}{metric}" in logged_metrics
