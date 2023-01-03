#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from typing import Any, Optional, Set

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


class EpochTimers:
    """
    Contains all information necessary to compute the IO metrics: Epoch times, batch times, loading times.
    """

    def __init__(self,
                 max_batch_load_time_seconds: float = 0.5,
                 max_load_time_warnings: int = 3,
                 max_load_time_epochs: int = 5
                 ) -> None:
        """
        :param max_batch_load_time_seconds: The maximum expected loading time for a minibatch (given in seconds).
            If the loading time exceeds this threshold, a warning is printed.
        :param max_load_time_warnings: The maximum number of warnings that will be printed per epoch.
        :param max_load_time_epochs: The maximum number of epochs where warnings about the loading time are printed.
        """
        self.max_batch_load_time_seconds = max_batch_load_time_seconds
        self.max_load_time_warnings = max_load_time_warnings
        self.max_load_time_epochs = max_load_time_epochs
        self.load_time_warning_epochs: Set[int] = set()
        self.epoch_start_time: float = 0.0
        self.epoch_end_time: float = 0.0
        self.batch_start_time: float = 0.0
        self.num_load_time_warnings: int = 0
        self.num_load_time_exceeded: int = 0
        self.total_extra_load_time: float = 0.0
        self.total_load_time: float = 0.0
        self.num_batches: int = 0

    def epoch_start(self) -> None:
        """
        Resets all timers to the current time, and all counters to 0. The set of epochs for which warnings about
        load time were produced will not be reset.
        """
        current_time = time.time()
        self.epoch_start_time = current_time
        self.epoch_end_time = current_time
        self.batch_start_time = current_time
        self.num_load_time_warnings = 0
        self.num_load_time_exceeded = 0
        self.total_extra_load_time = 0.0
        self.total_load_time = 0.0
        self.num_batches = 0

    def epoch_end(self) -> None:
        """
        Stores the present time in the epoch_end_time field of the object.
        """
        self.epoch_end_time = time.time()

    @property
    def total_epoch_time(self) -> float:
        """
        Gets the time in seconds between epoch start and epoch end.
        """
        return self.epoch_end_time - self.epoch_start_time

    @property
    def should_warn_in_this_epoch(self) -> bool:
        """
        Returns True if warnings about loading time should be printed in the present epoch. Returns False if
        this warning has been printed already in more than self.max_load_time_epochs epochs.
        """
        return len(self.load_time_warning_epochs) <= self.max_load_time_epochs

    def batch_start(self, batch_index: int, epoch: int, message_prefix: str) -> float:
        """
        Called when a minibatch of data has been loaded. This computes the time it took to load the minibatch
        (computed between now and the end of the previous minibatch)
        and adds it to the internal bookkeeping. If the minibatch loading time exceeds a threshold, then warnings
        are printed (unless too many warnings have been printed already)

        :param message_prefix: A prefix string that is added to all diagnostic output.
        :param epoch: The index of the current epoch.
        :param batch_index: The index of the current minibatch.
        :return: The time it took to load the minibatch, in seconds.
        """
        item_finish_time = time.time()
        item_load_time = item_finish_time - self.batch_start_time
        self.total_load_time += item_load_time
        # Having slow minibatch loading is OK in the very first batch of the every epoch, where processes
        # are spawned. Later, the load time should be zero.
        if batch_index == 0:
            logging.info(f"{message_prefix}: Loaded the first minibatch of data in {item_load_time:0.2f} sec.")
        elif item_load_time > self.max_batch_load_time_seconds:
            self.load_time_warning_epochs.add(epoch)
            self.num_load_time_exceeded += 1
            self.total_extra_load_time += item_load_time
            if self.num_load_time_warnings < self.max_load_time_warnings and self.should_warn_in_this_epoch:
                logging.warning(f"{message_prefix}: Loading minibatch {batch_index} took {item_load_time:0.2f} sec. "
                                "This can mean that there are not enough data loader worker processes, or that there "
                                "is a performance problem in loading. This warning will be printed at most "
                                f"{self.max_load_time_warnings} times in at most {self.max_load_time_epochs} epochs.")
                self.num_load_time_warnings += 1
        return item_load_time

    def batch_end(self) -> float:
        """
        Called after a minibatch has been processed (training or validation step completed). Returns the time it took
        to process the current batch (including loading).

        :return: The time it took to process the current batch, in seconds.
        """
        current_time = time.time()
        elapsed = current_time - self.batch_start_time
        self.batch_start_time = current_time
        self.num_batches += 1
        return elapsed


class BatchTimeCallback(Callback):
    """
    This callback provides tools to measure batch loading time and other diagnostic information.
    It prints alerts to the console or to `logging` if the batch loading time is over a threshold for several epochs.
    Metrics for loading time, as well as epoch time, and maximum and average batch processing time are logged to
    the loggers that are set up on the module.
    In distributed training, all logging to the console and to the Lightning loggers will only happen on global rank 0.

    The loading time for a minibatch is estimated by the difference between the start time of a minibatch and the
    end time of the previous minibatch. It will consequently also include other operations that happen between the
    end of a batch and the start of the next one. For example, computationally expensive callbacks could also
    drive up this time.

    Usage example:
        >>> from health_ml.utils import BatchTimeCallback
        >>> from pytorch_lightning import Trainer
        >>> batchtime = BatchTimeCallback(max_batch_load_time_seconds=0.5)
        >>> trainer = Trainer(callbacks=[batchtime])
    """

    EPOCH_TIME = "epoch_time [sec]"
    """The name that is used to log the execution time per epoch """
    BATCH_TIME = "batch_time [sec]"
    """The name that is used to log the execution time per batch."""
    EXCESS_LOADING_TIME = "batch_loading_over_threshold [sec]"
    """The name that is used to log the time spent loading all the batches that exceeding the loading time threshold."""
    METRICS_PREFIX = "timing/"
    """The prefix for all metrics collected by this callback."""
    TRAIN_PREFIX = "train/"
    """The prefix for all metrics collected during training."""
    VAL_PREFIX = "val/"
    """The prefix for all metrics collected during validation."""

    def __init__(self,
                 max_batch_load_time_seconds: float = 0.5,
                 max_load_time_warnings: int = 3,
                 max_load_time_epochs: int = 5
                 ) -> None:
        """
        :param max_batch_load_time_seconds: The maximum expected loading time for a minibatch (given in seconds).
            If the loading time exceeds this threshold, a warning is printed. The maximum number of such warnings is
            controlled by the other arguments.
        :param max_load_time_warnings: The maximum number of warnings about increased loading time that will be printed
            per epoch. For example, if max_load_time_warnings=3, at most 3 of these warnings will be printed within an
            epoch. The 4th minibatch with loading time over the threshold would not generate any warning anymore.
            If set to 0, no warnings are printed at all.
        :param max_load_time_epochs: The maximum number of epochs where warnings about the loading time are printed.
            For example, if max_load_time_epochs=2, and at least 1 batch with increased loading time is observed in
            epochs 0 and 3, no further warnings about increased loading time would be printed from epoch 4 onwards.
        """
        # Timers for monitoring data loading time
        self.train_timers = EpochTimers(max_batch_load_time_seconds=max_batch_load_time_seconds,
                                        max_load_time_warnings=max_load_time_warnings,
                                        max_load_time_epochs=max_load_time_epochs)
        self.val_timers = EpochTimers(max_batch_load_time_seconds=max_batch_load_time_seconds,
                                      max_load_time_warnings=max_load_time_warnings,
                                      max_load_time_epochs=max_load_time_epochs)
        self.module: Optional[LightningModule] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        This is called at the start of training. It stores the model that is being trained, because it will be used
        later to log values.
        """
        self.module = pl_module

    @rank_zero_only
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_timers.epoch_start()

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_timers.epoch_start()
        # In Lightning, the validation epoch is running "inside" the training. If we get here, it means that training
        # is done for this epoch, even though the on_training_epoch hook has not yet been called.
        self.train_timers.epoch_end()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        This is a hook called at the end of a training or validation epoch. In here, we can still write
        metrics to a logger.
        """
        # In validation epochs, mark that it has been completed. Training epochs are marked completed already
        # at the start of the validation epoch.
        self.val_timers.epoch_end()
        # Write all IO stats here, so that the order on the console is Train start, train end, val start, val end.
        self.write_and_log_epoch_time(is_training=True)
        self.write_and_log_epoch_time(is_training=False)

    def on_train_batch_start(self,  # type: ignore
                             trainer: Trainer,
                             pl_module: LightningModule,
                             batch: Any,
                             batch_idx: int,
                             dataloader_idx: int,
                             ) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=True)

    def on_validation_batch_start(self,
                                  trainer: Trainer,
                                  pl_module: LightningModule,
                                  batch: Any,
                                  batch_idx: int,
                                  dataloader_idx: int,
                                  ) -> None:
        self.batch_start(batch_idx=batch_idx, is_training=False)

    def on_train_batch_end(self,  # type: ignore
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs: Any,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int,
                           ) -> None:
        self.batch_end(is_training=True)

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Any,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int,
                                ) -> None:
        self.batch_end(is_training=False)

    @rank_zero_only
    def batch_start(self, batch_idx: int, is_training: bool) -> None:
        """
        Shared code to keep track of minibatch loading times. This is only done on global rank zero.

        :param batch_idx: The index of the current minibatch.
        :param is_training: If true, this has been called from `on_train_batch_start`, otherwise it has been called from
            `on_validation_batch_start`.
        """
        timers = self.get_timers(is_training=is_training)
        assert self.module is not None
        epoch = self.module.current_epoch
        message_prefix = f"Epoch {epoch} {'training' if is_training else 'validation'}"
        timers.batch_start(batch_index=batch_idx, epoch=epoch, message_prefix=message_prefix)

    @rank_zero_only
    def batch_end(self, is_training: bool) -> None:
        """
        Shared code to keep track of minibatch loading times. This is only done on global rank zero.

        :param is_training: If true, this has been called from `on_train_batch_end`, otherwise it has been called from
            `on_validation_batch_end`.
        """
        timers = self.get_timers(is_training=is_training)
        batch_time = timers.batch_end()
        self.log_metric(self.BATCH_TIME + " avg",
                        value=batch_time,
                        is_training=is_training)
        self.log_metric(self.BATCH_TIME + " max",
                        value=batch_time,
                        is_training=is_training,
                        reduce_max=True)

    @rank_zero_only
    def write_and_log_epoch_time(self, is_training: bool) -> None:
        """
        Reads the IO timers for either the training or validation epoch, writes them to the console, and logs the
        time per epoch.

        :param is_training: If True, show and log the data for the training epoch. If False, use the data for the
            validation epoch.
        """
        timers = self.get_timers(is_training=is_training)
        epoch_time_seconds = timers.total_epoch_time
        status = "training" if is_training else "validation"
        assert self.module is not None
        logging.info(f"Epoch {self.module.current_epoch} {status} took {epoch_time_seconds:0.2f}sec, of which waiting "
                     f"for data took {timers.total_load_time:0.2f} sec total.")
        if timers.num_load_time_exceeded > 0 and timers.should_warn_in_this_epoch:
            logging.warning("The dataloaders were not fast enough to always supply the next batch in less than "
                            f"{timers.max_batch_load_time_seconds:0.2f}sec.")
            logging.warning(
                f"In this epoch, {timers.num_load_time_exceeded} out of {timers.num_batches} batches exceeded the load "
                f"time threshold. Total loading time for the slow batches was {timers.total_extra_load_time:0.2f}sec.")
        # This metric is only written at rank zero, and hence must not be synchronized across workers. If attempted,
        # training will get stuck.
        self.log_metric(self.EPOCH_TIME,
                        value=epoch_time_seconds,
                        is_training=is_training)
        self.log_metric(self.EXCESS_LOADING_TIME,
                        value=timers.total_extra_load_time,
                        is_training=is_training)

    @rank_zero_only
    def log_metric(self, name_suffix: str, value: float, is_training: bool, reduce_max: bool = False) -> None:
        """
        Write a metric given as a name/value pair to the currently trained module. The full name of the metric is
        composed of a fixed prefix "timing/", followed by either "train/" or "val/", and then the given suffix.

        :param name_suffix: The suffix for the logged metric name.
        :param value: The value to log.
        :param is_training: If True, use "train/" in the metric name, otherwise "val/"
        :param reduce_max: If True, use torch.max as the aggregation function for the logged values. If False, use
            torch.mean
        """
        # Metrics are only written at global rank 0, and hence must not be synchronized. Trying to synchronize will
        # block training.
        prefix = self.TRAIN_PREFIX if is_training else self.VAL_PREFIX
        assert self.module is not None
        self.module.log(name=self.METRICS_PREFIX + prefix + name_suffix, value=value,  # type: ignore
                        on_step=False, on_epoch=True, sync_dist=False,
                        reduce_fx=max if reduce_max else torch.mean)

    def get_timers(self, is_training: bool) -> EpochTimers:
        """
        Gets the object that holds all metrics and timers, for either the validation or the training epoch.
        """
        return self.train_timers if is_training else self.val_timers


class TrainingDiagnoticsCallback(Callback):
    """Callback that logs information about the training process."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_train_epoch_start on global rank {pl_module.global_rank}")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_train_epoch_end on global rank {pl_module.global_rank}")

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_validation_epoch_start on global rank {pl_module.global_rank}")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_validation_epoch_end on global rank {pl_module.global_rank}")

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_test_epoch_start on global rank {pl_module.global_rank}")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_test_epoch_end on global rank {pl_module.global_rank}")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached on_train_start on global rank {pl_module.global_rank}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached from on_train_end on global rank {pl_module.global_rank}")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached from on_validation_start on global rank {pl_module.global_rank}")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached from on_validation_end on global rank {pl_module.global_rank}")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached from on_test_start on global rank {pl_module.global_rank}")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Reached from on_test_end on global rank {pl_module.global_rank}")

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        logging.info(
            f"Reached from on_train_batch_start on global rank {pl_module.global_rank} for batch_idx {batch_idx}"
        )

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
        unused: int = 0
    ) -> None:
        logging.info(
            f"Reached from on_train_batch_end on global rank {pl_module.global_rank} for batch_idx {batch_idx}"
        )

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        logging.info(
            f"Reached from on_validation_batch_start on global rank {pl_module.global_rank} for batch_idx {batch_idx}"
        )

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int,
        dataloader_idx: int
    ) -> None:
        logging.info(
            f"Reached from on_validation_batch_end on global rank {pl_module.global_rank} for batch_idx {batch_idx}"
        )

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int,
        dataloader_idx: int
    ) -> None:
        logging.info(
            f"Reached from on_test_batch_end on global rank {pl_module.global_rank} for batch_idx {batch_idx}"
        )

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        logging.info(
            f"Reached from on_test_batch_start on global rank {pl_module.global_rank} for batch_idx {batch_idx}"
        )
