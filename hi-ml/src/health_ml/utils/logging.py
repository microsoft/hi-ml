#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
from io import TextIOWrapper
import logging
import math
import numbers
import operator
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
from azureml.core import Run, Workspace
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict, _sanitize_params

from health_azure import is_running_in_azure_ml
from health_azure.utils import PathOrString, RUN_CONTEXT, create_aml_run_object


class AzureMLLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that stores metrics in the current AzureML run. This logger will always write metrics
    to AzureML if the training run is executed in AzureML. It can optionally also write to AzureML if the training
    run is executed somewhere else, for example on a VM outside of AzureML.
    """

    HYPERPARAMS_NAME = "hyperparams"
    """
    The name under which hyperparameters are written to the AzureML run.
    """

    def __init__(
        self,
        enable_logging_outside_azure_ml: Optional[bool] = False,
        experiment_name: str = "azureml_logger",
        run: Optional[Run] = None,
        run_name: Optional[str] = None,
        workspace: Optional[Workspace] = None,
        workspace_config_path: Optional[Path] = None,
        snapshot_directory: Optional[PathOrString] = None,
    ) -> None:
        """
        :param enable_logging_outside_azure_ml: If True, the AzureML logger will write metrics to AzureML even if
            executed outside of an AzureML run (for example, when working on a separate virtual machine). If False,
            the logger will only write metrics to AzureML if the code is actually running inside of AzureML. Default
            False, do not log outside of AzureML.
        :param experiment_name: The AzureML experiment that should hold the run when executed outside of AzureML.
        :param run: The AzureML run to log to when the ``enable_logging_outside_azure_ml`` flag is True. If None,
            a new run will be created. When finished, the run should be completed by calling ``run.complete()``. The
            logger itself only calls ``run.flush()`` in its ``finalize()`` method.
        :param run_name: An optional name for the run (this will be used as the display name in the AzureML UI). This
            argument only matters when running outside of AzureML.
        :param workspace: If provided, use this workspace to create the run in.
        :param workspace_config_path: Use this path to read workspace configuration json file. If not provided,
            use the workspace specified by the `config.json` file in the current working directory or its parents.
        :param snapshot_directory: The folder that should be included as the code snapshot. By default, no snapshot
            is created. Set this to the folder that contains all the code your experiment uses. You can use a file
            .amlignore to skip specific files or folders, akin to .gitignore..
        """
        super().__init__()
        self.is_running_in_azure_ml = is_running_in_azure_ml()
        self.run: Optional[Run] = None
        self.has_user_provided_run = False
        self.enable_logging_outside_azure_ml = enable_logging_outside_azure_ml
        if self.is_running_in_azure_ml:
            self.run = RUN_CONTEXT
        elif enable_logging_outside_azure_ml:
            if run is not None:
                self.run = run
                self.has_user_provided_run = True
            else:
                try:
                    self.run = create_aml_run_object(
                        experiment_name=experiment_name,
                        run_name=run_name,
                        workspace=workspace,
                        workspace_config_path=workspace_config_path,
                        snapshot_directory=snapshot_directory,
                    )
                    # Display name should already be set when creating the run object, but this does not happen.
                    # In unit tests, the run has the expected display name, but not here. Hence, set it again.
                    self.run.display_name = run_name
                except Exception as ex:
                    logging.error(f"Unable to create an AzureML run to store the results because of {ex}.")
                    raise
            print(f"Writing metrics to run {self.run.id} in experiment {self.run.experiment.name}.")
            print(f"To check progress, visit this URL: {self.run.get_portal_url()}")
        else:
            print(
                "AzureMLLogger will not write any logs because it is running outside AzureML, and the "
                "'enable_logging_outside_azure_ml' flag is set to False"
            )

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Writes the given metrics dictionary to the AzureML run context. If the metrics dictionary has an `epoch` key,
        the `step` value (x-axis for plots) is left empty. If there is no `epoch` key, the `step` value is taken
        from the function argument. This is the case for metrics that are logged with the `on_step=True` flag.

        :param metrics: A dictionary with metrics to log. Keys are strings, values are floating point numbers.
        :param step: The trainer global step for logging.
        """
        logging.debug(f"AzureMLLogger step={step}: {metrics}")
        if self.run is None:
            return
        is_epoch_metric = "epoch" in metrics
        for key, value in metrics.items():
            # Log all epoch-level metrics without the step information
            # All step-level metrics with step
            self.run.log(key, value, step=None if is_epoch_metric else step)

    @rank_zero_only
    def log_hyperparams(self, params: Union[argparse.Namespace, Dict[str, Any]]) -> None:
        """
        Logs the given model hyperparameters to AzureML as a table. Namespaces are converted to dictionaries.
        Nested dictionaries are flattened out. The hyperparameters are then written as a table with two columns
        "name" and "value".
        """
        if self.run is None:
            return
        if params is None:
            return
        params_final = _preprocess_hyperparams(params)
        if len(params_final) > 0:
            # Log hyperparameters as a table with 2 columns. Each "step" is one hyperparameter
            self.run.log_table(
                self.HYPERPARAMS_NAME, {"name": list(params_final.keys()), "value": list(params_final.values())}
            )

    def experiment(self) -> Any:
        return None

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0

    def finalize(self, status: str) -> None:
        if self.enable_logging_outside_azure_ml and not self.is_running_in_azure_ml and self.run is not None:
            if self.has_user_provided_run:
                # The logger uses a run that was provided by the user: Flush it, but do not complete it.
                # The user should complete the run after finishing the experiment. This is important when running
                # training outside of AzureML, so that training and inference metrics can be written to the same run.
                self.run.flush()
            else:
                # Run.complete should only be called if we created an AzureML run here in the constructor.
                self.run.complete()


class AzureMLProgressBar(ProgressBarBase):
    """
    A PL progress bar that works better in AzureML. It prints timestamps for each message, and works well with a setup
    where there is no direct access to the console.

    Usage example:
        >>> from health_ml.utils import AzureMLProgressBar
        >>> from pytorch_lightning import Trainer
        >>> progress = AzureMLProgressBar(refresh_rate=100)
        >>> trainer = Trainer(callbacks=[progress])
    """

    PROGRESS_STAGE_TRAIN = "Training"
    """A string that indicates that the trainer loop is presently in training mode."""
    PROGRESS_STAGE_VAL = "Validation"
    """A string that indicates that the trainer loop is presently in validation mode."""
    PROGRESS_STAGE_TEST = "Testing"
    """A string that indicates that the trainer loop is presently in testing mode."""
    PROGRESS_STAGE_PREDICT = "Prediction"
    """A string that indicates that the trainer loop is presently in prediction mode."""

    def __init__(self, refresh_rate: int = 50, print_timestamp: bool = True, write_to_logging_info: bool = False):
        """
        :param refresh_rate: The number of steps after which the progress should be printed out.
        :param print_timestamp: If True, each message that the progress bar prints will be prefixed with the current
            time in UTC. If False, no such prefix will be added.
        :param write_to_logging_info: If True, the progress information will be printed via logging.info. If False,
            it will be printed to stdout via print.
        """
        super().__init__()
        self._refresh_rate = refresh_rate
        self._enabled = True
        self.stage = ""
        self.stage_start_time = 0.0
        self.total_num_batches = 0.0
        self.write_to_logging_info = write_to_logging_info
        self.print_timestamp = print_timestamp

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    @property
    def total_test_batches(self) -> int:
        assert self._trainer is not None
        return sum(self.trainer.num_test_batches)  # type: ignore

    @property
    def total_predict_batches(self) -> int:
        assert self._trainer is not None
        return sum(self.trainer.num_predict_batches)  # type: ignore

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.start_stage(self.PROGRESS_STAGE_TRAIN, self.total_train_batches)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.start_stage(self.PROGRESS_STAGE_VAL, self.total_val_batches)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.start_stage(self.PROGRESS_STAGE_TEST, self.total_test_batches)

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_predict_epoch_start(trainer, pl_module)
        self.start_stage(self.PROGRESS_STAGE_PREDICT, self.total_predict_batches)

    def start_stage(self, stage: str, total_num_batches: Union[int, float]) -> None:
        """
        Sets the information that a new stage of the PL loop is starting. The stage will be available in
        self.stage, total_num_batches in self.total_num_batches. The time when this method was called is recorded in
        self.stage_start_time

        :param stage: The string name of the stage that has just started.
        :param total_num_batches: The total number of batches that need to be processed in this stage. This is used
            only for progress reporting.
        """
        self.stage = stage
        self.total_num_batches = total_num_batches
        self.stage_start_time = time.time()

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        super().on_train_batch_end(*args, **kwargs)
        self.update_progress(batches_processed=self.train_batch_idx)

    def on_validation_batch_end(self, *args: Any, **kwargs: Any) -> None:
        super().on_validation_batch_end(*args, **kwargs)
        self.update_progress(batches_processed=self.val_batch_idx)

    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        super().on_test_batch_end(*args, **kwargs)
        self.update_progress(batches_processed=self.test_batch_idx)

    def on_predict_batch_end(self, *args: Any, **kwargs: Any) -> None:
        super().on_predict_batch_end(*args, **kwargs)
        self.update_progress(batches_processed=self.predict_batch_idx)

    def update_progress(self, batches_processed: int) -> None:
        """
        Writes progress information once the refresh interval is full.

        :param batches_processed: The number of batches that have been processed for the current stage.
        """

        def to_minutes(time_sec: float) -> str:
            minutes = int(time_sec / 60)
            seconds = int(time_sec % 60)
            return f"{minutes:02}:{seconds:02}"

        should_update = self.is_enabled and (
            batches_processed % self.refresh_rate == 0 or batches_processed == self.total_num_batches
        )  # noqa: W503
        if not should_update:
            return
        prefix = f"{self.stage}"
        assert self.trainer is not None and self.trainer.lightning_module is not None  # for pyright
        if self.stage in [self.PROGRESS_STAGE_TRAIN, self.PROGRESS_STAGE_VAL]:
            prefix += f" epoch {self.trainer.current_epoch}"
        if self.stage == self.PROGRESS_STAGE_TRAIN:
            prefix += f" (step {self.trainer.lightning_module.global_step})"
        prefix += ": "
        time_elapsed = time.time() - self.stage_start_time
        time_elapsed_min = to_minutes(time_elapsed)
        if math.isinf(self.total_num_batches):
            # Can't print out per-cent progress or time estimates if the data is infinite
            message = f"{prefix}{batches_processed:4} batches completed, {time_elapsed_min} since epoch start"
        else:
            fraction_completed = batches_processed / self.total_num_batches
            percent_completed = int(fraction_completed * 100)
            estimated_epoch_duration = time_elapsed / fraction_completed
            message = (
                f"{prefix}{batches_processed:4}/{self.total_num_batches} ({percent_completed:3}%) completed. "
                f"{time_elapsed_min} since epoch start, estimated total epoch time ~ "
                f"{to_minutes(estimated_epoch_duration)}"
            )
        self._print(message)

    def _print(self, message: str) -> None:
        if self.print_timestamp:
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ ")
            message = timestamp + message
        if self.write_to_logging_info:
            logging.info(message)
        else:
            print(message)
            sys.stdout.flush()


def _preprocess_hyperparams(params: Any) -> Dict[str, str]:
    """
    Converts arbitrary hyperparameters to a simple dictionary structure, in particular argparse Namespaces.
    Nested dictionaries are converted to folder-like strings, like ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    All hyperparameter values are converted to strings, because Run.log_table can't deal with mixed datatypes.
    :param params: The parameters to convert
    :return: A dictionary mapping from string to string.
    """
    # Convert from Namespace to dictionary
    params = _convert_params(params)
    # Convert nested dictionaries to folder-like structure
    params = _flatten_dict(params)
    # Convert anything that is not a primitive type to str
    params_final = _sanitize_params(params)
    if not isinstance(params_final, dict):
        raise ValueError(f"Expected the converted hyperparameters to be a dictionary, but got {type(params)}")
    return {str(key): str(value) for key, value in params_final.items()}


def log_on_epoch(
    module: LightningModule,
    name: Optional[str] = None,
    value: Optional[Any] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    reduce_fx: Callable = torch.mean,
    sync_dist: Optional[bool] = None,
) -> None:
    """
    Write a dictionary with metrics and/or an individual metric as a name/value pair to the loggers of the given module.
    Metrics are always logged upon epoch completion.
    The metrics in question first synchronized across GPUs if DDP with >1 node is used,. Afterwards, they are
    aggregated across all steps via the reduce_fx (default: mean).
    Metrics that are fed in as plain numbers rather than tensors (for example, plain Python integers) are converted
    to tensors before logging, to enable synchronization across GPUs if needed.

    :param name: The name of the metric to log.
    :param value: The actual value of the metric to log.
    :param metrics: A dictionary with metrics to log.
    :param module: The PyTorch Lightning module where the metrics should be logged.
    :param sync_dist: If not None, use this value for the sync_dist argument to module.log. If None,
        set it automatically depending on the use of DDP. Set this to False if you want to log metrics that are only
        available on Rank 0 of a DDP job.
    :param reduce_fx: The reduce function to apply to the per-step values, after synchronizing the tensors across GPUs.
        Default: torch.mean
    """
    assert module.trainer is not None, "No trainer is set for this module."
    if operator.xor(name is None, value is None):
        raise ValueError("Both or neither of 'name' and 'value' must be provided.")
    is_sync_dist = module.trainer.world_size > 1 if sync_dist is None else sync_dist
    metrics = metrics or {}
    if name is not None:
        metrics[name] = value  # type: ignore
    metrics_as_tensors = {
        key: torch.tensor(value, dtype=torch.float, device=module.device)
        if isinstance(value, numbers.Number)
        else value
        for key, value in metrics.items()
    }
    module.log_dict(metrics_as_tensors, on_epoch=True, on_step=False, sync_dist=is_sync_dist, reduce_fx=reduce_fx)


def log_learning_rate(module: LightningModule, name: str = "learning_rate") -> None:
    """
    Logs the learning rate(s) used by the given module. Multiple learning rate schedulers and/or multiple rates per
    scheduler are supported. The learning rates are logged under the given name. If multiple scheduler and/or multiple
    rates are used, a suffix like "/0/1" is added, to indicate the learning rate for scheduler 0, index 1, for example.
    Learning rates are logged on epoch.

    :param module: The module for which the learning rates should be logged.
    :param name: The name to use when logging the learning rates.
    """
    schedulers = module.lr_schedulers()
    if schedulers is None:
        raise ValueError("Learning rate logging can only be used during training.")
    single_scheduler = not isinstance(schedulers, list)
    if single_scheduler:
        schedulers = [schedulers]  # type: ignore
    lr_0 = schedulers[0].get_last_lr()  # type: ignore
    singleton_lr = single_scheduler and len(lr_0) == 1
    logged = {}
    for i, scheduler in enumerate(schedulers):  # type: ignore
        for j, lr_j in enumerate(scheduler.get_last_lr()):  # type: ignore
            full_name = name if singleton_lr else f"{name}/{i}/{j}"
            logged[full_name] = lr_j
    log_on_epoch(module, metrics=logged)


class ConsoleAndFileOutput(TextIOWrapper):
    """A file-like object that writes to both the console (sys.stdout) and an output file.
    The caller needs to ensure that the file is closed properly after using this object.

    :param TextIOWrapper: The file to write to.
    """

    def __init__(self, file: TextIOWrapper) -> None:
        self.file = file
        self.sys_stdout = sys.stdout

    def write(self, __buffer: Any) -> int:
        """Writes the buffer to both the file and to the console.

        :param __buffer: The buffer to write.
        :return: The number of bytes written to the console
        """
        self.file.write(__buffer)
        return self.sys_stdout.write(__buffer)

    def flush(self) -> None:
        """Flush both the output file and the console."""
        self.file.flush()
        self.sys_stdout.flush()

    def isatty(self) -> bool:
        """Returns if the given file is a terminal. Since the class is writing to a file, this always returns False."""
        return False

    def read(self, size: int = -1) -> str:  # type: ignore
        raise NotImplementedError("Read is not supported")

    def readline(self, size: int = -1) -> str:  # type: ignore
        raise NotImplementedError("Readline is not supported")
