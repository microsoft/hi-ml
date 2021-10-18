#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import numbers
import operator
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only

from health_azure import is_running_in_azure_ml
from health_azure.utils import RUN_CONTEXT


class AzureMLLogger(LightningLoggerBase):
    """
    A Pytorch Lightning logger that stores metrics in the current AzureML run. If the present run is not
    inside AzureML, nothing gets logged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.is_running_in_azure_ml = is_running_in_azure_ml()

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        logging.debug(f"AzureMLLogger step={step}: {metrics}")
        if self.is_running_in_azure_ml:
            for key, value in metrics.items():
                RUN_CONTEXT.log(key, value)

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        pass

    def experiment(self) -> Any:
        return None

    def name(self) -> Any:
        return ""

    def version(self) -> int:
        return 0


def log_on_epoch(module: LightningModule,
                 name: Optional[str] = None,
                 value: Optional[Any] = None,
                 metrics: Optional[Mapping[str, Any]] = None,
                 reduce_fx: Callable = torch.mean,
                 sync_dist: Optional[bool] = None,
                 sync_dist_op: Any = "mean") -> None:
    """
    Write a dictionary with metrics and/or an individual metric as a name/value pair to the loggers of the given module.
    Metrics are always logged upon epoch completion.
    The metrics in question first synchronized across GPUs if DDP with >1 node is used, using the sync_dist_op
    (default: mean). Afterwards, they are aggregated across all steps via the reduce_fx (default: mean).
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
    :param sync_dist_op: The reduce operation to use when synchronizing the tensors across GPUs. This must be
    a value recognized by sync_ddp: 'sum', 'mean', 'avg'
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
    module.log_dict(metrics_as_tensors,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=is_sync_dist,
                    reduce_fx=reduce_fx,
                    sync_dist_op=sync_dist_op)


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
        schedulers = [schedulers]
    lr_0 = schedulers[0].get_last_lr()  # type: ignore
    singleton_lr = single_scheduler and len(lr_0) == 1
    logged = {}
    for i, scheduler in enumerate(schedulers):
        for j, lr_j in enumerate(scheduler.get_last_lr()):  # type: ignore
            full_name = name if singleton_lr else f"{name}/{i}/{j}"
            logged[full_name] = lr_j
    log_on_epoch(module, metrics=logged)
