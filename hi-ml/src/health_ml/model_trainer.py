#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeVar

from azureml.core import Run
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profiler import BaseProfiler, SimpleProfiler, AdvancedProfiler, PyTorchProfiler

from health_azure.utils import RUN_CONTEXT, is_running_in_azure_ml

from health_ml.lightning_container import LightningContainer
from health_ml.utils import AzureMLProgressBar
from health_ml.utils.common_utils import AUTOSAVE_CHECKPOINT_FILE_NAME, EXPERIMENT_SUMMARY_FILE
from health_ml.utils.diagnostics import TrainingDiagnoticsCallback
from health_ml.utils.lightning_loggers import StoringLogger, HimlMLFlowLogger


T = TypeVar('T')


def write_experiment_summary_file(config: Any, outputs_folder: Path) -> None:
    """
    Writes the given config to disk in plain text in the default output folder.
    """
    output = str(config)
    outputs_folder.mkdir(exist_ok=True, parents=True)
    dst = outputs_folder / EXPERIMENT_SUMMARY_FILE
    dst.write_text(output)
    logging.info(output)


def get_pl_profiler(pl_profiler: Optional[str], outputs_folder: Path) -> Optional[BaseProfiler]:
    if pl_profiler:
        pl_profilers = {"simple": SimpleProfiler, "advanced": AdvancedProfiler, "pytorch": PyTorchProfiler}
        if pl_profiler not in pl_profilers:
            raise ValueError(
                "Unsupported profiler. Please choose one of the following options: simple, advanced, "
                "pytorch. You can refer to https://pytorch-lightning.readthedocs.io/en/stable/advanced/"
                "profiler.html to learn more about each profiler. You can specify a custom profiler by "
                "overriding the default behavior of get_trainer_arguments() in your lightning container. "
                "You can find an example here https://github.com/microsoft/hi-ml/tree/main/docs/source/"
                "debugging.md#L145"
            )
        profiler = pl_profilers[pl_profiler](dirpath=outputs_folder / "profiler")
        return profiler
    else:
        return None


def create_lightning_trainer(
    container: LightningContainer,
    resume_from_checkpoint: Optional[Path] = None,
    num_nodes: int = 1,
    multiple_trainloader_mode: str = "max_size_cycle",
    azureml_run_for_logging: Optional[Run] = None,
    mlflow_run_for_logging: Optional[str] = None,
) -> Tuple[Trainer, StoringLogger]:
    """
    Creates a Pytorch Lightning Trainer object for the given model configuration. It creates checkpoint handlers
    and loggers. That includes a diagnostic logger for use in unit tests, that is also returned as the second
    return value.

    :param container: The container with model and data.
    :param resume_from_checkpoint: If provided, training resumes from this checkpoint point.
    :param num_nodes: The number of nodes to use in distributed training.
    :param azureml_run_for_logging: An optional AzureML Run object to which all metrics should be logged. Use this
        argument to log to AzureML when the training is happening outside of AzureML. If `azureml_run_for_logging` is
        None and the present code is running in AzureML, the current run is used.
    :return: A tuple [Trainer object, diagnostic logger]
    """
    logging.debug(f"resume_from_checkpoint: {resume_from_checkpoint}")
    num_gpus = container.num_gpus_per_node()
    effective_num_gpus = num_gpus * num_nodes
    strategy = None
    if effective_num_gpus == 0:
        accelerator = "cpu"
        devices = 1
        message = "CPU"
    else:
        accelerator = "gpu"
        devices = num_gpus
        message = f"{devices} GPU"
        if effective_num_gpus > 1:
            # Accelerator should be "ddp" when running large models in AzureML (when using DDP_spawn, we get out of
            # GPU memory).
            # Initialize the DDP plugin. The default for pl_find_unused_parameters is False. If True, the plugin
            # prints out lengthy warnings about the performance impact of find_unused_parameters.
            strategy = DDPStrategy(
                find_unused_parameters=container.pl_find_unused_parameters,
                static_graph=container.pl_static_graph,
            )
            message += "s per node with DDP"
    logging.info(f"Using {message}")
    tensorboard_logger = TensorBoardLogger(save_dir=str(container.logs_folder), name="Lightning", version="")
    loggers: List[Any] = [tensorboard_logger]

    if is_running_in_azure_ml():
        mlflow_run_id = os.environ.get("MLFLOW_RUN_ID", None)
        logging.info(f"Logging to MLFlow run with id: {mlflow_run_id}")
        mlflow_logger = HimlMLFlowLogger(run_id=mlflow_run_id)
        loggers.append(mlflow_logger)
    else:
        mlflow_run_dir = container.outputs_folder / "mlruns"
        try:
            mlflow_run_dir.mkdir(exist_ok=True)
            mlflow_tracking_uri = "file:" + str(mlflow_run_dir)
            mlflow_logger = HimlMLFlowLogger(run_id=mlflow_run_for_logging, tracking_uri=mlflow_tracking_uri)
            loggers.append(mlflow_logger)
            logging.info(
                f"Logging to MLFlow run with id: {mlflow_run_for_logging}. Local MLFlow logs are stored in "
                f"{mlflow_tracking_uri}"
            )
        except FileNotFoundError as e:
            logging.warning(f"Unable to initialise MLFlowLogger due to error: {e}")

    storing_logger = StoringLogger()
    loggers.append(storing_logger)
    # Use 32bit precision when running on CPU. Otherwise, make it depend on use_mixed_precision flag.
    precision = 32 if num_gpus == 0 else 16 if container.use_mixed_precision else 32
    # The next two flags control the settings in torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    # https://pytorch.org/docs/stable/notes/randomness.html
    # Note that switching to deterministic models can have large performance downside.
    if container.pl_deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True

    # The last checkpoint is considered the "best" checkpoint. For large segmentation
    # models, this still appears to be the best way of choosing them because validation loss on the relatively small
    # training patches is not stable enough. Going by the validation loss somehow works for the Prostate model, but
    # not for the HeadAndNeck model.
    # Note that "last" is somehow a misnomer, it should rather be "latest". There is a "last" checkpoint written in
    # every epoch. We could use that for recovery too, but it could happen that the job gets preempted right during
    # writing that file, and we would end up with an invalid file.
    last_checkpoint_callback = ModelCheckpoint(dirpath=str(container.checkpoint_folder), save_last=True, save_top_k=0)
    recovery_checkpoint_callback = ModelCheckpoint(
        dirpath=str(container.checkpoint_folder),
        filename=AUTOSAVE_CHECKPOINT_FILE_NAME,
        every_n_epochs=container.autosave_every_n_val_epochs,
        save_last=False,
    )
    callbacks: List[Callback] = [
        last_checkpoint_callback,
        recovery_checkpoint_callback,
    ]
    if container.monitor_loading:
        # TODO antonsc: Remove after fixing the callback.
        raise NotImplementedError("Monitoring batch loading times has been temporarily disabled.")
        # callbacks.append(BatchTimeCallback())
    if container.monitor_training:
        callbacks.append(TrainingDiagnoticsCallback())
    if num_gpus > 0 and container.monitor_gpu:
        logging.info("Adding monitoring for GPU utilization")
        callbacks.append(GPUStatsMonitor(intra_step_time=True, inter_step_time=True))
    # Add the additional callbacks that were specified in get_trainer_arguments for LightningContainers
    additional_args = container.get_trainer_arguments()
    # Callbacks can be specified via the "callbacks" argument (the legacy behaviour) or the new get_callbacks method
    if "callbacks" in additional_args:
        more_callbacks = additional_args.pop("callbacks")
        if isinstance(more_callbacks, list):
            callbacks.extend(more_callbacks)  # type: ignore
        else:
            callbacks.append(more_callbacks)  # type: ignore
    callbacks.extend(container.get_callbacks())
    # Set profiler: if --pl_profiler=profiler is specified, it overrides any custom profiler defined in the container
    custom_profiler = additional_args.pop("profiler", None)
    profiler = get_pl_profiler(container.pl_profiler, container.outputs_folder)
    profiler = profiler if profiler else custom_profiler
    is_azureml_run = is_running_in_azure_ml(RUN_CONTEXT)
    progress_bar_refresh_rate = container.pl_progress_bar_refresh_rate
    if progress_bar_refresh_rate is None:
        progress_bar_refresh_rate = 50
        logging.info(
            f"The progress bar refresh rate is not set. Using a default of {progress_bar_refresh_rate}. "
            f"To change, modify the pl_progress_bar_refresh_rate field of the container."
        )
    if is_azureml_run:
        callbacks.append(
            AzureMLProgressBar(
                refresh_rate=progress_bar_refresh_rate, write_to_logging_info=True, print_timestamp=False
            )
        )
    else:
        # Use a local import here to be able to support older PL versions
        from pytorch_lightning.callbacks import TQDMProgressBar

        callbacks.append(TQDMProgressBar(refresh_rate=progress_bar_refresh_rate))
    # Read out additional model-specific args here.
    # We probably want to keep essential ones like numgpu and logging.
    trainer = Trainer(
        default_root_dir=str(container.outputs_folder),
        deterministic=deterministic,
        benchmark=benchmark,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=container.max_epochs,
        # All of the following limit_batches  arguments can be integers or floats.
        # If integers, it is the number of batches.
        # If float, it's the fraction of batches. We default to 1.0 (processing all batches).
        limit_train_batches=container.pl_limit_train_batches or 1.0,
        limit_val_batches=container.pl_limit_val_batches or 1.0,
        limit_test_batches=container.pl_limit_test_batches or 1.0,
        fast_dev_run=container.pl_fast_dev_run,  # type: ignore
        num_sanity_val_steps=container.pl_num_sanity_val_steps,
        log_every_n_steps=container.pl_log_every_n_steps,
        # check_val_every_n_epoch=container.pl_check_val_every_n_epoch,
        callbacks=callbacks,
        logger=loggers,
        num_nodes=num_nodes,
        devices=devices,
        precision=precision,
        sync_batchnorm=container.pl_sync_batchnorm,
        detect_anomaly=container.detect_anomaly,
        profiler=profiler,
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
        multiple_trainloader_mode=multiple_trainloader_mode,
        accumulate_grad_batches=container.pl_accumulate_grad_batches,
        replace_sampler_ddp=container.pl_replace_sampler_ddp,
        **additional_args,
    )
    return trainer, storing_logger
